"""Boucle OODA (Observe-Orient-Decide-Act) de Jarvis.

En Itération 2, la loop est volontairement **minimale** :

- Démarre, loggue l'état initial.
- Attend simultanément :
    - le déclenchement du `KillSwitch` (shutdown externe)
    - l'appel de `stop()` (shutdown programmatique propre)
- Si le kill switch a gagné : transitionne la state machine en `EMERGENCY_STOP`.
- Si `stop()` a gagné : sort proprement sans forcer l'état.
- Nettoie les tâches en attente, puis quitte.

Les itérations 3+ enrichiront cette boucle avec les consommateurs d'événements
réels (VAD → transition LISTENING, whisper → ROUTING, vision → ACTING, etc.).
Pour l'instant, l'event bus tourne à vide côté producteurs : la structure est
prête à recevoir les abonnements.
"""

from __future__ import annotations

import asyncio

from core.event_bus import EventBus
from core.state_machine import InvalidTransitionError, State, StateMachine
from observability.logger import get_logger
from safety.kill_switch import KillSwitch

log = get_logger(__name__)


class OODALoop:
    """Orchestrateur async. Une instance = un run."""

    def __init__(
        self,
        event_bus: EventBus,
        state_machine: StateMachine,
        kill_switch: KillSwitch,
    ) -> None:
        self._bus = event_bus
        self._sm = state_machine
        self._kill_switch = kill_switch
        self._stop_event = asyncio.Event()
        self._running: bool = False

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        """Demande l'arrêt propre de la loop (sans passer par EMERGENCY_STOP).

        Appelable avant ou pendant `run()`. Idempotent.
        """
        self._stop_event.set()

    async def run(self) -> None:
        """Boucle principale : bloque jusqu'à kill switch OU `stop()`.

        Raises:
            RuntimeError: si `run()` est rappelée alors que la loop tourne déjà.
        """
        if self._running:
            raise RuntimeError("La loop est déjà en cours d'exécution")

        self._running = True
        log.info("ooda_loop_started", initial_state=self._sm.state.value)

        kill_task = asyncio.create_task(
            self._kill_switch.wait_async(),
            name="ooda_wait_kill_switch",
        )
        stop_task = asyncio.create_task(
            self._stop_event.wait(),
            name="ooda_wait_stop",
        )

        try:
            done, pending = await asyncio.wait(
                {kill_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Annule et consomme les tâches non terminées pour éviter
            # les warnings "Task was destroyed but it is pending!".
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Route selon qui a déclenché la sortie.
            if kill_task in done:
                reason = self._kill_switch.reason or "kill_switch"
                log.critical(
                    "ooda_loop_shutdown",
                    source="kill_switch",
                    reason=reason,
                )
                await self._transition_to_emergency_stop(reason)
            else:
                log.info(
                    "ooda_loop_shutdown",
                    source="programmatic_stop",
                    reason="stop() invoqué",
                )
        finally:
            self._running = False
            log.info("ooda_loop_stopped", final_state=self._sm.state.value)

    async def _transition_to_emergency_stop(self, reason: str) -> None:
        """Passe la state machine en EMERGENCY_STOP si ce n'est pas déjà le cas.

        En pratique EMERGENCY_STOP est atteignable depuis tous les états non-
        EMERGENCY (cf. `state_machine._ALLOWED_TRANSITIONS`), donc une erreur
        ici signalerait un problème structurel — on la loggue mais on n'échoue
        pas le shutdown pour autant.
        """
        if self._sm.state is State.EMERGENCY_STOP:
            return
        try:
            await self._sm.transition(State.EMERGENCY_STOP, reason=reason)
        except InvalidTransitionError as exc:
            log.error(
                "cannot_transition_to_emergency_stop",
                current_state=self._sm.state.value,
                error=str(exc),
            )
