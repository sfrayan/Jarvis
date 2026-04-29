"""Service de routage d'intention.

`BrainService` s'abonne aux transcriptions produites par `ears/`, appelle
`IntentRouter`, puis delegue au dialogue.

Le dialogue peut poser une question, proposer un plan, ou relayer une intention
claire vers Hands.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, cast

from brain.confirmation import ConfirmationManager
from brain.dialogue_service import DialogueService
from brain.events import IntentRouted
from brain.router import IntentRouter
from brain.session_store import SessionStore
from core.event_bus import EventBus, SubscriptionHandle
from core.state_machine import State, StateMachine
from ears.events import Transcription
from observability.logger import get_logger

log = get_logger(__name__)


def _is_low_value_unknown(routed: IntentRouted) -> bool:
    """Évite de polluer la console pour les fallbacks réseau attendus."""
    return (
        routed.intent == "unknown"
        and routed.confidence == 0.0
        and "Routeur indisponible" in routed.reason
    )


class IntentRouterLike(Protocol):
    """Contrat minimal du routeur utilisé par `BrainService`."""

    async def route(self, text: str) -> IntentRouted:
        """Route un texte utilisateur en intention."""


class DialogueProcessorLike(Protocol):
    """Contrat minimal du service de dialogue."""

    async def process(self, routed: IntentRouted) -> None:
        """Traite une intention routee."""


class BrainService:
    """Service reactif : Transcription -> IntentRouted -> Dialogue."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        state_machine: StateMachine,
        router: IntentRouterLike,
        dialogue: DialogueProcessorLike | None = None,
    ) -> None:
        self._bus = event_bus
        self._sm = state_machine
        self._router = router
        self._dialogue = dialogue
        self._subscription: SubscriptionHandle | None = None
        self._routing_tasks: set[asyncio.Task[None]] = set()

    @classmethod
    def create_default(
        cls,
        *,
        event_bus: EventBus,
        state_machine: StateMachine,
        router: IntentRouter | None = None,
        dialogue: DialogueProcessorLike | None = None,
        session_store: SessionStore | None = None,
        confirmation: ConfirmationManager | None = None,
    ) -> BrainService:
        """Factory utilisée par `main.py`."""
        return cls(
            event_bus=event_bus,
            state_machine=state_machine,
            router=router if router is not None else IntentRouter(),
            dialogue=dialogue
            or DialogueService(
                event_bus=event_bus,
                state_machine=state_machine,
                session_store=session_store,
                confirmation=confirmation,
            ),
        )

    def start(self) -> None:
        """S'abonne aux transcriptions. Idempotent."""
        if self._subscription is not None and self._subscription.active:
            return
        self._subscription = self._bus.subscribe(Transcription, self._on_transcription)
        log.info("brain_service_started")

    def stop(self) -> None:
        """Retire l'abonnement. Idempotent."""
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None
        self._cancel_routing_tasks()
        log.info("brain_service_stopped")

    async def wait_for_pending(self) -> None:
        """Attend les routages en cours.

        Méthode utilisée par les tests et par le shutdown contrôlé : les tâches
        sont déjà protégées par `_on_routing_task_done`, donc on consomme aussi
        les annulations ici sans masquer les logs utiles.
        """
        if not self._routing_tasks:
            return
        tasks = tuple(self._routing_tasks)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _on_transcription(self, event: Transcription) -> None:
        await self._transition_if_allowed(State.ROUTING, reason="transcription_ready")
        task = asyncio.create_task(
            self._route_and_publish(event),
            name="brain_route_transcription",
        )
        self._routing_tasks.add(task)
        task.add_done_callback(self._on_routing_task_done)

    async def _route_and_publish(self, event: Transcription) -> None:
        routed = await self._router.route(event.text)

        if self._sm.state is State.EMERGENCY_STOP:
            log.debug(
                "intent_routed_ignored_after_emergency_stop",
                text=routed.normalized_text,
                intent=routed.intent,
            )
            return

        if self._dialogue is None:
            await self._bus.publish(routed)
        else:
            await self._dialogue.process(routed)
        log_fn = log.debug if _is_low_value_unknown(routed) else log.info
        log_fn(
            "intent_routed",
            intent=routed.intent,
            confidence=round(routed.confidence, 3),
            reason=routed.reason,
        )
        if self._sm.state is State.ROUTING:
            await self._transition_if_allowed(State.IDLE, reason="dialogue_iteration_finished")

    def _on_routing_task_done(self, task: asyncio.Future[None]) -> None:
        routing_task = cast(asyncio.Task[None], task)
        self._routing_tasks.discard(routing_task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error(
                "brain_routing_task_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )

    def _cancel_routing_tasks(self) -> None:
        for task in tuple(self._routing_tasks):
            task.cancel()

    async def _transition_if_allowed(self, target: State, *, reason: str) -> None:
        current = self._sm.state
        if target in StateMachine.allowed_from(current):
            await self._sm.transition(target, reason=reason)
        else:
            log.debug(
                "brain_transition_skipped",
                from_state=current.value,
                to_state=target.value,
                reason=reason,
            )
