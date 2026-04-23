"""Machine à états de la boucle OODA de Jarvis.

Les transitions autorisées suivent le schéma du master prompt §5 :

    IDLE → LISTENING → TRANSCRIBING → ROUTING → {CHAT_ANSWER | SCREENSHOT}
                                                       ↓              ↓
                                                       ↓          VISION → ACTING → VERIFYING
                                                       ↓                                 ↓
                                                    SPEAKING ←──────────────────────────┘
                                                       ↓
                                                    IDLE (ou LISTENING pour barge-in)

EMERGENCY_STOP est atteignable **depuis n'importe quel état autre que lui-même**,
et ne revient qu'à IDLE.

Contrat :

- `transition(new_state, reason=...)` : seul point d'entrée pour muter l'état.
  Valide la transition, met à jour l'état, publie `StateTransition` sur le bus.
- `InvalidTransitionError` est levée si la transition n'est pas permise.
- Une transition `state == self.state` est un no-op silencieux.
- Thread-safe : lock interne pour la lecture/écriture de l'état courant.
"""

from __future__ import annotations

import enum
import threading
import time

from pydantic import BaseModel, ConfigDict

from core.event_bus import EventBus
from observability.logger import get_logger

log = get_logger(__name__)


class State(enum.Enum):
    """États de la boucle OODA."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    ROUTING = "routing"
    CHAT_ANSWER = "chat_answer"
    SCREENSHOT = "screenshot"
    VISION = "vision"
    ACTING = "acting"
    VERIFYING = "verifying"
    SPEAKING = "speaking"
    EMERGENCY_STOP = "emergency_stop"


class InvalidTransitionError(RuntimeError):
    """Levée quand une transition d'état non autorisée est demandée."""


# Transitions autorisées, hors EMERGENCY_STOP (ajouté plus bas à tous les états).
_BASE_TRANSITIONS: dict[State, frozenset[State]] = {
    State.IDLE: frozenset({State.LISTENING}),
    State.LISTENING: frozenset({State.TRANSCRIBING, State.IDLE}),
    State.TRANSCRIBING: frozenset({State.ROUTING, State.IDLE}),
    State.ROUTING: frozenset({State.CHAT_ANSWER, State.SCREENSHOT}),
    State.CHAT_ANSWER: frozenset({State.SPEAKING}),
    State.SCREENSHOT: frozenset({State.VISION}),
    State.VISION: frozenset({State.ACTING}),
    State.ACTING: frozenset({State.VERIFYING}),
    # VERIFYING : SPEAKING si task_complete, SCREENSHOT si retry (iter<5)
    State.VERIFYING: frozenset({State.SPEAKING, State.SCREENSHOT}),
    # SPEAKING → LISTENING pour le barge-in (voix détectée pendant TTS)
    State.SPEAKING: frozenset({State.IDLE, State.LISTENING}),
    State.EMERGENCY_STOP: frozenset({State.IDLE}),
}


def _compute_transitions() -> dict[State, frozenset[State]]:
    """Injecte EMERGENCY_STOP comme cible depuis tout état non-EMERGENCY."""
    result: dict[State, frozenset[State]] = {}
    for state, targets in _BASE_TRANSITIONS.items():
        if state is State.EMERGENCY_STOP:
            result[state] = targets
        else:
            result[state] = targets | {State.EMERGENCY_STOP}
    return result


_ALLOWED_TRANSITIONS: dict[State, frozenset[State]] = _compute_transitions()


class StateTransition(BaseModel):
    """Événement publié sur l'event bus à chaque transition validée."""

    model_config = ConfigDict(frozen=True)

    from_state: State
    to_state: State
    timestamp: float
    reason: str | None = None


class StateMachine:
    """FSM de Jarvis. Seul `transition()` mute l'état."""

    def __init__(
        self,
        event_bus: EventBus,
        initial: State = State.IDLE,
    ) -> None:
        self._state = initial
        self._bus = event_bus
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        return self._state

    @staticmethod
    def allowed_from(state: State) -> frozenset[State]:
        """États atteignables depuis `state`."""
        return _ALLOWED_TRANSITIONS.get(state, frozenset())

    async def transition(self, new_state: State, reason: str | None = None) -> None:
        """Transitionne vers `new_state`.

        - Si `new_state == self.state` : no-op silencieux.
        - Sinon valide contre `_ALLOWED_TRANSITIONS`, met à jour l'état sous
          lock, puis publie `StateTransition` sur le bus (hors lock).

        Raises:
            InvalidTransitionError: transition non autorisée depuis l'état courant.
        """
        with self._lock:
            current = self._state
            if new_state is current:
                return
            allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
            if new_state not in allowed:
                raise InvalidTransitionError(
                    f"Transition invalide : {current.value} → {new_state.value}. "
                    f"Autorisées depuis {current.value} : "
                    f"{sorted(s.value for s in allowed)}"
                )
            self._state = new_state

        log.info(
            "state_transition",
            from_state=current.value,
            to_state=new_state.value,
            reason=reason,
        )
        await self._bus.publish(
            StateTransition(
                from_state=current,
                to_state=new_state,
                timestamp=time.time(),
                reason=reason,
            )
        )
