"""Service de mediation entre Brain et Hands.

`DialogueService` applique la decision du `DialogueManager` :
- question ou plan : publie les evenements de dialogue et une phrase assistant ;
- intention claire : republie `IntentRouted` pour que Hands continue le pipeline ;
- demande incomplete : ne laisse rien partir vers Hands.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from brain.dialogue import DialogueManager, DialogueTurn
from brain.events import IntentRouted
from core.event_bus import EventBus
from core.state_machine import State, StateMachine
from observability.logger import get_logger
from voice.feedback import AssistantUtterance

log = get_logger(__name__)


class DialogueManagerLike(Protocol):
    """Contrat minimal du manager de dialogue."""

    def handle(self, routed: IntentRouted) -> DialogueTurn:
        """Retourne la decision a appliquer pour l'intention routee."""


class DialogueService:
    """Applique les decisions dialogue sur l'EventBus."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        state_machine: StateMachine,
        manager: DialogueManagerLike | None = None,
    ) -> None:
        self._bus = event_bus
        self._sm = state_machine
        self._manager = manager or DialogueManager()

    async def process(self, routed: IntentRouted) -> None:
        """Traite une intention routee et publie les sorties adaptees."""
        turn = self._manager.handle(routed)

        await self._publish_optional(turn.session_event)
        await self._publish_optional(turn.clarification)
        await self._publish_optional(turn.plan)
        await self._publish_optional(turn.draft)

        if turn.intent is not None:
            await self._publish_intent(turn.intent, reason=turn.reason)
            return

        if turn.utterance is not None:
            await self._publish_utterance(turn.utterance, reason=turn.reason)
            return

        await self._transition_if_allowed(State.IDLE, reason="dialogue_noop")

    async def _publish_intent(self, event: IntentRouted, *, reason: str) -> None:
        await self._bus.publish(event)
        log.info(
            "dialogue_intent_relayed",
            intent=event.intent,
            domain=event.domain,
            reason=reason,
        )
        if self._sm.state is State.ROUTING:
            await self._transition_if_allowed(State.IDLE, reason="dialogue_intent_relayed")

    async def _publish_utterance(self, utterance: AssistantUtterance, *, reason: str) -> None:
        await self._transition_if_allowed(State.CHAT_ANSWER, reason=reason)
        await self._transition_if_allowed(State.SPEAKING, reason=reason)
        await self._bus.publish(utterance)
        log.info(
            "dialogue_utterance_published",
            text=utterance.text,
            priority=utterance.priority,
            reason=reason,
        )
        await self._transition_if_allowed(State.IDLE, reason="dialogue_utterance_spoken")

    async def _publish_optional(self, event: BaseModel | None) -> None:
        if event is not None:
            await self._bus.publish(event)

    async def _transition_if_allowed(self, target: State, *, reason: str) -> bool:
        current = self._sm.state
        if target in StateMachine.allowed_from(current):
            await self._sm.transition(target, reason=reason)
            return True
        log.debug(
            "dialogue_transition_skipped",
            from_state=current.value,
            to_state=target.value,
            reason=reason,
        )
        return False
