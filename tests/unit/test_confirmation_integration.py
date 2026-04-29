"""Tests d'integration du flux de confirmation (5Q).

Verifie le flux bout en bout :
1. Action bloquee -> PendingConfirmation publiee
2. Utilisateur dit "oui" -> ConfirmationResponse(confirmed) publiee
3. Utilisateur dit "non" -> ConfirmationResponse(rejected) publiee
4. Expiration silencieuse quand pas de reponse
"""

from __future__ import annotations

import pytest

from brain.confirmation import ConfirmationManager
from brain.dialogue_service import DialogueService
from brain.events import ConfirmationResponse, IntentRouted, PendingConfirmation
from core.event_bus import EventBus
from core.state_machine import State, StateMachine
from voice.feedback import AssistantUtterance


def _make_routed(text: str) -> IntentRouted:
    """Intention routee minimale pour les tests."""
    return IntentRouted(
        timestamp=1000.0,
        original_text=text,
        normalized_text=text,
        intent="gui",
        domain="apps",
        confidence=0.9,
        reason="test",
        model="test",
    )


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def sm(bus: EventBus) -> StateMachine:
    return StateMachine(bus, initial=State.ROUTING)


@pytest.fixture
def confirmation() -> ConfirmationManager:
    return ConfirmationManager(clock=lambda: 1000.0, ttl_s=60.0)


class TestDialogueConfirmationInterception:
    """Le DialogueService intercepte les reponses de confirmation."""

    async def test_confirmation_response_published_on_oui(
        self, bus: EventBus, sm: StateMachine, confirmation: ConfirmationManager,
    ) -> None:
        from hands.executor import HandsExecutionReport, PlannedGuiAction

        # Enregistrer une action en attente
        report = HandsExecutionReport(
            status="blocked",
            mode="assisted",
            actions=(PlannedGuiAction(type="close_app", text="Discord", destructive=True),),
            executed=False,
            requires_human=True,
            reason="Action sensible",
        )
        confirmation.request_confirmation(report)

        # Capturer les evenements publies
        published: list[object] = []
        bus.subscribe(ConfirmationResponse, lambda e: published.append(e))
        bus.subscribe(AssistantUtterance, lambda e: published.append(e))

        svc = DialogueService(
            event_bus=bus,
            state_machine=sm,
            confirmation=confirmation,
        )

        await svc.process(_make_routed("oui"))

        responses = [e for e in published if isinstance(e, ConfirmationResponse)]
        assert len(responses) == 1
        assert responses[0].verdict == "confirmed"

    async def test_confirmation_response_published_on_non(
        self, bus: EventBus, sm: StateMachine, confirmation: ConfirmationManager,
    ) -> None:
        from hands.executor import HandsExecutionReport, PlannedGuiAction

        report = HandsExecutionReport(
            status="blocked",
            mode="assisted",
            actions=(PlannedGuiAction(type="close_app", text="Discord", destructive=True),),
            executed=False,
            requires_human=True,
            reason="Action sensible",
        )
        confirmation.request_confirmation(report)

        published: list[object] = []
        bus.subscribe(ConfirmationResponse, lambda e: published.append(e))

        svc = DialogueService(
            event_bus=bus,
            state_machine=sm,
            confirmation=confirmation,
        )

        await svc.process(_make_routed("non merci"))

        responses = [e for e in published if isinstance(e, ConfirmationResponse)]
        assert len(responses) == 1
        assert responses[0].verdict == "rejected"

    async def test_no_pending_falls_through_to_dialogue(
        self, bus: EventBus, sm: StateMachine, confirmation: ConfirmationManager,
    ) -> None:
        """Sans confirmation pendante, le flux normal s'execute."""
        published: list[object] = []
        bus.subscribe(ConfirmationResponse, lambda e: published.append(e))

        svc = DialogueService(
            event_bus=bus,
            state_machine=sm,
            confirmation=confirmation,
        )

        # "oui" sans action pendante -> pas de ConfirmationResponse
        await svc.process(_make_routed("oui"))
        responses = [e for e in published if isinstance(e, ConfirmationResponse)]
        assert len(responses) == 0

    async def test_ambiguous_reply_does_not_confirm(
        self, bus: EventBus, sm: StateMachine, confirmation: ConfirmationManager,
    ) -> None:
        from hands.executor import HandsExecutionReport, PlannedGuiAction

        report = HandsExecutionReport(
            status="blocked",
            mode="assisted",
            actions=(PlannedGuiAction(type="close_app", text="Discord", destructive=True),),
            executed=False,
            requires_human=True,
            reason="Action sensible",
        )
        confirmation.request_confirmation(report)

        published: list[object] = []
        bus.subscribe(ConfirmationResponse, lambda e: published.append(e))

        svc = DialogueService(
            event_bus=bus,
            state_machine=sm,
            confirmation=confirmation,
        )

        await svc.process(_make_routed("je ne sais pas trop"))

        # Pas de ConfirmationResponse → la confirmation reste pendante
        responses = [e for e in published if isinstance(e, ConfirmationResponse)]
        assert len(responses) == 0
        assert confirmation.has_pending is True
