"""Tests unitaires des evenements `brain/`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from brain.events import IntentDomain, IntentRouted
from core.event_bus import EventBus

pytestmark = pytest.mark.unit


def _mk_event(**overrides: object) -> IntentRouted:
    defaults: dict[str, object] = {
        "timestamp": 1.0,
        "original_text": "ouvre Chrome",
        "normalized_text": "ouvre chrome",
        "intent": "gui",
        "domain": "apps",
        "confidence": 0.91,
        "reason": "Demande d'ouverture d'application",
        "model": "qwen3:latest",
    }
    defaults.update(overrides)
    return IntentRouted(**defaults)  # type: ignore[arg-type]


class TestIntentRouted:
    def test_construct_valid_gui_intent(self) -> None:
        event = _mk_event()
        assert event.intent == "gui"
        assert event.domain == "apps"
        assert event.confidence == pytest.approx(0.91)
        assert event.normalized_text == "ouvre chrome"

    @pytest.mark.parametrize("intent", ["chat", "gui", "unknown"])
    def test_all_supported_intents(self, intent: str) -> None:
        event = _mk_event(intent=intent)
        assert event.intent == intent

    @pytest.mark.parametrize(
        "domain",
        [
            "general",
            "system",
            "apps",
            "folders",
            "media",
            "home_assistant",
            "vision",
            "memory",
            "web_search",
            "google_workspace",
            "routine",
            "unknown",
        ],
    )
    def test_all_supported_domains(self, domain: IntentDomain) -> None:
        event = _mk_event(domain=domain)
        assert event.domain == domain

    def test_domain_defaults_to_general_for_backward_compatibility(self) -> None:
        event = IntentRouted(
            timestamp=1.0,
            original_text="bonjour",
            normalized_text="bonjour",
            intent="chat",
            confidence=0.8,
            reason="Conversation",
            model="qwen3:latest",
        )
        assert event.domain == "general"

    @pytest.mark.parametrize("bad_confidence", [-0.1, 1.1])
    def test_confidence_bounds_rejected(self, bad_confidence: float) -> None:
        with pytest.raises(ValidationError):
            _mk_event(confidence=bad_confidence)

    def test_invalid_intent_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _mk_event(intent="action")

    def test_invalid_domain_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _mk_event(domain="calendar")

    @pytest.mark.parametrize(
        "field",
        ["original_text", "normalized_text", "reason", "model"],
    )
    def test_empty_required_text_fields_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            _mk_event(**{field: ""})

    def test_negative_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _mk_event(timestamp=-1.0)

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IntentRouted(
                timestamp=1.0,
                original_text="bonjour",
                normalized_text="bonjour",
                intent="chat",
                domain="general",
                confidence=0.8,
                reason="Conversation",
                model="qwen3:latest",
                unexpected=True,  # type: ignore[call-arg]
            )

    def test_event_is_frozen(self) -> None:
        event = _mk_event()
        with pytest.raises(ValidationError):
            event.confidence = 0.1  # type: ignore[misc]


class TestEventBusIntegration:
    @pytest.mark.asyncio
    async def test_intent_routed_is_published_by_type(self) -> None:
        bus = EventBus()
        received: list[IntentRouted] = []

        async def handler(event: IntentRouted) -> None:
            received.append(event)

        bus.subscribe(IntentRouted, handler)
        await bus.publish(_mk_event(intent="chat", confidence=0.77))

        assert len(received) == 1
        assert received[0].intent == "chat"
        assert received[0].confidence == pytest.approx(0.77)
