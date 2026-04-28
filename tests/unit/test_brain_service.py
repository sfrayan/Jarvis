"""Tests unitaires de `BrainService`."""

from __future__ import annotations

import asyncio

import pytest

from brain.events import IntentRouted, IntentType
from brain.service import BrainService
from core.event_bus import EventBus
from core.state_machine import State, StateMachine, StateTransition
from ears.events import Transcription

pytestmark = pytest.mark.unit


class _FakeRouter:
    def __init__(self, routed: IntentRouted, *, delay_s: float = 0.0) -> None:
        self.routed = routed
        self.delay_s = delay_s
        self.calls: list[str] = []

    async def route(self, text: str) -> IntentRouted:
        self.calls.append(text)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return self.routed


def _transcription(text: str = "ouvre Chrome") -> Transcription:
    return Transcription(
        timestamp=1.0,
        text=text,
        language="fr",
        language_probability=1.0,
        inference_duration_ms=100.0,
        audio_duration_ms=1000.0,
    )


def _routed(intent: IntentType = "gui") -> IntentRouted:
    return IntentRouted(
        timestamp=2.0,
        original_text="ouvre Chrome",
        normalized_text="ouvre Chrome",
        intent=intent,
        confidence=0.95,
        reason="Demande d'ouverture d'application",
        model="qwen3:latest",
    )


class TestBrainService:
    @pytest.mark.asyncio
    async def test_routes_transcription_and_publishes_intent(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.TRANSCRIBING)
        router = _FakeRouter(_routed("gui"))
        service = BrainService(event_bus=bus, state_machine=sm, router=router)
        received: list[IntentRouted] = []

        async def handler(event: IntentRouted) -> None:
            received.append(event)

        bus.subscribe(IntentRouted, handler)
        service.start()

        await bus.publish(_transcription("ouvre Chrome"))
        await service.wait_for_pending()

        assert router.calls == ["ouvre Chrome"]
        assert len(received) == 1
        assert received[0].intent == "gui"
        assert sm.state is State.IDLE

    @pytest.mark.asyncio
    async def test_transitions_transcribing_to_routing_then_idle(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.TRANSCRIBING)
        service = BrainService(
            event_bus=bus,
            state_machine=sm,
            router=_FakeRouter(_routed("chat")),
        )
        transitions: list[StateTransition] = []

        async def handler(event: StateTransition) -> None:
            transitions.append(event)

        bus.subscribe(StateTransition, handler)
        service.start()

        await bus.publish(_transcription("bonjour Jarvis"))
        await service.wait_for_pending()

        assert [(t.from_state, t.to_state) for t in transitions] == [
            (State.TRANSCRIBING, State.ROUTING),
            (State.ROUTING, State.IDLE),
        ]

    @pytest.mark.asyncio
    async def test_transcription_publish_does_not_wait_for_slow_router(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.TRANSCRIBING)
        router = _FakeRouter(_routed("chat"), delay_s=0.2)
        service = BrainService(event_bus=bus, state_machine=sm, router=router)
        service.start()

        await asyncio.wait_for(bus.publish(_transcription("salut")), timeout=0.05)

        assert sm.state.value == State.ROUTING.value
        await service.wait_for_pending()
        assert sm.state.value == State.IDLE.value

    @pytest.mark.asyncio
    async def test_routing_completion_does_not_leave_emergency_stop(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.TRANSCRIBING)
        router = _FakeRouter(_routed("chat"), delay_s=0.01)
        service = BrainService(event_bus=bus, state_machine=sm, router=router)
        received: list[IntentRouted] = []

        async def handler(event: IntentRouted) -> None:
            received.append(event)

        bus.subscribe(IntentRouted, handler)
        service.start()

        await bus.publish(_transcription("salut"))
        await sm.transition(State.EMERGENCY_STOP, reason="kill_switch")
        await service.wait_for_pending()

        assert sm.state is State.EMERGENCY_STOP
        assert received == []

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.TRANSCRIBING)
        router = _FakeRouter(_routed("gui"))
        service = BrainService(event_bus=bus, state_machine=sm, router=router)

        service.start()
        service.stop()
        await bus.publish(_transcription("ouvre Chrome"))

        assert router.calls == []
        assert sm.state is State.TRANSCRIBING

    def test_start_is_idempotent(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus)
        service = BrainService(event_bus=bus, state_machine=sm, router=_FakeRouter(_routed()))

        service.start()
        service.start()

        assert bus.subscriber_count(Transcription) == 1
