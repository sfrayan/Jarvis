"""Tests unitaires de la machine à états de la boucle OODA."""

from __future__ import annotations

import pytest

from core.event_bus import EventBus
from core.state_machine import (
    InvalidTransitionError,
    State,
    StateMachine,
    StateTransition,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------
# État initial
# ---------------------------------------------------------------------
class TestInitialState:
    def test_default_initial_state_is_idle(self) -> None:
        sm = StateMachine(EventBus())
        assert sm.state is State.IDLE

    def test_custom_initial_state(self) -> None:
        sm = StateMachine(EventBus(), initial=State.LISTENING)
        assert sm.state is State.LISTENING


# ---------------------------------------------------------------------
# Transitions valides
# ---------------------------------------------------------------------
class TestValidTransitions:
    @pytest.mark.asyncio
    async def test_idle_to_listening(self) -> None:
        sm = StateMachine(EventBus())
        await sm.transition(State.LISTENING, reason="vad_speech")
        assert sm.state is State.LISTENING

    @pytest.mark.asyncio
    async def test_transition_publishes_event(self) -> None:
        bus = EventBus()
        received: list[StateTransition] = []

        async def handler(evt: StateTransition) -> None:
            received.append(evt)

        bus.subscribe(StateTransition, handler)
        sm = StateMachine(bus)
        await sm.transition(State.LISTENING, reason="test")

        assert len(received) == 1
        evt = received[0]
        assert evt.from_state is State.IDLE
        assert evt.to_state is State.LISTENING
        assert evt.reason == "test"
        assert evt.timestamp > 0

    @pytest.mark.asyncio
    async def test_full_ooda_flow(self) -> None:
        """Parcourt la route CHAT : idle→listening→transcribing→routing→chat_answer→speaking→idle."""
        sm = StateMachine(EventBus())
        flow = [
            State.LISTENING,
            State.TRANSCRIBING,
            State.ROUTING,
            State.CHAT_ANSWER,
            State.SPEAKING,
            State.IDLE,
        ]
        for target in flow:
            await sm.transition(target)
            assert sm.state is target

    @pytest.mark.asyncio
    async def test_full_gui_flow(self) -> None:
        """Route GUI : idle→…→screenshot→vision→acting→verifying→speaking→idle."""
        sm = StateMachine(EventBus())
        flow = [
            State.LISTENING,
            State.TRANSCRIBING,
            State.ROUTING,
            State.SCREENSHOT,
            State.VISION,
            State.ACTING,
            State.VERIFYING,
            State.SPEAKING,
            State.IDLE,
        ]
        for target in flow:
            await sm.transition(target)
        assert sm.state is State.IDLE

    @pytest.mark.asyncio
    async def test_verifying_can_loop_back_to_screenshot(self) -> None:
        """VERIFYING → SCREENSHOT est autorisé (retry si task_complete=False)."""
        sm = StateMachine(EventBus(), initial=State.VERIFYING)
        await sm.transition(State.SCREENSHOT)
        assert sm.state is State.SCREENSHOT

    @pytest.mark.asyncio
    async def test_speaking_to_listening_is_barge_in(self) -> None:
        """SPEAKING → LISTENING : barge-in (voix détectée pendant TTS)."""
        sm = StateMachine(EventBus(), initial=State.SPEAKING)
        await sm.transition(State.LISTENING, reason="barge_in")
        assert sm.state is State.LISTENING


# ---------------------------------------------------------------------
# Transitions invalides
# ---------------------------------------------------------------------
class TestInvalidTransitions:
    @pytest.mark.asyncio
    async def test_idle_to_speaking_raises(self) -> None:
        sm = StateMachine(EventBus())
        with pytest.raises(InvalidTransitionError, match="Transition invalide"):
            await sm.transition(State.SPEAKING)

    @pytest.mark.asyncio
    async def test_error_message_lists_allowed_targets(self) -> None:
        sm = StateMachine(EventBus())  # état IDLE
        with pytest.raises(InvalidTransitionError) as exc_info:
            await sm.transition(State.ACTING)
        message = str(exc_info.value)
        assert "idle" in message
        assert "listening" in message  # allowed depuis IDLE
        assert "emergency_stop" in message

    @pytest.mark.asyncio
    async def test_state_unchanged_after_invalid_transition(self) -> None:
        sm = StateMachine(EventBus())
        with pytest.raises(InvalidTransitionError):
            await sm.transition(State.SPEAKING)
        assert sm.state is State.IDLE

    @pytest.mark.asyncio
    async def test_invalid_transition_does_not_publish_event(self) -> None:
        bus = EventBus()
        received: list[StateTransition] = []

        async def handler(evt: StateTransition) -> None:
            received.append(evt)

        bus.subscribe(StateTransition, handler)
        sm = StateMachine(bus)
        with pytest.raises(InvalidTransitionError):
            await sm.transition(State.SPEAKING)
        assert received == []


# ---------------------------------------------------------------------
# No-op
# ---------------------------------------------------------------------
class TestNoOpTransition:
    @pytest.mark.asyncio
    async def test_same_state_is_silent_noop(self) -> None:
        bus = EventBus()
        received: list[StateTransition] = []

        async def handler(evt: StateTransition) -> None:
            received.append(evt)

        bus.subscribe(StateTransition, handler)
        sm = StateMachine(bus)
        # IDLE → IDLE : no-op, pas d'event publié, pas d'erreur
        await sm.transition(State.IDLE)
        assert sm.state is State.IDLE
        assert received == []


# ---------------------------------------------------------------------
# EMERGENCY_STOP accessible depuis tous les états
# ---------------------------------------------------------------------
class TestEmergencyStop:
    @pytest.mark.parametrize(
        "from_state",
        [s for s in State if s is not State.EMERGENCY_STOP],
    )
    @pytest.mark.asyncio
    async def test_emergency_stop_reachable_from_every_state(
        self, from_state: State
    ) -> None:
        sm = StateMachine(EventBus(), initial=from_state)
        await sm.transition(State.EMERGENCY_STOP, reason="kill_switch")
        assert sm.state is State.EMERGENCY_STOP

    @pytest.mark.asyncio
    async def test_emergency_stop_only_returns_to_idle(self) -> None:
        sm = StateMachine(EventBus(), initial=State.EMERGENCY_STOP)
        # EMERGENCY → IDLE : OK
        await sm.transition(State.IDLE)
        assert sm.state is State.IDLE

    @pytest.mark.asyncio
    async def test_emergency_stop_cannot_go_to_listening(self) -> None:
        sm = StateMachine(EventBus(), initial=State.EMERGENCY_STOP)
        with pytest.raises(InvalidTransitionError):
            await sm.transition(State.LISTENING)


# ---------------------------------------------------------------------
# allowed_from
# ---------------------------------------------------------------------
class TestAllowedFrom:
    def test_idle_allows_listening_and_emergency(self) -> None:
        allowed = StateMachine.allowed_from(State.IDLE)
        assert State.LISTENING in allowed
        assert State.EMERGENCY_STOP in allowed

    def test_emergency_stop_only_allows_idle(self) -> None:
        allowed = StateMachine.allowed_from(State.EMERGENCY_STOP)
        assert allowed == frozenset({State.IDLE})

    def test_speaking_allows_both_idle_and_listening(self) -> None:
        allowed = StateMachine.allowed_from(State.SPEAKING)
        assert State.IDLE in allowed
        assert State.LISTENING in allowed  # barge-in
