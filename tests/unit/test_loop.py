"""Tests unitaires de la boucle OODA (Itération 2 : squelette minimal)."""

from __future__ import annotations

import asyncio
import time

import pytest

from config.schema import KillSwitchConfig
from core.event_bus import EventBus
from core.loop import OODALoop
from core.state_machine import State, StateMachine, StateTransition
from safety.kill_switch import KillSwitch

pytestmark = pytest.mark.unit


def _make_trio() -> tuple[EventBus, StateMachine, KillSwitch]:
    bus = EventBus()
    sm = StateMachine(bus)
    ks = KillSwitch(KillSwitchConfig())
    return bus, sm, ks


# ---------------------------------------------------------------------
# Cycle de vie
# ---------------------------------------------------------------------
class TestLifecycle:
    @pytest.mark.asyncio
    async def test_running_false_before_run(self) -> None:
        bus, sm, ks = _make_trio()
        loop = OODALoop(bus, sm, ks)
        assert loop.running is False

    @pytest.mark.asyncio
    async def test_stop_exits_loop(self) -> None:
        bus, sm, ks = _make_trio()
        loop = OODALoop(bus, sm, ks)

        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.02)
        assert loop.running is True

        loop.stop()
        await asyncio.wait_for(run_task, timeout=1.0)
        assert loop.running is False

    @pytest.mark.asyncio
    async def test_cannot_run_twice_concurrently(self) -> None:
        bus, sm, ks = _make_trio()
        loop = OODALoop(bus, sm, ks)

        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.02)

        with pytest.raises(RuntimeError, match="déjà en cours"):
            await loop.run()

        loop.stop()
        await run_task


# ---------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------
class TestKillSwitchIntegration:
    @pytest.mark.asyncio
    async def test_kill_switch_stops_loop_quickly(self) -> None:
        bus, sm, ks = _make_trio()
        loop = OODALoop(bus, sm, ks)

        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.02)

        start = time.monotonic()
        ks.trigger("test_kill")
        await asyncio.wait_for(run_task, timeout=1.0)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert loop.running is False
        assert elapsed_ms < 200, f"arret trop lent: {elapsed_ms:.0f} ms"

    @pytest.mark.asyncio
    async def test_kill_switch_transitions_to_emergency_stop(self) -> None:
        bus, sm, ks = _make_trio()
        loop = OODALoop(bus, sm, ks)
        assert sm.state is State.IDLE

        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.02)
        ks.trigger("test_emergency")
        await asyncio.wait_for(run_task, timeout=1.0)

        assert sm.state is State.EMERGENCY_STOP

    @pytest.mark.asyncio
    async def test_kill_switch_publishes_state_transition_on_shutdown(self) -> None:
        bus, sm, ks = _make_trio()
        loop = OODALoop(bus, sm, ks)

        received: list[StateTransition] = []

        async def handler(evt: StateTransition) -> None:
            received.append(evt)

        bus.subscribe(StateTransition, handler)

        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.02)
        ks.trigger("publish_test")
        await asyncio.wait_for(run_task, timeout=1.0)

        assert len(received) == 1
        assert received[0].from_state is State.IDLE
        assert received[0].to_state is State.EMERGENCY_STOP
        assert received[0].reason == "publish_test"

    @pytest.mark.asyncio
    async def test_kill_switch_from_non_idle_state(self) -> None:
        """Depuis n'importe quel état, le kill switch mène à EMERGENCY_STOP."""
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ACTING)
        ks = KillSwitch(KillSwitchConfig())
        loop = OODALoop(bus, sm, ks)

        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.02)
        ks.trigger("mid_acting")
        await asyncio.wait_for(run_task, timeout=1.0)

        assert sm.state is State.EMERGENCY_STOP


# ---------------------------------------------------------------------
# Stop programmatique
# ---------------------------------------------------------------------
class TestProgrammaticStop:
    @pytest.mark.asyncio
    async def test_stop_does_not_change_state(self) -> None:
        """`stop()` arrête proprement sans forcer EMERGENCY_STOP."""
        bus, sm, ks = _make_trio()
        loop = OODALoop(bus, sm, ks)

        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.02)
        loop.stop()
        await asyncio.wait_for(run_task, timeout=1.0)

        assert sm.state is State.IDLE  # pas d'EMERGENCY_STOP

    @pytest.mark.asyncio
    async def test_stop_before_run_makes_run_exit_immediately(self) -> None:
        bus, sm, ks = _make_trio()
        loop = OODALoop(bus, sm, ks)

        loop.stop()  # avant tout run() — ne doit pas lever

        start = time.monotonic()
        await asyncio.wait_for(loop.run(), timeout=0.5)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert loop.running is False
        assert elapsed_ms < 100, f"sortie trop lente: {elapsed_ms:.0f} ms"

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        bus, sm, ks = _make_trio()
        loop = OODALoop(bus, sm, ks)

        loop.stop()
        loop.stop()  # ne doit pas lever

        await asyncio.wait_for(loop.run(), timeout=0.5)
        assert loop.running is False
