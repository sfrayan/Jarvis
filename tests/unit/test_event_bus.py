"""Tests unitaires du bus d'événements."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from pydantic import BaseModel

from core.event_bus import EventBus, SubscriptionHandle

pytestmark = pytest.mark.unit


# --- Événements de test ----------------------------------------------
class _EventA(BaseModel):
    payload: str


class _EventB(BaseModel):
    number: int


# ---------------------------------------------------------------------
# Publication basique
# ---------------------------------------------------------------------
class TestBasicPublish:
    @pytest.mark.asyncio
    async def test_single_subscriber_receives_event(self) -> None:
        bus = EventBus()
        received: list[_EventA] = []

        async def handler(event: _EventA) -> None:
            received.append(event)

        bus.subscribe(_EventA, handler)
        await bus.publish(_EventA(payload="hello"))

        assert len(received) == 1
        assert received[0].payload == "hello"

    @pytest.mark.asyncio
    async def test_multiple_subscribers_all_receive(self) -> None:
        bus = EventBus()
        received1: list[_EventA] = []
        received2: list[_EventA] = []

        async def h1(event: _EventA) -> None:
            received1.append(event)

        async def h2(event: _EventA) -> None:
            received2.append(event)

        bus.subscribe(_EventA, h1)
        bus.subscribe(_EventA, h2)
        await bus.publish(_EventA(payload="broadcast"))

        assert len(received1) == 1
        assert len(received2) == 1

    @pytest.mark.asyncio
    async def test_publish_without_subscribers_is_noop(self) -> None:
        bus = EventBus()
        # Ne doit rien lever
        await bus.publish(_EventA(payload="lonely"))
        assert bus.subscriber_count() == 0

    @pytest.mark.asyncio
    async def test_handlers_isolated_by_event_type(self) -> None:
        bus = EventBus()
        received_a: list[_EventA] = []
        received_b: list[_EventB] = []

        async def ha(event: _EventA) -> None:
            received_a.append(event)

        async def hb(event: _EventB) -> None:
            received_b.append(event)

        bus.subscribe(_EventA, ha)
        bus.subscribe(_EventB, hb)

        await bus.publish(_EventA(payload="a"))
        await bus.publish(_EventB(number=42))

        assert len(received_a) == 1
        assert len(received_b) == 1
        assert received_a[0].payload == "a"
        assert received_b[0].number == 42


# ---------------------------------------------------------------------
# Désabonnement
# ---------------------------------------------------------------------
class TestUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsubscribed_handler_does_not_receive(self) -> None:
        bus = EventBus()
        received: list[_EventA] = []

        async def handler(event: _EventA) -> None:
            received.append(event)

        handle = bus.subscribe(_EventA, handler)
        await bus.publish(_EventA(payload="first"))
        handle.unsubscribe()
        await bus.publish(_EventA(payload="second"))

        assert len(received) == 1
        assert received[0].payload == "first"

    @pytest.mark.asyncio
    async def test_unsubscribe_idempotent(self) -> None:
        bus = EventBus()

        async def handler(event: _EventA) -> None:
            pass

        handle = bus.subscribe(_EventA, handler)
        assert handle.active is True
        handle.unsubscribe()
        handle.unsubscribe()  # second appel ne doit pas lever
        assert handle.active is False

    @pytest.mark.asyncio
    async def test_context_manager_auto_unsubscribes(self) -> None:
        bus = EventBus()
        received: list[_EventA] = []

        async def handler(event: _EventA) -> None:
            received.append(event)

        with bus.subscribe(_EventA, handler):
            await bus.publish(_EventA(payload="inside"))
        await bus.publish(_EventA(payload="outside"))

        assert len(received) == 1
        assert received[0].payload == "inside"

    @pytest.mark.asyncio
    async def test_subscriber_count_tracks_state(self) -> None:
        bus = EventBus()

        async def h(event: _EventA) -> None:
            pass

        assert bus.subscriber_count(_EventA) == 0

        handle1 = bus.subscribe(_EventA, h)
        handle2 = bus.subscribe(_EventA, h)
        assert bus.subscriber_count(_EventA) == 2
        assert bus.subscriber_count() == 2

        handle1.unsubscribe()
        assert bus.subscriber_count(_EventA) == 1

        handle2.unsubscribe()
        assert bus.subscriber_count(_EventA) == 0
        assert bus.subscriber_count() == 0


# ---------------------------------------------------------------------
# Isolation des exceptions
# ---------------------------------------------------------------------
class TestExceptionIsolation:
    @pytest.mark.asyncio
    async def test_handler_exception_does_not_block_others(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def failing(event: _EventA) -> None:
            raise RuntimeError("boom")

        async def ok(event: _EventA) -> None:
            received.append(event.payload)

        bus.subscribe(_EventA, failing)
        bus.subscribe(_EventA, ok)

        # Ne doit PAS lever — les exceptions sont capturées
        await bus.publish(_EventA(payload="survive"))

        assert received == ["survive"]


# ---------------------------------------------------------------------
# Parallélisme
# ---------------------------------------------------------------------
class TestConcurrency:
    @pytest.mark.asyncio
    async def test_handlers_run_in_parallel(self) -> None:
        """Deux handlers de 50 ms chacun doivent finir en ~50 ms total,
        pas 100 ms — preuve qu'ils sont lancés concurremment."""
        bus = EventBus()

        async def slow(event: _EventA) -> None:
            await asyncio.sleep(0.05)

        bus.subscribe(_EventA, slow)
        bus.subscribe(_EventA, slow)
        bus.subscribe(_EventA, slow)

        start = time.monotonic()
        await bus.publish(_EventA(payload="parallel"))
        elapsed = time.monotonic() - start

        # Tolerance : < 150 ms (3x50ms=150 si sequentiel, ~55ms si parallele)
        assert elapsed < 0.12, f"publish a pris {elapsed*1000:.0f} ms"


# ---------------------------------------------------------------------
# publish_threadsafe depuis un thread externe
# ---------------------------------------------------------------------
class TestThreadsafePublish:
    @pytest.mark.asyncio
    async def test_publish_from_another_thread(self) -> None:
        bus = EventBus()
        received: list[_EventA] = []

        async def handler(event: _EventA) -> None:
            received.append(event)

        bus.subscribe(_EventA, handler)
        loop = asyncio.get_running_loop()

        def publisher() -> None:
            bus.publish_threadsafe(_EventA(payload="from_thread"), loop)

        thread = threading.Thread(target=publisher, daemon=True)
        thread.start()
        thread.join(timeout=1.0)

        # Laisse le temps à la task planifiée sur la loop de s'exécuter
        await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0].payload == "from_thread"


# ---------------------------------------------------------------------
# SubscriptionHandle
# ---------------------------------------------------------------------
class TestSubscriptionHandle:
    def test_returns_handle_instance(self) -> None:
        bus = EventBus()

        async def h(event: _EventA) -> None:
            pass

        handle = bus.subscribe(_EventA, h)
        assert isinstance(handle, SubscriptionHandle)
        assert handle.active
