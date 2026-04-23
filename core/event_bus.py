"""Bus d'événements async interne à Jarvis.

Pattern pub/sub **par type d'événement** (Pydantic BaseModel) :

    class SpeechDetected(BaseModel):
        timestamp: float
        probability: float

    bus = EventBus()
    bus.subscribe(SpeechDetected, my_handler)
    await bus.publish(SpeechDetected(timestamp=..., probability=0.8))

Caractéristiques :

- Les handlers sont des coroutines (`async def handler(event): ...`).
- Plusieurs handlers par type sont appelés **en parallèle** via `asyncio.gather`.
- Une exception levée par un handler est capturée et loguée : les autres
  handlers ne sont **pas** impactés.
- Thread-safe : `subscribe`/`unsubscribe`/`publish_threadsafe` peuvent être
  appelés depuis n'importe quel thread (le kill switch par exemple).
- `SubscriptionHandle` sert de ticket pour retirer un handler, et fonctionne
  comme context manager pour un abonnement limité dans le temps.

Les événements eux-mêmes sont définis par les modules consommateurs (ears,
brain, hands, safety…), le bus n'impose qu'un contrat : hériter de
`pydantic.BaseModel`.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from observability.logger import get_logger

log = get_logger(__name__)

EventT = TypeVar("EventT", bound=BaseModel)
AsyncHandler = Callable[[Any], Awaitable[None]]


class SubscriptionHandle:
    """Ticket d'abonnement. Appeler `unsubscribe()` pour retirer le handler.

    Utilisable comme context manager :

        with bus.subscribe(MyEvent, handler):
            ...  # handler actif dans ce bloc
        # retiré automatiquement à la sortie
    """

    def __init__(
        self,
        bus: EventBus,
        event_type: type[BaseModel],
        handler: AsyncHandler,
    ) -> None:
        self._bus = bus
        self._event_type = event_type
        self._handler = handler
        self._active: bool = True

    def unsubscribe(self) -> None:
        """Retire le handler du bus. Idempotent."""
        if self._active:
            self._bus._remove_handler(self._event_type, self._handler)
            self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def __enter__(self) -> SubscriptionHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.unsubscribe()


class EventBus:
    """Pub/sub async typé par classe d'événement Pydantic."""

    def __init__(self) -> None:
        self._handlers: dict[type[BaseModel], list[AsyncHandler]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Abonnement
    # ------------------------------------------------------------------
    def subscribe(
        self,
        event_type: type[EventT],
        handler: Callable[[EventT], Awaitable[None]],
    ) -> SubscriptionHandle:
        """Enregistre un handler async pour un type d'événement."""
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)
        log.debug(
            "event_bus_subscribed",
            event_type=event_type.__name__,
            handler=getattr(handler, "__name__", repr(handler)),
        )
        return SubscriptionHandle(self, event_type, handler)

    def _remove_handler(
        self,
        event_type: type[BaseModel],
        handler: AsyncHandler,
    ) -> None:
        """Retire un handler. Appelé par `SubscriptionHandle.unsubscribe`."""
        with self._lock:
            handlers = self._handlers.get(event_type)
            if handlers is None:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                return
            if not handlers:
                del self._handlers[event_type]

    # ------------------------------------------------------------------
    # Publication
    # ------------------------------------------------------------------
    async def publish(self, event: BaseModel) -> None:
        """Publie un événement à tous les handlers enregistrés pour son type.

        Les handlers tournent en parallèle. Une exception dans un handler est
        loguée, les autres continuent.
        """
        event_type = type(event)
        with self._lock:
            handlers = list(self._handlers.get(event_type, []))

        if not handlers:
            return

        await asyncio.gather(*(self._safe_invoke(h, event) for h in handlers))

    def publish_threadsafe(
        self,
        event: BaseModel,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Publie depuis un thread non-asyncio vers la loop cible.

        Typique : le kill switch (pynput, thread dédié) qui veut notifier la
        boucle OODA d'un événement d'arrêt.
        """
        asyncio.run_coroutine_threadsafe(self.publish(event), loop)

    async def _safe_invoke(self, handler: AsyncHandler, event: BaseModel) -> None:
        try:
            await handler(event)
        except Exception as exc:
            log.error(
                "event_handler_failed",
                handler=getattr(handler, "__name__", repr(handler)),
                event_type=type(event).__name__,
                error=str(exc),
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Diagnostic
    # ------------------------------------------------------------------
    def subscriber_count(self, event_type: type[BaseModel] | None = None) -> int:
        """Nombre de handlers enregistrés, globalement ou pour un type donné."""
        with self._lock:
            if event_type is None:
                return sum(len(h) for h in self._handlers.values())
            return len(self._handlers.get(event_type, []))
