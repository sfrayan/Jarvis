"""Kill switch global de Jarvis.

Priorité absolue : ce module DOIT être démarré avant tout autre thread
applicatif dans `main.py` (cf. CLAUDE.md §Sécurité).

Déclencheurs gérés ici :

- **F-key configurable** (défaut `F12`) : arrêt immédiat dès qu'on la presse.
- **Échap maintenu** au-delà de `escape_long_ms` (défaut 1000 ms) : arrêt après
  maintien prolongé. Relâcher avant le seuil annule.
- **pyautogui.FAILSAFE** (souris en (0,0)) : le module active le flag global
  `pyautogui.FAILSAFE = True` lors du `start()`. Les actuators (Itération 5)
  attrapent `FailSafeException` et appellent `trigger_from_failsafe()`.

Contrat :

- `shutdown_event` (threading.Event) : unique source de vérité partagée avec
  les consumers synchrones (threads audio, listeners, etc.).
- `wait_async()` : coroutine qui bloque jusqu'au déclenchement — pour la boucle
  OODA asyncio.
- `trigger()` est **idempotent** : premier appel gagne, les suivants sont
  silencieusement ignorés. Les callbacks enregistrés via `register_cleanup`
  ne sont exécutés qu'une fois.
- Latence cible : déclenchement → `shutdown_event.is_set() == True` en < 50 ms
  (vérifié par test unitaire).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

import pyautogui
import structlog
from pynput import keyboard

from config.schema import KillSwitchConfig

log = structlog.get_logger(__name__)


class KillSwitch:
    """Listener pynput global + événement d'arrêt partagé.

    Thread-safe : les handlers pynput tournent dans un thread dédié, l'API
    publique (`trigger`, `register_cleanup`) utilise un lock pour l'état.
    """

    def __init__(self, config: KillSwitchConfig) -> None:
        self._config = config
        self._hotkey_key = self._resolve_hotkey(config.hotkey)

        self.shutdown_event: threading.Event = threading.Event()

        self._triggered: bool = False
        self._trigger_reason: str | None = None
        self._trigger_timestamp: float | None = None

        self._listener: Any = None  # keyboard.Listener, non typé par pynput
        self._escape_timer: threading.Timer | None = None

        self._cleanup_callbacks: list[Callable[[str], None]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Démarre le listener pynput (thread dédié) et active FAILSAFE.

        Idempotent : second appel est no-op tant que `stop()` n'a pas été
        invoqué.
        """
        if self._listener is not None:
            return

        pyautogui.FAILSAFE = True

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,  # Ne consomme pas les touches globalement
        )
        self._listener.daemon = True
        self._listener.start()

        log.info(
            "kill_switch_started",
            hotkey=self._config.hotkey,
            escape_long_ms=self._config.escape_long_ms,
            corner_trigger=self._config.corner_trigger,
        )

    def stop(self) -> None:
        """Arrête le listener et annule le timer Escape. Pour les tests et
        l'arrêt programmatique propre."""
        if self._escape_timer is not None:
            self._escape_timer.cancel()
            self._escape_timer = None
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def trigger(self, reason: str) -> None:
        """Déclenche le kill switch. Idempotent.

        Appelable depuis n'importe quel thread. Exécute la cascade :
        log CRITICAL → `shutdown_event.set()` → callbacks enregistrés.
        """
        with self._lock:
            if self._triggered:
                return
            self._triggered = True
            self._trigger_reason = reason
            self._trigger_timestamp = time.time()
            callbacks = list(self._cleanup_callbacks)

        # Hors du lock : log + callbacks, pour ne pas sérialiser l'observabilité.
        log.critical(
            "kill_switch_triggered",
            reason=reason,
            timestamp=self._trigger_timestamp,
        )
        self.shutdown_event.set()

        for callback in callbacks:
            try:
                callback(reason)
            except Exception as exc:
                log.error(
                    "kill_switch_cleanup_failed",
                    callback=getattr(callback, "__name__", repr(callback)),
                    error=str(exc),
                    exc_info=True,
                )

    def trigger_from_failsafe(self) -> None:
        """À appeler par les actuators quand ils attrapent
        `pyautogui.FailSafeException`."""
        self.trigger("pyautogui_failsafe_corner")

    def register_cleanup(self, callback: Callable[[str], None]) -> None:
        """Enregistre un callback à exécuter lors du déclenchement.

        Si le kill switch est déjà déclenché au moment de l'enregistrement,
        le callback est invoqué immédiatement (état "fail-closed").
        """
        with self._lock:
            already_triggered = self._triggered
            reason = self._trigger_reason
            if not already_triggered:
                self._cleanup_callbacks.append(callback)

        if already_triggered and reason is not None:
            try:
                callback(reason)
            except Exception as exc:
                log.error(
                    "kill_switch_cleanup_failed_late",
                    error=str(exc),
                    exc_info=True,
                )

    async def wait_async(self) -> None:
        """Attend (async-friendly) le déclenchement du kill switch."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.shutdown_event.wait)

    # ------------------------------------------------------------------
    # Propriétés de diagnostic
    # ------------------------------------------------------------------
    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def reason(self) -> str | None:
        return self._trigger_reason

    @property
    def trigger_timestamp(self) -> float | None:
        return self._trigger_timestamp

    # ------------------------------------------------------------------
    # Handlers pynput (thread du listener)
    # ------------------------------------------------------------------
    def _on_press(self, key: Any) -> None:
        if self._triggered:
            return
        try:
            if key == self._hotkey_key:
                self.trigger(f"{self._config.hotkey}_pressed")
                return

            if key == keyboard.Key.esc and self._escape_timer is None:
                timer = threading.Timer(
                    self._config.escape_long_ms / 1000.0,
                    self.trigger,
                    args=(f"escape_held_{self._config.escape_long_ms}ms",),
                )
                timer.daemon = True
                self._escape_timer = timer
                timer.start()
        except Exception as exc:
            log.error("kill_switch_on_press_error", error=str(exc), exc_info=True)

    def _on_release(self, key: Any) -> None:
        if self._triggered:
            return
        try:
            if key == keyboard.Key.esc and self._escape_timer is not None:
                self._escape_timer.cancel()
                self._escape_timer = None
        except Exception as exc:
            log.error("kill_switch_on_release_error", error=str(exc), exc_info=True)

    # ------------------------------------------------------------------
    # Outils internes
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_hotkey(name: str) -> Any:
        """Convertit le nom de touche (config) en objet pynput.

        Supporte F1..F24 et `esc`. Lève `ValueError` sinon.
        """
        lowered = name.lower()
        if lowered.startswith("f") and lowered[1:].isdigit():
            n = int(lowered[1:])
            if 1 <= n <= 24:
                return getattr(keyboard.Key, f"f{n}")
        if lowered == "esc":
            return keyboard.Key.esc
        raise ValueError(
            f"Hotkey '{name}' non supportée. Valeurs acceptées : f1..f24, esc."
        )
