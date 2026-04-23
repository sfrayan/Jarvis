"""Tests unitaires du kill switch — priorité absolue (CLAUDE.md).

Aucun test ne démarre le vrai listener pynput : on exerce directement les
handlers `_on_press` / `_on_release` et l'API publique, pour éviter de
capturer les frappes de l'utilisateur pendant l'exécution des tests.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from pynput import keyboard

from config.schema import KillSwitchConfig
from safety.kill_switch import KillSwitch

pytestmark = pytest.mark.unit


def _mk_config(**overrides: object) -> KillSwitchConfig:
    """Config par défaut adaptée aux tests (seuils courts)."""
    defaults: dict[str, object] = {
        "hotkey": "f12",
        "escape_long_ms": 100,  # 100 ms pour garder les tests rapides
        "corner_trigger": True,
    }
    defaults.update(overrides)
    return KillSwitchConfig(**defaults)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# trigger() — API publique
# ----------------------------------------------------------------------
class TestTrigger:
    def test_trigger_sets_shutdown_event(self) -> None:
        ks = KillSwitch(_mk_config())
        ks.trigger("manual")
        assert ks.shutdown_event.is_set()
        assert ks.triggered is True
        assert ks.reason == "manual"
        assert ks.trigger_timestamp is not None

    def test_trigger_latency_under_50ms(self) -> None:
        """Obligation master prompt §8 : latence déclenchement → event < 50 ms."""
        ks = KillSwitch(_mk_config())
        start = time.monotonic()
        ks.trigger("latency")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert ks.shutdown_event.is_set()
        assert elapsed_ms < 50, f"Trigger a pris {elapsed_ms:.2f} ms"

    def test_trigger_idempotent(self) -> None:
        """Second trigger ne réinvoque pas les callbacks ni ne change la raison."""
        ks = KillSwitch(_mk_config())
        calls: list[str] = []
        ks.register_cleanup(lambda reason: calls.append(reason))

        ks.trigger("first")
        ks.trigger("second")

        assert len(calls) == 1
        assert calls[0] == "first"
        assert ks.reason == "first"

    def test_trigger_from_failsafe(self) -> None:
        ks = KillSwitch(_mk_config())
        ks.trigger_from_failsafe()
        assert ks.shutdown_event.is_set()
        assert ks.reason == "pyautogui_failsafe_corner"


# ----------------------------------------------------------------------
# Callbacks cleanup
# ----------------------------------------------------------------------
class TestCleanupCallbacks:
    def test_registered_callback_receives_reason(self) -> None:
        ks = KillSwitch(_mk_config())
        calls: list[str] = []
        ks.register_cleanup(lambda r: calls.append(r))
        ks.trigger("abc")
        assert calls == ["abc"]

    def test_failing_callback_does_not_block_others(self) -> None:
        ks = KillSwitch(_mk_config())
        calls: list[str] = []

        def boom(reason: str) -> None:
            raise RuntimeError("boom")

        ks.register_cleanup(boom)
        ks.register_cleanup(lambda r: calls.append(r))
        ks.trigger("test")

        assert calls == ["test"]
        assert ks.shutdown_event.is_set()

    def test_register_after_trigger_fires_immediately(self) -> None:
        """Fail-closed : si on enregistre tardivement, le callback voit bien
        l'état déclenché."""
        ks = KillSwitch(_mk_config())
        ks.trigger("pre")
        calls: list[str] = []
        ks.register_cleanup(lambda r: calls.append(r))
        assert calls == ["pre"]


# ----------------------------------------------------------------------
# F-key handler (F12 par défaut)
# ----------------------------------------------------------------------
class TestHotkey:
    def test_f12_triggers_immediately(self) -> None:
        ks = KillSwitch(_mk_config())
        ks._on_press(keyboard.Key.f12)
        assert ks.shutdown_event.is_set()
        assert ks.reason == "f12_pressed"

    def test_other_keys_do_not_trigger(self) -> None:
        ks = KillSwitch(_mk_config())
        ks._on_press(keyboard.Key.space)
        ks._on_press(keyboard.Key.tab)
        ks._on_press(keyboard.Key.enter)
        assert not ks.shutdown_event.is_set()

    def test_configurable_hotkey_f11(self) -> None:
        ks = KillSwitch(_mk_config(hotkey="f11"))
        ks._on_press(keyboard.Key.f11)
        assert ks.shutdown_event.is_set()
        assert ks.reason == "f11_pressed"

    def test_unsupported_hotkey_raises(self) -> None:
        with pytest.raises(ValueError, match="non supportée"):
            KillSwitch(_mk_config(hotkey="ctrl+alt+x"))


# ----------------------------------------------------------------------
# Échap maintenu
# ----------------------------------------------------------------------
class TestEscapeLongPress:
    def test_escape_held_beyond_threshold_triggers(self) -> None:
        # Le schema Pydantic impose escape_long_ms >= 100.
        ks = KillSwitch(_mk_config(escape_long_ms=100))
        ks._on_press(keyboard.Key.esc)
        time.sleep(0.17)  # 170 ms > 100 ms + marge
        assert ks.shutdown_event.is_set()
        assert ks.reason is not None
        assert "escape_held" in ks.reason

    def test_escape_released_early_cancels_trigger(self) -> None:
        ks = KillSwitch(_mk_config(escape_long_ms=100))
        ks._on_press(keyboard.Key.esc)
        time.sleep(0.03)  # 30 ms < 100 ms
        ks._on_release(keyboard.Key.esc)
        time.sleep(0.15)  # attend au-delà du seuil initial
        assert not ks.shutdown_event.is_set()

    def test_second_press_while_timer_running_is_noop(self) -> None:
        """Un second press sans release n'ajoute pas un 2e timer concurrent."""
        ks = KillSwitch(_mk_config(escape_long_ms=100))
        ks._on_press(keyboard.Key.esc)
        ks._on_press(keyboard.Key.esc)  # ignoré (timer déjà armé)
        time.sleep(0.17)
        assert ks.shutdown_event.is_set()

    def test_stop_cancels_pending_escape_timer(self) -> None:
        ks = KillSwitch(_mk_config(escape_long_ms=200))
        ks._on_press(keyboard.Key.esc)
        ks.stop()
        time.sleep(0.25)
        assert not ks.shutdown_event.is_set()


# ----------------------------------------------------------------------
# wait_async()
# ----------------------------------------------------------------------
class TestAsyncWait:
    @pytest.mark.asyncio
    async def test_wait_async_returns_after_trigger(self) -> None:
        ks = KillSwitch(_mk_config())

        async def fire_soon() -> None:
            await asyncio.sleep(0.02)
            ks.trigger("async")

        task = asyncio.create_task(fire_soon())
        await asyncio.wait_for(ks.wait_async(), timeout=1.0)
        await task  # évite RUF006 et garantit la terminaison propre
        assert ks.shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_wait_async_returns_immediately_if_already_triggered(self) -> None:
        ks = KillSwitch(_mk_config())
        ks.trigger("pre_existing")
        await asyncio.wait_for(ks.wait_async(), timeout=0.5)
        assert ks.shutdown_event.is_set()
