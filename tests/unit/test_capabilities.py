"""Tests unitaires du registre de capacites locales."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hands.capabilities import CapabilityDecision, CapabilityRegistry
from hands.inventory import (
    CapabilityAction,
    CapabilityTargetType,
    InstalledApp,
    LocalCapability,
    LocalInventory,
    WindowsServiceInfo,
)

pytestmark = pytest.mark.unit


class _FakeInventoryScanner:
    def __init__(self, *inventories: LocalInventory) -> None:
        self._inventories = list(inventories)
        self.calls = 0

    def scan(self) -> LocalInventory:
        self.calls += 1
        if not self._inventories:
            raise RuntimeError("no inventory")
        if len(self._inventories) == 1:
            return self._inventories[0]
        return self._inventories.pop(0)


def _inventory(*capabilities: LocalCapability) -> LocalInventory:
    return LocalInventory(
        scanned_at=1.0,
        computer_name="JARVIS-PC",
        is_windows=True,
        admin_available=False,
        apps=(
            InstalledApp(name="Chrome", source="start_menu"),
            InstalledApp(name="Spotify", source="known_local_action"),
        ),
        services=(
            WindowsServiceInfo(
                name="WinDefend",
                display_name="Microsoft Defender Antivirus Service",
                status="running",
                sensitive=True,
            ),
        ),
        capabilities=capabilities,
    )


def _capability(
    *,
    target_type: CapabilityTargetType,
    target_name: str,
    action: CapabilityAction,
    requires_confirmation: bool = False,
    requires_admin: bool = False,
    destructive: bool = False,
) -> LocalCapability:
    return LocalCapability(
        target_type=target_type,
        target_name=target_name,
        action=action,
        available=True,
        requires_confirmation=requires_confirmation,
        requires_admin=requires_admin,
        destructive=destructive,
        reason="test capability",
    )


class TestCapabilityRegistry:
    def test_find_capabilities_by_exact_target_and_action(self) -> None:
        registry = CapabilityRegistry(
            inventory=_inventory(
                _capability(target_type="app", target_name="Chrome", action="open"),
                _capability(target_type="app", target_name="Chrome", action="close"),
            )
        )

        matches = registry.find_capabilities(
            target_type="app",
            target_name="Chrome",
            action="open",
        )

        assert len(matches) == 1
        assert matches[0].target_name == "Chrome"
        assert matches[0].action == "open"

    def test_find_capabilities_accepts_partial_and_accent_folded_names(self) -> None:
        registry = CapabilityRegistry(
            inventory=_inventory(
                _capability(target_type="app", target_name="Parametres", action="open"),
            )
        )

        matches = registry.find_capabilities(
            target_type="app",
            target_name="paramètres",
            action="open",
        )

        assert len(matches) == 1
        assert matches[0].target_name == "Parametres"

    def test_open_app_can_execute_now(self) -> None:
        registry = CapabilityRegistry(
            inventory=_inventory(
                _capability(target_type="app", target_name="Spotify", action="open"),
            )
        )

        decision = registry.can_execute(
            target_type="app",
            target_name="Spotify",
            action="open",
        )

        assert decision.available is True
        assert decision.can_execute_now is True
        assert decision.requires_confirmation is False
        assert decision.requires_admin is False
        assert decision.destructive is False
        assert decision.capability is not None

    def test_sensitive_service_stop_requires_confirmation_admin_and_is_destructive(self) -> None:
        registry = CapabilityRegistry(
            inventory=_inventory(
                _capability(
                    target_type="service",
                    target_name="WinDefend",
                    action="stop",
                    requires_confirmation=True,
                    requires_admin=True,
                    destructive=True,
                ),
            )
        )

        decision = registry.can_execute(
            target_type="service",
            target_name="windows defender",
            action="stop",
        )

        assert decision.available is True
        assert decision.can_execute_now is False
        assert decision.requires_confirmation is True
        assert decision.requires_admin is True
        assert decision.destructive is True
        assert "confirmation requise" in decision.reason
        assert "droits admin requis" in decision.reason

    def test_unknown_capability_is_refused_cleanly(self) -> None:
        registry = CapabilityRegistry(inventory=_inventory())

        decision = registry.can_execute(
            target_type="app",
            target_name="Application Inconnue",
            action="open",
        )

        assert decision == CapabilityDecision(
            available=False,
            can_execute_now=False,
            reason="Capacite locale inconnue ou non detectee",
        )

    def test_refresh_replaces_snapshot_from_scanner(self) -> None:
        first = _inventory(_capability(target_type="app", target_name="Chrome", action="open"))
        second = _inventory(_capability(target_type="app", target_name="Firefox", action="open"))
        scanner = _FakeInventoryScanner(first, second)
        registry = CapabilityRegistry(scanner=scanner)

        assert registry.can_execute(
            target_type="app",
            target_name="Chrome",
            action="open",
        ).available

        refreshed = registry.refresh()

        assert scanner.calls == 2
        assert refreshed is second
        assert (
            registry.can_execute(
                target_type="app",
                target_name="Chrome",
                action="open",
            ).available
            is False
        )
        assert (
            registry.can_execute(
                target_type="app",
                target_name="Firefox",
                action="open",
            ).available
            is True
        )

    def test_decision_model_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            CapabilityDecision.model_validate(
                {
                    "available": True,
                    "can_execute_now": True,
                    "reason": "ok",
                    "unexpected": True,
                }
            )
