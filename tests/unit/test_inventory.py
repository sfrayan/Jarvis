"""Tests unitaires de l'inventaire local readonly."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hands.inventory import (
    InstalledApp,
    LocalCapability,
    LocalInventoryScanner,
    RunningProcessInfo,
    StartupEntry,
    WindowsServiceInfo,
)

pytestmark = pytest.mark.unit


class _FakeInventorySource:
    def __init__(
        self,
        *,
        apps: tuple[InstalledApp, ...] = (),
        processes: tuple[RunningProcessInfo, ...] = (),
        services: tuple[WindowsServiceInfo, ...] = (),
        startup_entries: tuple[StartupEntry, ...] = (),
        admin: bool = False,
        fail_services: bool = False,
    ) -> None:
        self._apps = apps
        self._processes = processes
        self._services = services
        self._startup_entries = startup_entries
        self._admin = admin
        self._fail_services = fail_services

    def computer_name(self) -> str:
        return "JARVIS-PC"

    def is_windows(self) -> bool:
        return True

    def has_admin_rights(self) -> bool:
        return self._admin

    def iter_start_menu_apps(self) -> tuple[InstalledApp, ...]:
        return self._apps

    def iter_running_processes(self) -> tuple[RunningProcessInfo, ...]:
        return self._processes

    def iter_services(self) -> tuple[WindowsServiceInfo, ...]:
        if self._fail_services:
            raise RuntimeError("service unavailable")
        return self._services

    def iter_startup_entries(self) -> tuple[StartupEntry, ...]:
        return self._startup_entries


def _capability(
    capabilities: tuple[LocalCapability, ...],
    *,
    target_type: str,
    target_name: str,
    action: str,
) -> LocalCapability:
    for capability in capabilities:
        if (
            capability.target_type == target_type
            and capability.target_name == target_name
            and capability.action == action
        ):
            return capability
    raise AssertionError(f"Capacite absente: {target_type} {target_name} {action}")


class TestLocalInventoryScanner:
    def test_scan_merges_detected_apps_with_known_local_actions(self) -> None:
        scanner = LocalInventoryScanner(
            _FakeInventorySource(
                apps=(
                    InstalledApp(
                        name="Chrome",
                        source="start_menu",
                        path=r"C:\ProgramData\Start Menu\Chrome.lnk",
                        confidence=0.9,
                    ),
                ),
                processes=(RunningProcessInfo(name="chrome.exe", pid=42, memory_kb=1024),),
                startup_entries=(
                    StartupEntry(
                        name="Discord",
                        source="startup_folder",
                        path=r"C:\Startup\Discord.lnk",
                    ),
                ),
            )
        )

        inventory = scanner.scan()

        app_names = {app.name for app in inventory.apps}
        assert {"Chrome", "Spotify", "Discord", "Settings"} <= app_names
        chrome_apps = [app for app in inventory.apps if app.name == "Chrome"]
        assert chrome_apps == [
            InstalledApp(
                name="Chrome",
                source="start_menu",
                path=r"C:\ProgramData\Start Menu\Chrome.lnk",
                confidence=0.9,
            )
        ]
        assert inventory.computer_name == "JARVIS-PC"
        assert inventory.is_windows is True
        assert inventory.admin_available is False
        assert len(inventory.startup_entries) == 1
        assert (
            _capability(
                inventory.capabilities,
                target_type="app",
                target_name="Chrome",
                action="open",
            ).requires_confirmation
            is False
        )

    def test_app_close_capability_requires_confirmation(self) -> None:
        inventory = LocalInventoryScanner(_FakeInventorySource()).scan()

        capability = _capability(
            inventory.capabilities,
            target_type="app",
            target_name="Discord",
            action="close",
        )

        assert capability.requires_confirmation is True
        assert capability.destructive is True
        assert capability.requires_admin is False

    def test_service_control_requires_confirmation_and_admin(self) -> None:
        inventory = LocalInventoryScanner(
            _FakeInventorySource(
                services=(
                    WindowsServiceInfo(
                        name="Spooler",
                        display_name="Print Spooler",
                        status="running",
                        start_type="Automatic",
                    ),
                )
            )
        ).scan()

        stop_capability = _capability(
            inventory.capabilities,
            target_type="service",
            target_name="Spooler",
            action="stop",
        )

        assert stop_capability.requires_confirmation is True
        assert stop_capability.requires_admin is True
        assert stop_capability.destructive is False

    def test_sensitive_service_stop_is_marked_destructive(self) -> None:
        inventory = LocalInventoryScanner(
            _FakeInventorySource(
                services=(
                    WindowsServiceInfo(
                        name="WinDefend",
                        display_name="Microsoft Defender Antivirus Service",
                        status="running",
                    ),
                )
            )
        ).scan()

        stop_capability = _capability(
            inventory.capabilities,
            target_type="service",
            target_name="WinDefend",
            action="stop",
        )

        assert inventory.services[0].sensitive is True
        assert stop_capability.requires_confirmation is True
        assert stop_capability.requires_admin is True
        assert stop_capability.destructive is True

    def test_processes_are_inspect_only(self) -> None:
        inventory = LocalInventoryScanner(
            _FakeInventorySource(
                processes=(RunningProcessInfo(name="python.exe", pid=1234, memory_kb=2048),)
            )
        ).scan()

        capability = _capability(
            inventory.capabilities,
            target_type="process",
            target_name="python.exe",
            action="inspect",
        )

        assert capability.requires_confirmation is False
        assert capability.requires_admin is False
        assert capability.destructive is False

    def test_system_and_media_capabilities_are_present(self) -> None:
        inventory = LocalInventoryScanner(_FakeInventorySource()).scan()

        assert _capability(
            inventory.capabilities,
            target_type="system",
            target_name="volume",
            action="volume_up",
        )
        assert _capability(
            inventory.capabilities,
            target_type="media",
            target_name="media_keys",
            action="play_pause",
        )

    def test_source_errors_become_warnings_without_breaking_scan(self) -> None:
        inventory = LocalInventoryScanner(
            _FakeInventorySource(
                fail_services=True, apps=(InstalledApp(name="Chrome", source="start_menu"),)
            )
        ).scan()

        assert inventory.apps
        assert inventory.services == ()
        assert inventory.warnings == ("services: RuntimeError",)

    def test_models_reject_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            InstalledApp.model_validate(
                {
                    "name": "Chrome",
                    "source": "start_menu",
                    "unexpected": True,
                }
            )
