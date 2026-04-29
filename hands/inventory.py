"""Inventaire local readonly du PC Windows.

Iteration 5G-C : cette brique observe l'environnement local sans executer
d'action utilisateur. Elle sert a construire une carte prudente des capacites
que Jarvis pourra proposer ensuite en mode `assisted`.
"""

from __future__ import annotations

import csv
import os
import platform
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

AppSource = Literal["start_menu", "known_local_action"]
CapabilityAction = Literal[
    "open",
    "close",
    "inspect",
    "start",
    "stop",
    "volume_up",
    "volume_down",
    "mute",
    "play_pause",
    "next",
    "previous",
]
CapabilityTargetType = Literal["app", "folder", "service", "process", "system", "media"]
ProcessSource = Literal["tasklist"]
ServiceStatus = Literal["running", "stopped", "paused", "unknown"]
StartupSource = Literal["startup_folder", "registry_run"]

_KNOWN_ACTION_APPS = (
    "Spotify",
    "Chrome",
    "Discord",
    "Task Manager",
    "Settings",
    "VS Code",
    "Docker Desktop",
    "KeePass",
    "Steam",
    "Origin",
    "EA App",
    "Opera GX",
    "Opera",
    "Firefox",
    "Edge",
    "Calculator",
    "Notepad",
    "Paint",
)
_KNOWN_FOLDERS = ("Downloads", "Documents", "Pictures", "Videos", "Desktop", "Music")
_SENSITIVE_SERVICES = frozenset(
    {
        "bits",
        "dnscache",
        "eventlog",
        "mpssvc",
        "rpcss",
        "samss",
        "schedule",
        "trustedinstaller",
        "windefend",
        "winmgmt",
        "wuauserv",
    }
)


class InstalledApp(BaseModel):
    """Application detectee ou connue localement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    source: AppSource
    launch_hint: str | None = Field(default=None, min_length=1)
    path: str | None = Field(default=None, min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RunningProcessInfo(BaseModel):
    """Processus observe via une commande systeme readonly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    pid: int = Field(..., ge=0)
    source: ProcessSource = "tasklist"
    memory_kb: int | None = Field(default=None, ge=0)


class WindowsServiceInfo(BaseModel):
    """Etat readonly d'un service Windows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    status: ServiceStatus = "unknown"
    start_type: str | None = Field(default=None, min_length=1)
    sensitive: bool = False


class StartupEntry(BaseModel):
    """Programme configure au demarrage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    source: StartupSource
    command: str | None = Field(default=None, min_length=1)
    path: str | None = Field(default=None, min_length=1)


class LocalCapability(BaseModel):
    """Action potentiellement disponible, annotee par garde-fous."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_type: CapabilityTargetType
    target_name: str = Field(..., min_length=1)
    action: CapabilityAction
    available: bool
    requires_confirmation: bool
    requires_admin: bool
    destructive: bool
    reason: str = Field(..., min_length=1)


class LocalInventory(BaseModel):
    """Carte readonly du PC vue par Jarvis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scanned_at: float = Field(..., ge=0.0)
    computer_name: str | None = Field(default=None, min_length=1)
    is_windows: bool
    admin_available: bool
    apps: tuple[InstalledApp, ...] = ()
    running_processes: tuple[RunningProcessInfo, ...] = ()
    services: tuple[WindowsServiceInfo, ...] = ()
    startup_entries: tuple[StartupEntry, ...] = ()
    capabilities: tuple[LocalCapability, ...] = ()
    warnings: tuple[str, ...] = ()


class InventorySource(Protocol):
    """Source readonly injectable pour tester sans interroger le vrai PC."""

    def computer_name(self) -> str | None:
        """Retourne le nom de machine si disponible."""

    def is_windows(self) -> bool:
        """Indique si la plateforme courante ressemble a Windows."""

    def has_admin_rights(self) -> bool:
        """Indique si le process courant a des droits administrateur."""

    def iter_start_menu_apps(self) -> Sequence[InstalledApp]:
        """Liste les raccourcis applicatifs observables."""

    def iter_running_processes(self) -> Sequence[RunningProcessInfo]:
        """Liste les processus visibles."""

    def iter_services(self) -> Sequence[WindowsServiceInfo]:
        """Liste les services Windows visibles."""

    def iter_startup_entries(self) -> Sequence[StartupEntry]:
        """Liste les programmes au demarrage visibles."""


class WindowsInventorySource:
    """Source Windows readonly basee sur fichiers, registre et commandes de lecture."""

    def computer_name(self) -> str | None:
        return os.environ.get("COMPUTERNAME") or platform.node() or None

    def is_windows(self) -> bool:
        return platform.system() == "Windows"

    def has_admin_rights(self) -> bool:
        if not self.is_windows():
            return False
        try:
            import ctypes

            windll = getattr(ctypes, "windll", None)
            shell32 = getattr(windll, "shell32", None)
            if shell32 is None:
                return False
            return bool(shell32.IsUserAnAdmin())
        except Exception:
            return False

    def iter_start_menu_apps(self) -> Sequence[InstalledApp]:
        apps: list[InstalledApp] = []
        for root in self._start_menu_roots():
            if not root.exists():
                continue
            for shortcut in root.rglob("*.lnk"):
                apps.append(
                    InstalledApp(
                        name=_clean_shortcut_name(shortcut.stem),
                        source="start_menu",
                        launch_hint=shortcut.stem,
                        path=str(shortcut),
                        confidence=0.85,
                    )
                )
        return tuple(apps)

    def iter_running_processes(self) -> Sequence[RunningProcessInfo]:
        if not self.is_windows():
            return ()
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return ()
        processes: list[RunningProcessInfo] = []
        reader = csv.reader(result.stdout.splitlines())
        for row in reader:
            if len(row) < 5:
                continue
            processes.append(
                RunningProcessInfo(
                    name=row[0],
                    pid=_parse_int(row[1]),
                    memory_kb=_parse_memory_kb(row[4]),
                )
            )
        return tuple(processes)

    def iter_services(self) -> Sequence[WindowsServiceInfo]:
        if not self.is_windows():
            return ()
        command = (
            "Get-Service | Select-Object Name,DisplayName,Status,StartType "
            "| ConvertTo-Csv -NoTypeInformation"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return ()
        services: list[WindowsServiceInfo] = []
        for row in csv.DictReader(result.stdout.splitlines()):
            name = row.get("Name", "").strip()
            display_name = row.get("DisplayName", "").strip() or name
            if not name:
                continue
            services.append(
                WindowsServiceInfo(
                    name=name,
                    display_name=display_name,
                    status=_service_status(row.get("Status", "")),
                    start_type=row.get("StartType") or None,
                    sensitive=_is_sensitive_service(name),
                )
            )
        return tuple(services)

    def iter_startup_entries(self) -> Sequence[StartupEntry]:
        entries = [*self._startup_folder_entries()]
        entries.extend(self._registry_startup_entries())
        return tuple(entries)

    def _start_menu_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        program_data = os.environ.get("PROGRAMDATA")
        app_data = os.environ.get("APPDATA")
        if program_data:
            roots.append(Path(program_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
        if app_data:
            roots.append(Path(app_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
        return tuple(roots)

    def _startup_folder_entries(self) -> tuple[StartupEntry, ...]:
        entries: list[StartupEntry] = []
        for root in self._startup_roots():
            if not root.exists():
                continue
            for shortcut in root.glob("*.lnk"):
                entries.append(
                    StartupEntry(
                        name=_clean_shortcut_name(shortcut.stem),
                        source="startup_folder",
                        path=str(shortcut),
                    )
                )
        return tuple(entries)

    def _startup_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        program_data = os.environ.get("PROGRAMDATA")
        app_data = os.environ.get("APPDATA")
        if program_data:
            roots.append(
                Path(program_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "StartUp"
            )
        if app_data:
            roots.append(
                Path(app_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
            )
        return tuple(roots)

    def _registry_startup_entries(self) -> tuple[StartupEntry, ...]:
        if not self.is_windows():
            return ()
        try:
            import winreg
        except ImportError:
            return ()

        registry_keys: tuple[tuple[Any, str], ...] = (
            (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
        )
        entries: list[StartupEntry] = []
        for hive, key_path in registry_keys:
            try:
                with winreg.OpenKey(hive, key_path) as key:
                    count = winreg.QueryInfoKey(key)[1]
                    for index in range(count):
                        name, command, _ = winreg.EnumValue(key, index)
                        entries.append(
                            StartupEntry(
                                name=str(name),
                                source="registry_run",
                                command=str(command),
                            )
                        )
            except OSError:
                continue
        return tuple(entries)


class LocalInventoryScanner:
    """Construit une carte locale readonly a partir d'une source injectable."""

    def __init__(self, source: InventorySource | None = None) -> None:
        self._source = source or WindowsInventorySource()

    def scan(self) -> LocalInventory:
        """Collecte l'inventaire et derive les capacites prudentes."""
        warnings: list[str] = []
        apps = _merge_apps(
            [
                *_safe_collect(self._source.iter_start_menu_apps, warnings, "start_menu_apps"),
                *_known_apps(),
            ]
        )
        processes = _safe_collect(self._source.iter_running_processes, warnings, "processes")
        services = tuple(
            _normalize_service(service)
            for service in _safe_collect(self._source.iter_services, warnings, "services")
        )
        startup_entries = _safe_collect(self._source.iter_startup_entries, warnings, "startup")
        capabilities = _build_capabilities(
            apps=apps,
            services=services,
            processes=processes,
        )

        return LocalInventory(
            scanned_at=time.time(),
            computer_name=_safe_value(self._source.computer_name, warnings, "computer_name"),
            is_windows=bool(_safe_value(self._source.is_windows, warnings, "is_windows")),
            admin_available=bool(
                _safe_value(self._source.has_admin_rights, warnings, "admin_rights")
            ),
            apps=apps,
            running_processes=processes,
            services=services,
            startup_entries=startup_entries,
            capabilities=capabilities,
            warnings=tuple(warnings),
        )


T = TypeVar("T")


def _safe_collect(
    collect: Callable[[], Sequence[T]],
    warnings: list[str],
    label: str,
) -> tuple[T, ...]:
    try:
        return tuple(collect())
    except Exception as exc:
        warnings.append(f"{label}: {type(exc).__name__}")
        return ()


def _safe_value(
    collect: Callable[[], T],
    warnings: list[str],
    label: str,
) -> T | None:
    try:
        return collect()
    except Exception as exc:
        warnings.append(f"{label}: {type(exc).__name__}")
        return None


def _known_apps() -> tuple[InstalledApp, ...]:
    return tuple(
        InstalledApp(
            name=name,
            source="known_local_action",
            launch_hint=name,
            confidence=0.65,
        )
        for name in _KNOWN_ACTION_APPS
    )


def _merge_apps(apps: Sequence[InstalledApp]) -> tuple[InstalledApp, ...]:
    merged: dict[str, InstalledApp] = {}
    for app in apps:
        key = _fold_key(app.name)
        existing = merged.get(key)
        if existing is None or existing.source == "known_local_action":
            merged[key] = app
    return tuple(sorted(merged.values(), key=lambda app: app.name.casefold()))


def _normalize_service(service: WindowsServiceInfo) -> WindowsServiceInfo:
    if service.sensitive == _is_sensitive_service(service.name):
        return service
    return service.model_copy(update={"sensitive": _is_sensitive_service(service.name)})


def _build_capabilities(
    *,
    apps: Sequence[InstalledApp],
    services: Sequence[WindowsServiceInfo],
    processes: Sequence[RunningProcessInfo],
) -> tuple[LocalCapability, ...]:
    capabilities: list[LocalCapability] = []

    for app in apps:
        capabilities.append(
            LocalCapability(
                target_type="app",
                target_name=app.name,
                action="open",
                available=True,
                requires_confirmation=False,
                requires_admin=False,
                destructive=False,
                reason="Ouverture d'application allowlistee ou detectee",
            )
        )
        capabilities.append(
            LocalCapability(
                target_type="app",
                target_name=app.name,
                action="close",
                available=True,
                requires_confirmation=True,
                requires_admin=False,
                destructive=True,
                reason="Fermeture d'application: risque de perte de travail non sauvegarde",
            )
        )

    for folder in _KNOWN_FOLDERS:
        capabilities.append(
            LocalCapability(
                target_type="folder",
                target_name=folder,
                action="open",
                available=True,
                requires_confirmation=False,
                requires_admin=False,
                destructive=False,
                reason="Dossier utilisateur connu",
            )
        )

    for process in processes:
        capabilities.append(
            LocalCapability(
                target_type="process",
                target_name=process.name,
                action="inspect",
                available=True,
                requires_confirmation=False,
                requires_admin=False,
                destructive=False,
                reason="Observation readonly du processus",
            )
        )

    for service in services:
        capabilities.append(
            LocalCapability(
                target_type="service",
                target_name=service.name,
                action="inspect",
                available=True,
                requires_confirmation=False,
                requires_admin=False,
                destructive=False,
                reason="Observation readonly du service",
            )
        )
        for action in ("start", "stop"):
            capabilities.append(
                LocalCapability(
                    target_type="service",
                    target_name=service.name,
                    action=action,
                    available=True,
                    requires_confirmation=True,
                    requires_admin=True,
                    destructive=action == "stop" and service.sensitive,
                    reason=_service_action_reason(service, action),
                )
            )

    capabilities.extend(_system_capabilities())
    return tuple(
        sorted(capabilities, key=lambda item: (item.target_type, item.target_name, item.action))
    )


def _system_capabilities() -> tuple[LocalCapability, ...]:
    return (
        LocalCapability(
            target_type="system",
            target_name="volume",
            action="volume_up",
            available=True,
            requires_confirmation=False,
            requires_admin=False,
            destructive=False,
            reason="Touche multimedia locale",
        ),
        LocalCapability(
            target_type="system",
            target_name="volume",
            action="volume_down",
            available=True,
            requires_confirmation=False,
            requires_admin=False,
            destructive=False,
            reason="Touche multimedia locale",
        ),
        LocalCapability(
            target_type="system",
            target_name="volume",
            action="mute",
            available=True,
            requires_confirmation=False,
            requires_admin=False,
            destructive=False,
            reason="Touche multimedia locale",
        ),
        LocalCapability(
            target_type="media",
            target_name="media_keys",
            action="play_pause",
            available=True,
            requires_confirmation=False,
            requires_admin=False,
            destructive=False,
            reason="Touche multimedia locale",
        ),
        LocalCapability(
            target_type="media",
            target_name="media_keys",
            action="next",
            available=True,
            requires_confirmation=False,
            requires_admin=False,
            destructive=False,
            reason="Touche multimedia locale",
        ),
        LocalCapability(
            target_type="media",
            target_name="media_keys",
            action="previous",
            available=True,
            requires_confirmation=False,
            requires_admin=False,
            destructive=False,
            reason="Touche multimedia locale",
        ),
    )


def _service_action_reason(service: WindowsServiceInfo, action: str) -> str:
    if service.sensitive and action == "stop":
        return "Service sensible: confirmation explicite et droits admin requis"
    return "Controle de service Windows: confirmation explicite et droits admin requis"


def _clean_shortcut_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.removesuffix(".lnk")).strip()


def _fold_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _parse_int(value: str) -> int:
    try:
        return int(value.strip())
    except ValueError:
        return 0


def _parse_memory_kb(value: str) -> int | None:
    digits = re.sub(r"\D", "", value)
    if not digits:
        return None
    return int(digits)


def _service_status(value: str) -> ServiceStatus:
    normalized = value.strip().casefold()
    if normalized == "running":
        return "running"
    if normalized == "stopped":
        return "stopped"
    if normalized == "paused":
        return "paused"
    return "unknown"


def _is_sensitive_service(name: str) -> bool:
    return name.casefold() in _SENSITIVE_SERVICES
