"""Planification et execution locale pour les commandes PC simples.

Iteration 5G : ce module traduit certaines intentions `apps`, `system`, `media`
et `folders` en `HandsExecutionReport`. `observe` et `dry_run` restent sans
effet reel ; `assisted` et `autonomous` executent les actions locales sures.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Protocol, cast

from brain.events import IntentRouted
from config.schema import SafetyConfig, SafetyMode
from hands.capabilities import CapabilityDecision, CapabilityRegistry
from hands.executor import HandsExecutionReport, PlannedGuiAction
from hands.inventory import CapabilityAction, CapabilityTargetType, LocalCapability
from observability.logger import get_logger

log = get_logger(__name__)

_LOCAL_DOMAINS = frozenset({"apps", "folders", "media", "system"})
_OPEN_PATTERN = re.compile(r"\b(ouvre|ouvres|ouvrir|ouvre-moi|lance|lances|demarre|demarres)\b")
_CLOSE_PATTERN = re.compile(r"\b(ferme|fermes|quitte|quittes|eteins|eteint|arrete|arretes|coupe)\b")
_DESTRUCTIVE_SYSTEM_PATTERN = re.compile(
    r"\b(eteins|eteint|eteindre|shutdown|redemarre|reboot|arrete)\b.*\b(pc|ordinateur|windows)\b"
)

_APP_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("spotify",), "Spotify"),
    (("chrome", "google chrome"), "Chrome"),
    (("discord",), "Discord"),
    (("task manager", "gestionnaire de taches"), "Task Manager"),
    (("parametres", "settings"), "Settings"),
    (("vs code", "visual studio code"), "VS Code"),
    (("docker desktop",), "Docker Desktop"),
    (("keepass", "key pass", "qui passe"), "KeePass"),
    (("steam",), "Steam"),
    (("origin",), "Origin"),
    (("ea app", "ea desktop"), "EA App"),
    (("opera gx",), "Opera GX"),
    (("opera",), "Opera"),
    (("firefox",), "Firefox"),
    (("edge",), "Edge"),
    (("calculatrice",), "Calculator"),
    (("bloc-notes", "bloc notes", "notepad"), "Notepad"),
    (("paint",), "Paint"),
)

_FOLDER_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("telechargements", "downloads"), "Downloads"),
    (("documents",), "Documents"),
    (("images", "photos"), "Pictures"),
    (("videos",), "Videos"),
    (("bureau", "desktop"), "Desktop"),
    (("musique", "music"), "Music"),
)

_APP_SEARCH_TERMS = {
    "Spotify": "Spotify",
    "Chrome": "Google Chrome",
    "Discord": "Discord",
    "Settings": "Parametres",
    "VS Code": "Visual Studio Code",
    "Docker Desktop": "Docker Desktop",
    "KeePass": "KeePass",
    "Steam": "Steam",
    "Origin": "Origin",
    "EA App": "EA App",
    "Opera GX": "Opera GX",
    "Opera": "Opera",
    "Firefox": "Firefox",
    "Edge": "Microsoft Edge",
    "Calculator": "Calculatrice",
    "Notepad": "Bloc-notes",
    "Paint": "Paint",
}

_FOLDER_SHELL_COMMANDS = {
    "Downloads": "shell:Downloads",
    "Documents": "shell:Personal",
    "Pictures": "shell:My Pictures",
    "Videos": "shell:My Video",
    "Desktop": "shell:Desktop",
    "Music": "shell:My Music",
}

_VOLUME_KEYS = {
    "volume_up": "volumeup",
    "volume_down": "volumedown",
    "mute": "volumemute",
}

_MEDIA_KEYS = {
    "pause": "playpause",
    "play": "playpause",
    "next": "nexttrack",
    "previous": "prevtrack",
}


class LocalActionBackend(Protocol):
    """Contrat minimal d'un actuateur local injectable."""

    def perform(self, action: PlannedGuiAction) -> None:
        """Execute une action locale deja validee comme sure."""


class CapabilityRegistryLike(Protocol):
    """Contrat minimal du registre de capacites."""

    def find_capabilities(
        self,
        *,
        target_type: CapabilityTargetType | None = None,
        target_name: str | None = None,
        action: CapabilityAction | None = None,
    ) -> tuple[LocalCapability, ...]:
        """Recherche les capacites locales disponibles."""

    def can_execute(
        self,
        *,
        target_type: CapabilityTargetType,
        target_name: str,
        action: CapabilityAction,
    ) -> CapabilityDecision:
        """Retourne les garde-fous pour une action locale."""


@dataclass(frozen=True)
class _MatchedLocalAction:
    action: PlannedGuiAction
    decision: CapabilityDecision | None = None


class _PyAutoGuiLike(Protocol):
    def hotkey(self, *keys: str) -> None:
        """Appuie sur une combinaison de touches."""

    def write(self, message: str, interval: float = 0.0) -> None:
        """Tape du texte."""

    def press(self, key: str) -> None:
        """Appuie sur une touche."""


class PyAutoGuiLocalActionBackend:
    """Actuateur Windows minimal base sur pyautogui."""

    def perform(self, action: PlannedGuiAction) -> None:
        """Execute l'action locale supportee."""
        if action.type == "launch_app":
            self._launch_app(_required_text(action))
            return
        if action.type == "system_tool":
            self._open_system_tool(_required_text(action))
            return
        if action.type == "open_folder":
            self._open_folder(_required_text(action))
            return
        if action.type == "system_volume":
            self._press_mapped_key(_required_text(action), _VOLUME_KEYS)
            return
        if action.type == "media_control":
            self._press_mapped_key(_required_text(action), _MEDIA_KEYS)
            return
        raise ValueError(f"Action locale non supportee: {action.type}")

    def _launch_app(self, app: str) -> None:
        term = _APP_SEARCH_TERMS.get(app, app)
        pyautogui = _pyautogui()
        pyautogui.hotkey("win")
        time.sleep(0.15)
        pyautogui.write(term, interval=0.01)
        pyautogui.press("enter")

    def _open_system_tool(self, tool: str) -> None:
        if tool != "Task Manager":
            raise ValueError(f"Outil systeme non supporte: {tool}")
        _pyautogui().hotkey("ctrl", "shift", "esc")

    def _open_folder(self, folder: str) -> None:
        command = _FOLDER_SHELL_COMMANDS.get(folder)
        if command is None:
            raise ValueError(f"Dossier local non allowliste: {folder}")
        pyautogui = _pyautogui()
        pyautogui.hotkey("win", "r")
        time.sleep(0.15)
        pyautogui.write(command, interval=0.01)
        pyautogui.press("enter")

    def _press_mapped_key(self, name: str, mapping: dict[str, str]) -> None:
        key = mapping.get(name)
        if key is None:
            raise ValueError(f"Action clavier non supportee: {name}")
        _pyautogui().press(key)


class LocalActionPlanner:
    """Produit et execute les commandes locales sans vision."""

    def __init__(
        self,
        safety: SafetyConfig,
        *,
        backend: LocalActionBackend | None = None,
        capabilities: CapabilityRegistryLike | None = None,
    ) -> None:
        self._safety = safety
        self._backend = backend or PyAutoGuiLocalActionBackend()
        self._capabilities = capabilities or CapabilityRegistry()

    def plan(self, event: IntentRouted) -> HandsExecutionReport | None:
        """Retourne un rapport local si l'intention est reconnue."""
        if event.intent != "gui" or event.domain not in _LOCAL_DOMAINS:
            return None

        text = _fold(event.normalized_text)
        match = _match_local_action(text, domain=event.domain, capabilities=self._capabilities)
        if match is None:
            return None

        return _build_report(
            mode=self._safety.mode,
            action=match.action,
            backend=self._backend,
            decision=match.decision,
        )


def _match_local_action(
    text: str,
    *,
    domain: str,
    capabilities: CapabilityRegistryLike,
) -> _MatchedLocalAction | None:
    if _DESTRUCTIVE_SYSTEM_PATTERN.search(text):
        return _MatchedLocalAction(
            PlannedGuiAction(
                type="system_command",
                text="shutdown",
                destructive=True,
            )
        )

    volume_action = _match_volume_action(text)
    if volume_action is not None:
        return _with_capability_decision(volume_action, capabilities)

    if _CLOSE_PATTERN.search(text):
        app = _match_alias(text, _APP_ALIASES)
        if app is None:
            app = _match_dynamic_target(
                text,
                target_type="app",
                action="close",
                capabilities=capabilities,
            )
        if app is None:
            return None
        return _with_capability_decision(
            PlannedGuiAction(type="close_app", text=app, destructive=True),
            capabilities,
        )

    media_action = _match_media_action(text)
    if media_action is not None:
        return _with_capability_decision(media_action, capabilities)

    if _OPEN_PATTERN.search(text):
        folder = _match_alias(text, _FOLDER_ALIASES)
        if folder is not None:
            return _with_capability_decision(
                PlannedGuiAction(type="open_folder", text=folder),
                capabilities,
            )

        app = _match_alias(text, _APP_ALIASES)
        if app is not None:
            if app == "Task Manager":
                return _with_capability_decision(
                    PlannedGuiAction(type="system_tool", text=app),
                    capabilities,
                )
            return _with_capability_decision(
                PlannedGuiAction(type="launch_app", text=app),
                capabilities,
            )

        dynamic_target = _match_dynamic_open_target(
            text,
            domain=domain,
            capabilities=capabilities,
        )
        if dynamic_target is not None:
            action_type = "open_folder" if dynamic_target[0] == "folder" else "launch_app"
            return _with_capability_decision(
                PlannedGuiAction(type=action_type, text=dynamic_target[1]),
                capabilities,
            )

    return None


def _match_media_action(text: str) -> PlannedGuiAction | None:
    if "spotify" not in text and "musique" not in text and "youtube" not in text:
        return None
    if _OPEN_PATTERN.search(text):
        return PlannedGuiAction(type="launch_app", text="Spotify")
    if re.search(r"\b(pause|mets en pause)\b", text):
        return PlannedGuiAction(type="media_control", text="pause")
    if re.search(
        r"\b(reprends|resume|play|active la musique|mets de la musique|met de la musique|joue de la musique)\b",
        text,
    ):
        return PlannedGuiAction(type="media_control", text="play")
    if re.search(r"\b(suivante|next)\b", text):
        return PlannedGuiAction(type="media_control", text="next")
    if re.search(r"\b(precedente|previous)\b", text):
        return PlannedGuiAction(type="media_control", text="previous")
    return None


def _match_volume_action(text: str) -> PlannedGuiAction | None:
    if "volume" not in text:
        return None
    if re.search(r"\b(monte|augmente|plus fort)\b", text):
        return PlannedGuiAction(type="system_volume", text="volume_up")
    if re.search(r"\b(baisse|diminue|moins fort)\b", text):
        return PlannedGuiAction(type="system_volume", text="volume_down")
    if re.search(r"\b(coupe|muet|mute)\b", text):
        return PlannedGuiAction(type="system_volume", text="mute")
    return None


def _build_report(
    *,
    mode: SafetyMode,
    action: PlannedGuiAction,
    backend: LocalActionBackend,
    decision: CapabilityDecision | None = None,
) -> HandsExecutionReport:
    if decision is not None and not decision.can_execute_now:
        return HandsExecutionReport(
            status="blocked",
            mode=mode,
            actions=(action,),
            executed=False,
            requires_human=True,
            reason=decision.reason,
        )

    if action.destructive:
        return HandsExecutionReport(
            status="blocked",
            mode=mode,
            actions=(action,),
            executed=False,
            requires_human=True,
            reason="Action locale sensible: confirmation humaine requise",
        )

    if mode == "observe":
        return HandsExecutionReport(
            status="observe",
            mode=mode,
            actions=(action,),
            executed=False,
            requires_human=False,
            reason="Mode observe: action locale planifiee sans execution",
        )

    if mode == "dry_run":
        return HandsExecutionReport(
            status="dry_run",
            mode=mode,
            actions=(action,),
            executed=False,
            requires_human=False,
            reason="Mode dry_run: action locale journalisee sans execution",
        )

    if mode in {"assisted", "autonomous"}:
        try:
            backend.perform(action)
        except Exception as exc:
            return HandsExecutionReport(
                status="blocked",
                mode=mode,
                actions=(action,),
                executed=False,
                requires_human=True,
                reason=f"Action locale impossible: {type(exc).__name__}",
            )
        return HandsExecutionReport(
            status="completed",
            mode=mode,
            actions=(action,),
            executed=True,
            requires_human=False,
            reason=f"Mode {mode}: action locale executee",
        )

    return HandsExecutionReport(
        status="blocked",
        mode=mode,
        actions=(action,),
        executed=False,
        requires_human=True,
        reason="Actuators locaux non branches en iteration 5E",
    )


def _with_capability_decision(
    action: PlannedGuiAction,
    capabilities: CapabilityRegistryLike,
) -> _MatchedLocalAction:
    decision = _capability_decision_for_action(action, capabilities)
    if decision is not None:
        log.info(
            "local_action_capability_checked",
            action_type=action.type,
            target=action.text,
            available=decision.available,
            can_execute_now=decision.can_execute_now,
            requires_confirmation=decision.requires_confirmation,
            requires_admin=decision.requires_admin,
            destructive=decision.destructive,
            reason=decision.reason,
        )
    return _MatchedLocalAction(
        action=action,
        decision=decision,
    )


def _capability_decision_for_action(
    action: PlannedGuiAction,
    capabilities: CapabilityRegistryLike,
) -> CapabilityDecision | None:
    target = action.text
    if target is None:
        return None
    if action.type == "launch_app":
        return capabilities.can_execute(target_type="app", target_name=target, action="open")
    if action.type == "system_tool":
        return capabilities.can_execute(target_type="app", target_name=target, action="open")
    if action.type == "close_app":
        return capabilities.can_execute(target_type="app", target_name=target, action="close")
    if action.type == "open_folder":
        return capabilities.can_execute(target_type="folder", target_name=target, action="open")
    if action.type == "system_volume":
        return capabilities.can_execute(
            target_type="system",
            target_name="volume",
            action=_volume_capability_action(target),
        )
    if action.type == "media_control":
        return capabilities.can_execute(
            target_type="media",
            target_name="media_keys",
            action=_media_capability_action(target),
        )
    return None


def _media_capability_action(target: str) -> CapabilityAction:
    if target == "next":
        return "next"
    if target == "previous":
        return "previous"
    return "play_pause"


def _volume_capability_action(target: str) -> CapabilityAction:
    if target == "volume_up":
        return "volume_up"
    if target == "volume_down":
        return "volume_down"
    return "mute"


def _match_dynamic_open_target(
    text: str,
    *,
    domain: str,
    capabilities: CapabilityRegistryLike,
) -> tuple[CapabilityTargetType, str] | None:
    target_type: CapabilityTargetType = "folder" if domain == "folders" else "app"
    target = _extract_action_target(text)
    if target is None:
        return None
    return _match_dynamic_target_with_type(
        target,
        target_type=target_type,
        action="open",
        capabilities=capabilities,
    )


def _match_dynamic_target(
    text: str,
    *,
    target_type: CapabilityTargetType,
    action: CapabilityAction,
    capabilities: CapabilityRegistryLike,
) -> str | None:
    target = _extract_action_target(text)
    if target is None:
        return None
    matched = _match_dynamic_target_with_type(
        target,
        target_type=target_type,
        action=action,
        capabilities=capabilities,
    )
    if matched is None:
        return None
    return matched[1]


def _match_dynamic_target_with_type(
    target: str,
    *,
    target_type: CapabilityTargetType,
    action: CapabilityAction,
    capabilities: CapabilityRegistryLike,
) -> tuple[CapabilityTargetType, str] | None:
    matches = capabilities.find_capabilities(
        target_type=target_type,
        target_name=target,
        action=action,
    )
    if not matches:
        return None
    return matches[0].target_type, matches[0].target_name


def _extract_action_target(text: str) -> str | None:
    target = _OPEN_PATTERN.sub("", text, count=1)
    target = _CLOSE_PATTERN.sub("", target, count=1)
    target = re.sub(r"^(le|la|les|l'|un|une|des|du|de|d')\s+", "", target.strip())
    target = target.strip(" .,!?:;")
    return target or None


def _match_alias(text: str, aliases: tuple[tuple[tuple[str, ...], str], ...]) -> str | None:
    for candidates, canonical in aliases:
        if any(candidate in text for candidate in candidates):
            return canonical
    return None


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold().replace("\u2019", "'"))
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", stripped).strip()


def _required_text(action: PlannedGuiAction) -> str:
    if action.text is None:
        raise ValueError(f"Action locale sans cible: {action.type}")
    return action.text


def _pyautogui() -> _PyAutoGuiLike:
    import pyautogui

    return cast(_PyAutoGuiLike, pyautogui)
