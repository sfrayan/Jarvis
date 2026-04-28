"""Planification et execution locale pour les commandes PC simples.

Iteration 5G : ce module traduit certaines intentions `apps`, `system`, `media`
et `folders` en `HandsExecutionReport`. `observe` et `dry_run` restent sans
effet reel ; `assisted` et `autonomous` executent les actions locales sures.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Protocol, cast

from brain.events import IntentRouted
from config.schema import SafetyConfig, SafetyMode
from hands.executor import HandsExecutionReport, PlannedGuiAction

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
    (("vs code", "visual studio code"), "VS Code"),
    (("docker desktop",), "Docker Desktop"),
    (("keepass", "key pass", "qui passe"), "KeePass"),
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
    "VS Code": "Visual Studio Code",
    "Docker Desktop": "Docker Desktop",
    "KeePass": "KeePass",
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
        term = _APP_SEARCH_TERMS.get(app)
        if term is None:
            raise ValueError(f"Application locale non allowlistee: {app}")
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
    ) -> None:
        self._safety = safety
        self._backend = backend or PyAutoGuiLocalActionBackend()

    def plan(self, event: IntentRouted) -> HandsExecutionReport | None:
        """Retourne un rapport local si l'intention est reconnue."""
        if event.intent != "gui" or event.domain not in _LOCAL_DOMAINS:
            return None

        text = _fold(event.normalized_text)
        action = _match_local_action(text)
        if action is None:
            return None

        return _build_report(mode=self._safety.mode, action=action, backend=self._backend)


def _match_local_action(text: str) -> PlannedGuiAction | None:
    if _DESTRUCTIVE_SYSTEM_PATTERN.search(text):
        return PlannedGuiAction(
            type="system_command",
            text="shutdown",
            destructive=True,
        )

    if _CLOSE_PATTERN.search(text):
        app = _match_alias(text, _APP_ALIASES)
        if app is None:
            return None
        return PlannedGuiAction(type="close_app", text=app, destructive=True)

    media_action = _match_media_action(text)
    if media_action is not None:
        return media_action

    if _OPEN_PATTERN.search(text):
        folder = _match_alias(text, _FOLDER_ALIASES)
        if folder is not None:
            return PlannedGuiAction(type="open_folder", text=folder)

        app = _match_alias(text, _APP_ALIASES)
        if app is not None:
            if app == "Task Manager":
                return PlannedGuiAction(type="system_tool", text=app)
            return PlannedGuiAction(type="launch_app", text=app)

    volume_action = _match_volume_action(text)
    if volume_action is not None:
        return volume_action

    return None


def _match_media_action(text: str) -> PlannedGuiAction | None:
    if "spotify" not in text and "musique" not in text and "youtube" not in text:
        return None
    if _OPEN_PATTERN.search(text):
        return PlannedGuiAction(type="launch_app", text="Spotify")
    if re.search(r"\b(pause|mets en pause)\b", text):
        return PlannedGuiAction(type="media_control", text="pause")
    if re.search(r"\b(reprends|resume|play|mets de la musique|met de la musique)\b", text):
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
) -> HandsExecutionReport:
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
