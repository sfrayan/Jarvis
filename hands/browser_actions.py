"""Planification et execution locale des actions navigateur sures.

Iteration 5G-L : ce module traite les intentions `web_search` sans vision.
Il ouvre Chrome et navigue vers une URL connue ou une recherche web simple.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Protocol, cast
from urllib.parse import quote_plus

from brain.events import IntentRouted
from config.schema import SafetyConfig, SafetyMode
from hands.executor import HandsExecutionReport, PlannedGuiAction
from observability.logger import get_logger

log = get_logger(__name__)

_UNSAFE_PATTERN = re.compile(
    r"\b(supprime|supprimer|efface|effacer|desinstalle|desinstaller|"
    r"eteins|eteint|eteindre|redemarre|reboot|shutdown|ferme|quitte)\b"
)
_YOUTUBE_SITE_PATTERN = r"(?:you\s*tube|youtube(?:\.com)?)"
_GOOGLE_SITE_PATTERN = r"(?:google(?:\.com)?)"
_NEW_TAB_PATTERN = re.compile(r"\b(nouvel|nouveau|nouvelle)\s+onglet\b|\bonglet\b")


class BrowserActionBackend(Protocol):
    """Contrat minimal d'un actuateur navigateur injectable."""

    def perform(self, actions: tuple[PlannedGuiAction, ...]) -> None:
        """Execute une sequence d'actions navigateur deja validees."""


class _PyAutoGuiLike(Protocol):
    def hotkey(self, *keys: str) -> None:
        """Appuie sur une combinaison de touches."""

    def write(self, message: str, interval: float = 0.0) -> None:
        """Tape du texte."""

    def press(self, key: str) -> None:
        """Appuie sur une touche."""


class PyAutoGuiBrowserActionBackend:
    """Actuateur Chrome minimal base sur pyautogui."""

    def perform(self, actions: tuple[PlannedGuiAction, ...]) -> None:
        """Execute les actions navigateur supportees."""
        if not actions:
            return

        pyautogui = _pyautogui()
        self._open_chrome(pyautogui)
        for action in actions:
            if action.type == "browser_open_chrome":
                continue
            if action.type == "browser_new_tab":
                pyautogui.hotkey("ctrl", "t")
                time.sleep(0.1)
                continue
            if action.type == "browser_navigate":
                self._navigate(pyautogui, _required_text(action))
                continue
            raise ValueError(f"Action navigateur non supportee: {action.type}")

    def _open_chrome(self, pyautogui: _PyAutoGuiLike) -> None:
        pyautogui.hotkey("win")
        time.sleep(0.15)
        pyautogui.write("Google Chrome", interval=0.01)
        pyautogui.press("enter")
        time.sleep(0.35)

    def _navigate(self, pyautogui: _PyAutoGuiLike, url: str) -> None:
        pyautogui.hotkey("ctrl", "l")
        time.sleep(0.05)
        pyautogui.write(url, interval=0.01)
        pyautogui.press("enter")


class BrowserActionPlanner:
    """Produit et execute les commandes navigateur sans vision."""

    def __init__(
        self,
        safety: SafetyConfig,
        *,
        backend: BrowserActionBackend | None = None,
    ) -> None:
        self._safety = safety
        self._backend = backend or PyAutoGuiBrowserActionBackend()

    def plan(self, event: IntentRouted) -> HandsExecutionReport | None:
        """Retourne un rapport navigateur si l'intention est supportee."""
        if event.intent != "gui" or event.domain != "web_search":
            return None

        text = _fold(event.normalized_text)
        actions = _match_browser_actions(text)
        if actions is None:
            return None

        return _build_report(
            mode=self._safety.mode,
            actions=actions,
            backend=self._backend,
        )


def _match_browser_actions(text: str) -> tuple[PlannedGuiAction, ...] | None:
    if _UNSAFE_PATTERN.search(text):
        return None

    url = _target_url(text)
    wants_new_tab = _wants_new_tab(text)
    if url is None and not wants_new_tab:
        return None

    actions = [PlannedGuiAction(type="browser_open_chrome", text="Chrome")]
    if wants_new_tab:
        actions.append(PlannedGuiAction(type="browser_new_tab"))
    if url is not None:
        actions.append(PlannedGuiAction(type="browser_navigate", text=url))
    return tuple(actions)


def _target_url(text: str) -> str | None:
    site = _known_site(text)
    if site == "youtube":
        query = _extract_site_query(text, site_pattern=_YOUTUBE_SITE_PATTERN)
        if query is not None:
            return _youtube_search_url(query)
        return "https://www.youtube.com"

    if site == "google":
        query = _extract_site_query(text, site_pattern=_GOOGLE_SITE_PATTERN)
        if query is not None:
            return _google_search_url(query)
        return "https://www.google.com"

    query = _extract_generic_search_query(text)
    if query is not None:
        return _google_search_url(query)
    return None


def _known_site(text: str) -> str | None:
    if re.search(rf"\b{_YOUTUBE_SITE_PATTERN}\b", text):
        return "youtube"
    if re.search(rf"\b{_GOOGLE_SITE_PATTERN}\b", text):
        return "google"
    return None


def _extract_site_query(text: str, *, site_pattern: str) -> str | None:
    patterns = (
        rf"\b(?:cherche|recherche|recherches)\s+(?P<query>.+?)\s+sur\s+{site_pattern}\b",
        rf"\b(?:cherche|recherche|recherches)\s+sur\s+{site_pattern}\s+(?P<query>.+)$",
        rf"\bouvre\s+une\s+recherche\s+{site_pattern}\s+sur\s+(?P<query>.+)$",
        rf"\b(?:cherche|recherche|recherches)\s+{site_pattern}\s+(?P<query>.+)$",
        rf"\b(?:fais|fait)\s+une\s+recherche\s+sur\s+{site_pattern}\s+(?P<query>.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is not None:
            return _clean_query(match.group("query"))
    return None


def _extract_generic_search_query(text: str) -> str | None:
    match = re.search(r"\b(?:cherche|recherche|recherches)\s+(?P<query>.+)$", text)
    if match is None:
        return None
    return _clean_query(match.group("query"))


def _clean_query(query: str) -> str | None:
    cleaned = re.sub(r"\b(?:dans|avec)\s+chrome\b.*$", "", query)
    cleaned = re.sub(r"\b(?:sur|dans)\s+(?:google|youtube|you tube)(?:\.com)?\b", "", cleaned)
    cleaned = re.sub(r"^(?:une|un)\s+recherche\s+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,!?:;")
    if not cleaned or cleaned in {"google", "google.com", "youtube", "youtube.com", "you tube"}:
        return None
    return cleaned


def _wants_new_tab(text: str) -> bool:
    return bool(_NEW_TAB_PATTERN.search(text))


def _google_search_url(query: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(query)}"


def _youtube_search_url(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"


def _build_report(
    *,
    mode: SafetyMode,
    actions: tuple[PlannedGuiAction, ...],
    backend: BrowserActionBackend,
) -> HandsExecutionReport:
    if mode == "observe":
        return HandsExecutionReport(
            status="observe",
            mode=mode,
            actions=actions,
            executed=False,
            requires_human=False,
            reason="Mode observe: action navigateur planifiee sans execution",
        )

    if mode == "dry_run":
        return HandsExecutionReport(
            status="dry_run",
            mode=mode,
            actions=actions,
            executed=False,
            requires_human=False,
            reason="Mode dry_run: action navigateur journalisee sans execution",
        )

    if mode in {"assisted", "autonomous"}:
        try:
            backend.perform(actions)
        except Exception as exc:
            return HandsExecutionReport(
                status="blocked",
                mode=mode,
                actions=actions,
                executed=False,
                requires_human=True,
                reason=f"Action navigateur impossible: {type(exc).__name__}",
            )
        return HandsExecutionReport(
            status="completed",
            mode=mode,
            actions=actions,
            executed=True,
            requires_human=False,
            reason=f"Mode {mode}: action navigateur executee",
        )

    return HandsExecutionReport(
        status="blocked",
        mode=mode,
        actions=actions,
        executed=False,
        requires_human=True,
        reason="Mode de securite inconnu pour l'action navigateur",
    )


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold().replace("\u2019", "'"))
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", stripped).strip()


def _required_text(action: PlannedGuiAction) -> str:
    if action.text is None:
        raise ValueError(f"Action navigateur sans URL: {action.type}")
    return action.text


def _pyautogui() -> _PyAutoGuiLike:
    import pyautogui

    return cast(_PyAutoGuiLike, pyautogui)
