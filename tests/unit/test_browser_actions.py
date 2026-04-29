"""Tests unitaires du planificateur navigateur."""

from __future__ import annotations

import pytest

from brain.events import IntentDomain, IntentRouted, IntentType
from config.schema import SafetyConfig, SafetyMode
from hands.browser_actions import BrowserActionPlanner
from hands.executor import PlannedGuiAction

pytestmark = pytest.mark.unit


def _intent(
    text: str,
    *,
    domain: IntentDomain = "web_search",
    intent: IntentType = "gui",
) -> IntentRouted:
    return IntentRouted(
        timestamp=1.0,
        original_text=text,
        normalized_text=text,
        intent=intent,
        domain=domain,
        confidence=0.9,
        reason="test",
        model="heuristic",
    )


class _FakeBrowserActionBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[PlannedGuiAction, ...]] = []
        self._fail = fail

    def perform(self, actions: tuple[PlannedGuiAction, ...]) -> None:
        self.calls.append(actions)
        if self._fail:
            raise RuntimeError("browser down")


def _planner(
    mode: SafetyMode = "dry_run",
    *,
    backend: _FakeBrowserActionBackend | None = None,
) -> BrowserActionPlanner:
    return BrowserActionPlanner(SafetyConfig(mode=mode), backend=backend)


def _navigate_action(actions: tuple[PlannedGuiAction, ...]) -> PlannedGuiAction:
    for action in actions:
        if action.type == "browser_navigate":
            return action
    raise AssertionError("browser_navigate action missing")


class TestBrowserActionPlanner:
    def test_observe_open_youtube_plans_without_execution(self) -> None:
        backend = _FakeBrowserActionBackend()
        planner = _planner("observe", backend=backend)

        report = planner.plan(_intent("ouvre YouTube"))

        assert report is not None
        assert report.status == "observe"
        assert report.executed is False
        assert report.requires_human is False
        assert _navigate_action(report.actions).text == "https://www.youtube.com"
        assert backend.calls == []

    def test_dry_run_google_search_builds_search_url(self) -> None:
        planner = _planner()

        report = planner.plan(_intent("cherche sur Google la météo"))

        assert report is not None
        assert report.status == "dry_run"
        assert report.executed is False
        assert _navigate_action(report.actions).text == ("https://www.google.com/search?q=la+meteo")

    def test_assisted_youtube_search_calls_backend(self) -> None:
        backend = _FakeBrowserActionBackend()
        planner = _planner("assisted", backend=backend)

        report = planner.plan(_intent("cherche sur YouTube cats"))

        assert report is not None
        assert report.status == "completed"
        assert report.executed is True
        assert report.requires_human is False
        assert len(backend.calls) == 1
        assert _navigate_action(backend.calls[0]).text == (
            "https://www.youtube.com/results?search_query=cats"
        )

    def test_autonomous_open_youtube_calls_backend(self) -> None:
        backend = _FakeBrowserActionBackend()
        planner = _planner("autonomous", backend=backend)

        report = planner.plan(_intent("ouvre YouTube"))

        assert report is not None
        assert report.status == "completed"
        assert report.executed is True
        assert len(backend.calls) == 1

    def test_non_browser_intent_returns_none(self) -> None:
        planner = _planner()

        report = planner.plan(_intent("ouvre Chrome", domain="apps"))

        assert report is None

    def test_system_or_destructive_command_returns_none(self) -> None:
        backend = _FakeBrowserActionBackend()
        planner = _planner("assisted", backend=backend)

        report = planner.plan(_intent("eteins le PC"))

        assert report is None
        assert backend.calls == []

    @pytest.mark.parametrize(
        "text",
        [
            "ouvre YouTube",
            "lance YouTube",
            "mets YouTube",
            "ouvre YouTube dans Chrome",
        ],
    )
    def test_youtube_never_launches_spotify(self, text: str) -> None:
        planner = _planner()

        report = planner.plan(_intent(text))

        assert report is not None
        assert all(action.type != "launch_app" for action in report.actions)
        assert all(action.text != "Spotify" for action in report.actions)
        assert _navigate_action(report.actions).text == "https://www.youtube.com"

    @pytest.mark.parametrize(
        ("text", "expected_url"),
        [
            (
                "cherche la météo sur Google",
                "https://www.google.com/search?q=la+meteo",
            ),
            (
                "recherche youtube musique relaxante",
                "https://www.youtube.com/results?search_query=musique+relaxante",
            ),
            (
                "ouvre une recherche YouTube sur chats calmes",
                "https://www.youtube.com/results?search_query=chats+calmes",
            ),
            (
                "chrome nouvel onglet recherche youtube cats",
                "https://www.youtube.com/results?search_query=cats",
            ),
        ],
    )
    def test_oral_variants_are_supported(self, text: str, expected_url: str) -> None:
        planner = _planner()

        report = planner.plan(_intent(text))

        assert report is not None
        assert _navigate_action(report.actions).text == expected_url

    def test_browser_workflow_opens_new_tab_then_youtube(self) -> None:
        planner = _planner()

        report = planner.plan(
            _intent("Chrome ouvre un nouvel onglet et fait une recherche sur YouTube")
        )

        assert report is not None
        assert [action.type for action in report.actions] == [
            "browser_open_chrome",
            "browser_new_tab",
            "browser_navigate",
        ]
        assert _navigate_action(report.actions).text == "https://www.youtube.com"

    def test_backend_failure_blocks_report(self) -> None:
        backend = _FakeBrowserActionBackend(fail=True)
        planner = _planner("assisted", backend=backend)

        report = planner.plan(_intent("ouvre YouTube"))

        assert report is not None
        assert report.status == "blocked"
        assert report.executed is False
        assert report.requires_human is True
        assert "RuntimeError" in report.reason
