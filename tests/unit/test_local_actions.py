"""Tests unitaires du planificateur local."""

from __future__ import annotations

import pytest

from brain.events import IntentDomain, IntentRouted, IntentType
from config.schema import SafetyConfig, SafetyMode
from hands.executor import PlannedGuiAction
from hands.local_actions import LocalActionPlanner

pytestmark = pytest.mark.unit


def _intent(
    text: str,
    *,
    domain: IntentDomain,
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


class _FakeLocalActionBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.actions: list[PlannedGuiAction] = []
        self._fail = fail

    def perform(self, action: PlannedGuiAction) -> None:
        self.actions.append(action)
        if self._fail:
            raise RuntimeError("backend down")


class TestLocalActionPlanner:
    @pytest.mark.parametrize(
        ("text", "domain", "action_type", "target"),
        [
            ("ouvre Spotify", "media", "launch_app", "Spotify"),
            ("ouvre Chrome", "apps", "launch_app", "Chrome"),
            ("ouvre Discord", "apps", "launch_app", "Discord"),
            ("ouvre le gestionnaire de tâches", "system", "system_tool", "Task Manager"),
            ("ouvre mes téléchargements", "folders", "open_folder", "Downloads"),
            ("volume monte", "system", "system_volume", "volume_up"),
            ("mets Spotify en pause", "media", "media_control", "pause"),
        ],
    )
    def test_plans_supported_local_actions(
        self,
        text: str,
        domain: IntentDomain,
        action_type: str,
        target: str,
    ) -> None:
        planner = LocalActionPlanner(SafetyConfig(mode="dry_run"))

        report = planner.plan(_intent(text, domain=domain))

        assert report is not None
        assert report.status == "dry_run"
        assert report.executed is False
        assert report.requires_human is False
        assert report.actions[0].type == action_type
        assert report.actions[0].text == target

    def test_shutdown_pc_is_blocked(self) -> None:
        backend = _FakeLocalActionBackend()
        planner = LocalActionPlanner(SafetyConfig(mode="assisted"), backend=backend)

        report = planner.plan(_intent("éteins le PC", domain="system"))

        assert report is not None
        assert report.status == "blocked"
        assert report.requires_human is True
        assert report.actions[0].destructive is True
        assert backend.actions == []

    @pytest.mark.parametrize("mode", ["observe", "dry_run"])
    def test_non_execution_modes_do_not_call_backend(self, mode: SafetyMode) -> None:
        backend = _FakeLocalActionBackend()
        planner = LocalActionPlanner(SafetyConfig(mode=mode), backend=backend)

        report = planner.plan(_intent("ouvre Chrome", domain="apps"))

        assert report is not None
        assert report.executed is False
        if mode == "observe":
            assert report.status == "observe"
        else:
            assert report.status == "dry_run"
        assert report.requires_human is False
        assert backend.actions == []

    @pytest.mark.parametrize("mode", ["assisted", "autonomous"])
    def test_execution_modes_call_backend_for_safe_actions(self, mode: SafetyMode) -> None:
        backend = _FakeLocalActionBackend()
        planner = LocalActionPlanner(SafetyConfig(mode=mode), backend=backend)

        report = planner.plan(_intent("ouvre Chrome", domain="apps"))

        assert report is not None
        assert report.status == "completed"
        assert report.executed is True
        assert report.requires_human is False
        assert len(backend.actions) == 1
        assert backend.actions[0].type == "launch_app"
        assert backend.actions[0].text == "Chrome"

    def test_backend_failure_blocks_action(self) -> None:
        backend = _FakeLocalActionBackend(fail=True)
        planner = LocalActionPlanner(SafetyConfig(mode="assisted"), backend=backend)

        report = planner.plan(_intent("ouvre Chrome", domain="apps"))

        assert report is not None
        assert report.status == "blocked"
        assert report.executed is False
        assert report.requires_human is True
        assert "RuntimeError" in report.reason
        assert len(backend.actions) == 1

    def test_unsupported_domain_returns_none(self) -> None:
        planner = LocalActionPlanner(SafetyConfig(mode="dry_run"))

        report = planner.plan(_intent("ouvre Gmail", domain="google_workspace"))

        assert report is None

    def test_chat_intent_returns_none(self) -> None:
        planner = LocalActionPlanner(SafetyConfig(mode="dry_run"))

        report = planner.plan(_intent("ouvre Chrome", domain="apps", intent="chat"))

        assert report is None

    def test_unknown_local_command_returns_none(self) -> None:
        planner = LocalActionPlanner(SafetyConfig(mode="dry_run"))

        report = planner.plan(_intent("fais un truc local", domain="system"))

        assert report is None
