"""Tests unitaires du planificateur local dry-run."""

from __future__ import annotations

import pytest

from brain.events import IntentDomain, IntentRouted, IntentType
from config.schema import SafetyConfig, SafetyMode
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
        planner = LocalActionPlanner(SafetyConfig(mode="dry_run"))

        report = planner.plan(_intent("éteins le PC", domain="system"))

        assert report is not None
        assert report.status == "blocked"
        assert report.requires_human is True
        assert report.actions[0].destructive is True

    @pytest.mark.parametrize("mode", ["observe", "assisted", "autonomous"])
    def test_safety_modes_are_respected(self, mode: SafetyMode) -> None:
        planner = LocalActionPlanner(SafetyConfig(mode=mode))

        report = planner.plan(_intent("ouvre Chrome", domain="apps"))

        assert report is not None
        assert report.executed is False
        if mode == "observe":
            assert report.status == "observe"
            assert report.requires_human is False
        else:
            assert report.status == "blocked"
            assert report.requires_human is True

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
