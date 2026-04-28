"""Tests unitaires de l'executeur Hands dry-run."""

from __future__ import annotations

import pytest

from brain.vision_contracts import VisionAction, VisionDecision
from config.schema import SafetyConfig, SafetyMode
from hands.executor import DryRunHandsExecutor
from hands.screenshot import ScreenshotFrame

pytestmark = pytest.mark.unit


def _decision(
    *,
    actions: list[VisionAction] | None = None,
    requires_human: bool = False,
    task_complete: bool = False,
) -> VisionDecision:
    return VisionDecision(
        thought="Je vois le bouton.",
        confidence=0.9,
        speech="Je prepare l'action.",
        actions=actions or [],
        external_tools=[],
        requires_human=requires_human,
        task_complete=task_complete,
    )


def _frame() -> ScreenshotFrame:
    return ScreenshotFrame(
        image_base64="abc",
        width=100,
        height=50,
        original_width=200,
        original_height=100,
        left=10,
        top=20,
    )


class TestDryRunHandsExecutor:
    def test_dry_run_plans_actions_without_execution(self) -> None:
        executor = DryRunHandsExecutor(SafetyConfig(mode="dry_run"))
        decision = _decision(
            actions=[
                VisionAction(type="left_click", x=25, y=10),
                VisionAction(type="type_text", text="bonjour"),
            ]
        )

        report = executor.execute(decision, frame=_frame())

        assert report.status == "dry_run"
        assert report.executed is False
        assert report.requires_human is False
        assert len(report.actions) == 2
        assert report.actions[0].x == 60
        assert report.actions[0].y == 40
        assert report.actions[1].text == "bonjour"

    def test_observe_mode_never_executes(self) -> None:
        executor = DryRunHandsExecutor(SafetyConfig(mode="observe"))
        decision = _decision(actions=[VisionAction(type="mouse_move", x=10, y=20)])

        report = executor.execute(decision)

        assert report.status == "observe"
        assert report.executed is False
        assert report.actions[0].x == 10
        assert report.actions[0].y == 20

    def test_task_complete_discards_actions(self) -> None:
        executor = DryRunHandsExecutor(SafetyConfig(mode="dry_run"))
        decision = _decision(
            actions=[VisionAction(type="left_click", x=1, y=2)],
            task_complete=True,
        )

        report = executor.execute(decision)

        assert report.status == "completed"
        assert report.actions == ()
        assert report.requires_human is False

    def test_requires_human_blocks_actions(self) -> None:
        executor = DryRunHandsExecutor(SafetyConfig(mode="dry_run"))
        decision = _decision(
            actions=[],
            requires_human=True,
        )

        report = executor.execute(decision)

        assert report.status == "blocked"
        assert report.requires_human is True
        assert "Confirmation humaine" in report.reason

    def test_destructive_text_blocks_even_in_dry_run(self) -> None:
        executor = DryRunHandsExecutor(SafetyConfig(mode="dry_run"))
        decision = _decision(
            actions=[VisionAction(type="type_text", text="shutdown /s")],
            requires_human=True,
        )

        report = executor.execute(decision)

        assert report.status == "blocked"
        assert report.requires_human is True
        assert report.actions[0].destructive is True

    def test_configured_destructive_pattern_blocks_action(self) -> None:
        executor = DryRunHandsExecutor(
            SafetyConfig(mode="dry_run", allowlist_destructive_patterns=["danger"])
        )
        decision = _decision(actions=[VisionAction(type="type_text", text="commande danger")])

        report = executor.execute(decision)

        assert report.status == "blocked"
        assert report.actions[0].destructive is True

    @pytest.mark.parametrize("mode", ["assisted", "autonomous"])
    def test_real_execution_modes_are_blocked_in_iteration_5b(self, mode: SafetyMode) -> None:
        executor = DryRunHandsExecutor(SafetyConfig(mode=mode))
        decision = _decision(actions=[VisionAction(type="left_click", x=10, y=20)])

        report = executor.execute(decision)

        assert report.status == "blocked"
        assert report.executed is False
        assert report.requires_human is True
        assert "5B" in report.reason

    def test_non_coordinate_actions_keep_payload(self) -> None:
        executor = DryRunHandsExecutor(SafetyConfig(mode="dry_run"))
        decision = _decision(
            actions=[
                VisionAction(type="scroll", amount=-3),
                VisionAction(type="key_combo", keys=["ctrl", "l"]),
                VisionAction(type="wait", duration_ms=250),
            ]
        )

        report = executor.execute(decision)

        assert report.actions[0].amount == -3
        assert report.actions[1].keys == ("ctrl", "l")
        assert report.actions[2].duration_ms == 250
