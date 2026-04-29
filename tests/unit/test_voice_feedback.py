"""Tests unitaires du feedback assistant 5G-H."""

from __future__ import annotations

import pytest

from brain.events import IntentDomain, IntentRouted
from hands.executor import HandsExecutionReport, HandsExecutionStatus, PlannedGuiAction
from voice.feedback import feedback_for_unhandled_local_intent, feedback_from_hands_report

pytestmark = pytest.mark.unit


def _report(
    *,
    status: HandsExecutionStatus = "completed",
    executed: bool = True,
    requires_human: bool = False,
    reason: str = "test",
    action: PlannedGuiAction | None = None,
) -> HandsExecutionReport:
    return HandsExecutionReport(
        status=status,
        mode="assisted",
        actions=() if action is None else (action,),
        executed=executed,
        requires_human=requires_human,
        reason=reason,
    )


def _launch_app(name: str = "Antigravity") -> PlannedGuiAction:
    return PlannedGuiAction(type="launch_app", text=name)


def _intent(
    *,
    domain: IntentDomain = "apps",
    text: str = "ouvre Obsidian",
) -> IntentRouted:
    return IntentRouted(
        timestamp=1.0,
        original_text=text,
        normalized_text=text,
        intent="gui",
        domain=domain,
        confidence=0.9,
        reason="test",
        model="heuristic",
    )


def test_feedback_completed_executed_app_mentions_real_target() -> None:
    utterance = feedback_from_hands_report(_report(action=_launch_app()))

    assert utterance.text == "Je l'ai trouvée dans ton PC. J'ai ouvert Antigravity."
    assert utterance.priority == "info"
    assert utterance.source == "hands"


def test_feedback_dry_run_explains_no_execution() -> None:
    utterance = feedback_from_hands_report(
        _report(
            status="dry_run",
            executed=False,
            reason="Mode dry_run",
            action=_launch_app("Antigravity"),
        )
    )

    assert utterance.text == (
        "Je l'ai trouvée dans ton PC. En mode dry run, je n'exécute pas encore: "
        "ouvrir Antigravity."
    )
    assert utterance.reason == "Mode dry_run"


def test_feedback_blocked_requests_confirmation() -> None:
    utterance = feedback_from_hands_report(
        _report(
            status="blocked",
            executed=False,
            requires_human=True,
            reason="confirmation requise",
            action=PlannedGuiAction(type="close_app", text="Discord", destructive=True),
        )
    )

    assert utterance.text == (
        "J'ai besoin de ta confirmation avant de continuer: confirmation requise."
    )
    assert utterance.priority == "warning"


def test_unhandled_app_intent_explains_inventory_miss() -> None:
    utterance = feedback_for_unhandled_local_intent(_intent())

    assert utterance is not None
    assert utterance.text == "Je ne trouve pas cette application dans ton inventaire local."
    assert utterance.priority == "warning"


def test_unhandled_non_local_intent_is_ignored() -> None:
    utterance = feedback_for_unhandled_local_intent(_intent(domain="google_workspace"))

    assert utterance is None
