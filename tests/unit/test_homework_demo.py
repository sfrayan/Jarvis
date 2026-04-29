"""Demo 5K: aide devoir jusqu'a une recherche navigateur sure."""

from __future__ import annotations

import pytest

from brain.dialogue import DialogueManager
from brain.events import IntentDomain, IntentRouted, IntentType
from config.schema import SafetyConfig
from hands.browser_actions import BrowserActionPlanner
from hands.executor import PlannedGuiAction

pytestmark = pytest.mark.unit


def _intent(
    text: str,
    *,
    intent: IntentType = "chat",
    domain: IntentDomain = "general",
) -> IntentRouted:
    return IntentRouted(
        timestamp=1.0,
        original_text=text,
        normalized_text=text,
        intent=intent,
        domain=domain,
        confidence=0.9,
        reason="demo",
        model="test",
    )


class _Clock:
    def __init__(self) -> None:
        self.value = 20.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


def _navigate_action(actions: tuple[PlannedGuiAction, ...]) -> PlannedGuiAction:
    for action in actions:
        if action.type == "browser_navigate":
            return action
    raise AssertionError("browser_navigate action missing")


def test_homework_help_demo_reaches_browser_dry_run_search() -> None:
    manager = DialogueManager(clock=_Clock())

    first = manager.handle(_intent("Jarvis j'ai un devoir a faire"))
    second = manager.handle(
        _intent("Consigne: exercice sur les fonctions en maths niveau seconde pour demain")
    )
    third = manager.handle(_intent("commence par une recherche Google"))

    assert first.decision == "clarify"
    assert first.clarification is not None
    assert "consigne exacte" in first.clarification.question
    assert second.decision == "plan"
    assert second.plan is not None
    assert second.plan.kind == "homework"
    assert third.decision == "pass_through"
    assert third.intent is not None
    assert third.intent.domain == "web_search"

    planner = BrowserActionPlanner(SafetyConfig(mode="dry_run"))
    report = planner.plan(third.intent)

    assert report is not None
    assert report.status == "dry_run"
    assert report.executed is False
    assert _navigate_action(report.actions).text == (
        "https://www.google.com/search?"
        "q=exercice+sur+les+fonctions+en+maths+niveau+seconde+pour+demain"
    )
