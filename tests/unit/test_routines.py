"""Tests unitaires des routines sures 5J."""

from __future__ import annotations

import pytest

from brain.routines import plan_routine

pytestmark = pytest.mark.unit


def test_mode_travail_returns_safe_work_plan() -> None:
    plan = plan_routine("Jarvis mode travail")

    assert plan is not None
    assert plan.kind == "work"
    assert plan.title == "Mode travail"
    assert len(plan.steps) == 3
    assert all(not suggestion.requires_confirmation for suggestion in plan.suggestions)
    assert "code, devoir, documents" in plan.next_question


def test_mode_code_suggests_tools_without_execution() -> None:
    plan = plan_routine("prepare un mode code")

    assert plan is not None
    assert plan.kind == "code"
    assert any(suggestion.command == "ouvre VS Code" for suggestion in plan.suggestions)
    assert "Attendre ton accord" in plan.steps[2]


def test_mode_recherche_mentions_google_and_youtube() -> None:
    plan = plan_routine("mode recherche web")

    assert plan is not None
    assert plan.kind == "research"
    assert {suggestion.kind for suggestion in plan.suggestions} == {"browser"}
    assert "Google ou YouTube" in plan.next_question


def test_unknown_routine_returns_none() -> None:
    assert plan_routine("bonjour Jarvis") is None
