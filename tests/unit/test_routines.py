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


# ------------------------------------------------------------------
# Tests match_routine_suggestion (5R)
# ------------------------------------------------------------------
from brain.routines import match_routine_suggestion


class TestMatchRoutineSuggestion:
    """Matching des reponses utilisateur contre les suggestions."""

    def test_code_ouvre_vscode_matches_local(self) -> None:
        match = match_routine_suggestion("ouvre VS Code", routine_kind="code")
        assert match is not None
        assert match.suggestion.kind == "local"
        assert match.normalized_command == "ouvre VS Code"

    def test_code_vscode_partial_matches(self) -> None:
        match = match_routine_suggestion("lance VS Code s'il te plait", routine_kind="code")
        assert match is not None
        assert match.suggestion.kind == "local"

    def test_research_google_matches_browser(self) -> None:
        match = match_routine_suggestion(
            "ouvre une recherche Google sur Python",
            routine_kind="research",
        )
        assert match is not None
        assert match.suggestion.kind == "browser"

    def test_research_youtube_matches_browser(self) -> None:
        match = match_routine_suggestion(
            "ouvre une recherche YouTube sur les chats",
            routine_kind="research",
        )
        assert match is not None
        assert match.suggestion.kind == "browser"

    def test_dialogue_suggestion_not_matched(self) -> None:
        """Les suggestions dialogue ne sont pas executables."""
        match = match_routine_suggestion(
            "web, desktop, bot, script Python ou autre",
            routine_kind="code",
        )
        # La suggestion "Choisir le type de projet" est kind=dialogue -> ignoree
        assert match is None

    def test_no_match_returns_none(self) -> None:
        match = match_routine_suggestion(
            "raconte-moi une blague",
            routine_kind="code",
        )
        assert match is None

    def test_unknown_routine_kind_returns_none(self) -> None:
        match = match_routine_suggestion("ouvre VS Code", routine_kind="unknown_kind")
        assert match is None

    def test_work_has_no_executable_suggestions(self) -> None:
        """La routine work n'a que des suggestions dialogue."""
        match = match_routine_suggestion("code", routine_kind="work")
        assert match is None
