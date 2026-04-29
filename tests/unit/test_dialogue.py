"""Tests unitaires du dialogue interactif 5H."""

from __future__ import annotations

import pytest

from brain.dialogue import DialogueManager
from brain.events import IntentDomain, IntentRouted, IntentType

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
        reason="test",
        model="test",
    )


class _Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class TestDialogueManager:
    def test_vague_homework_request_asks_for_clarification(self) -> None:
        manager = DialogueManager(clock=_Clock())

        turn = manager.handle(_intent("Jarvis j'ai un devoir a faire"))

        assert turn.decision == "clarify"
        assert turn.intent is None
        assert turn.utterance is not None
        assert "consigne exacte" in turn.utterance.text
        assert turn.clarification is not None
        assert turn.clarification.kind == "homework"
        session = manager.active_session
        assert session is not None
        assert session.kind == "homework"

    def test_coding_project_request_asks_for_application_type(self) -> None:
        manager = DialogueManager(clock=_Clock())

        turn = manager.handle(_intent("aide-moi a coder une app"))

        assert turn.decision == "clarify"
        assert turn.intent is None
        assert turn.utterance is not None
        assert "web, desktop, bot, script Python" in turn.utterance.text
        session = manager.active_session
        assert session is not None
        assert session.kind == "coding_project"

    def test_clear_web_search_passes_through(self) -> None:
        manager = DialogueManager(clock=_Clock())
        event = _intent(
            "cherche une video YouTube sur les chats",
            intent="gui",
            domain="web_search",
        )

        turn = manager.handle(event)

        assert turn.decision == "pass_through"
        assert turn.intent == event
        assert turn.utterance is None

    def test_clear_local_action_passes_through(self) -> None:
        manager = DialogueManager(clock=_Clock())
        event = _intent("ouvre Chrome", intent="gui", domain="apps")

        turn = manager.handle(event)

        assert turn.decision == "pass_through"
        assert turn.intent == event
        assert turn.utterance is None

    def test_media_stop_is_not_treated_as_session_cancel(self) -> None:
        manager = DialogueManager(clock=_Clock())
        event = _intent("stop la musique", intent="gui", domain="media")

        turn = manager.handle(event)

        assert turn.decision == "pass_through"
        assert turn.intent == event
        assert turn.utterance is None

    def test_homework_session_reply_completes_session_with_plan(self) -> None:
        manager = DialogueManager(clock=_Clock())
        manager.handle(_intent("j'ai un devoir a faire"))

        turn = manager.handle(
            _intent(
                "Consigne: exercice sur les fonctions en maths niveau seconde pour demain",
            )
        )

        assert turn.decision == "plan"
        assert turn.plan is not None
        assert turn.plan.kind == "homework"
        assert len(turn.plan.steps) == 3
        assert turn.utterance is not None
        assert turn.utterance.text.startswith("D'accord. Je te propose 3 etapes:")
        assert "commence par le brouillon" in turn.utterance.text
        session = manager.active_session
        assert session is not None
        assert session.status == "ready"

    def test_ready_homework_session_can_start_google_search(self) -> None:
        manager = DialogueManager(clock=_Clock())
        manager.handle(_intent("j'ai un devoir a faire"))
        manager.handle(
            _intent(
                "Consigne: exercice sur les fonctions en maths niveau seconde pour demain",
            )
        )

        turn = manager.handle(_intent("commence par une recherche Google"))

        assert turn.decision == "pass_through"
        assert turn.intent is not None
        assert turn.intent.intent == "gui"
        assert turn.intent.domain == "web_search"
        assert turn.intent.normalized_text == (
            "cherche sur Google exercice sur les fonctions en maths niveau seconde pour demain"
        )
        session = manager.active_session
        assert session is not None
        assert session.status == "ready"

    def test_ready_homework_session_can_start_draft_without_hands(self) -> None:
        manager = DialogueManager(clock=_Clock())
        manager.handle(_intent("j'ai un devoir a faire"))
        manager.handle(
            _intent(
                "Consigne: exercice sur les fonctions en maths niveau seconde pour demain",
            )
        )

        turn = manager.handle(_intent("commence par le brouillon"))

        assert turn.decision == "plan"
        assert turn.intent is None
        assert turn.plan is not None
        assert turn.plan.requires_confirmation is False
        assert turn.utterance is not None
        assert "brouillon structure" in turn.utterance.text
        assert "fonctions" in turn.utterance.text

    def test_routine_mode_code_produces_plan_without_hands_intent(self) -> None:
        manager = DialogueManager(clock=_Clock())

        turn = manager.handle(_intent("mode code", intent="gui", domain="routine"))

        assert turn.decision == "plan"
        assert turn.intent is None
        assert turn.plan is not None
        assert turn.plan.summary.startswith("Preparation prudente")
        assert turn.utterance is not None
        assert "Mode code" in turn.utterance.text
        assert "Ouvrir VS Code" in turn.utterance.text

    def test_routine_mode_research_produces_safe_plan(self) -> None:
        manager = DialogueManager(clock=_Clock())

        turn = manager.handle(_intent("mode recherche", intent="gui", domain="routine"))

        assert turn.decision == "plan"
        assert turn.intent is None
        assert turn.plan is not None
        assert "recherche web" in turn.plan.summary
        assert turn.utterance is not None
        assert "Google ou YouTube" in turn.utterance.text

    def test_vague_video_session_builds_safe_youtube_search_after_reply(self) -> None:
        manager = DialogueManager(clock=_Clock())
        manager.handle(_intent("cherche moi une video sur ca", intent="unknown", domain="unknown"))

        turn = manager.handle(_intent("les chats"))

        assert turn.decision == "pass_through"
        assert turn.intent is not None
        assert turn.intent.intent == "gui"
        assert turn.intent.domain == "web_search"
        assert turn.intent.normalized_text == "ouvre une recherche YouTube sur les chats"
        assert manager.active_session is None

    def test_cancel_resets_active_session(self) -> None:
        manager = DialogueManager(clock=_Clock())
        manager.handle(_intent("aide-moi a coder une app"))

        turn = manager.handle(_intent("annule"))

        assert turn.decision == "cancel"
        assert turn.utterance is not None
        assert manager.active_session is None

    def test_sensitive_windows_service_never_passes_to_hands(self) -> None:
        manager = DialogueManager(clock=_Clock())

        turn = manager.handle(
            _intent(
                "desactive un service Windows",
                intent="gui",
                domain="system",
            )
        )

        assert turn.decision == "clarify"
        assert turn.intent is None
        assert turn.utterance is not None
        assert turn.utterance.priority == "warning"
        assert "confirmation explicite" in turn.utterance.text
