"""Tests unitaires pour brain/confirmation.py.

Couvre :
- demande de confirmation a partir d'un rapport bloque ;
- confirmation vocale (oui, vas-y, confirme, etc.) ;
- rejet vocal (non, annule, refuse, etc.) ;
- expiration TTL ;
- phrase ambigue (ni confirmation ni rejet) ;
- pas de pending → rien ne se passe ;
- clear explicite.
"""

from __future__ import annotations

import pytest

from brain.confirmation import ConfirmationManager, _classify_reply, _confirmation_question
from brain.events import ConfirmationResponse, PendingConfirmation
from config.schema import SafetyConfig
from hands.executor import HandsExecutionReport, PlannedGuiAction


def _blocked_report(
    *,
    action_type: str = "close_app",
    action_target: str = "Discord",
    destructive: bool = True,
) -> HandsExecutionReport:
    """Fabrique un rapport bloque pour les tests."""
    return HandsExecutionReport(
        status="blocked",
        mode="assisted",
        actions=(
            PlannedGuiAction(
                type=action_type,
                text=action_target,
                destructive=destructive,
            ),
        ),
        executed=False,
        requires_human=True,
        reason="Action locale sensible: confirmation humaine requise",
    )


class TestRequestConfirmation:
    """Enregistrement d'une action en attente."""

    def test_request_returns_pending_confirmation(self) -> None:
        mgr = ConfirmationManager(clock=lambda: 1000.0, ttl_s=60.0)
        report = _blocked_report()

        pending = mgr.request_confirmation(report)

        assert isinstance(pending, PendingConfirmation)
        assert pending.action_type == "close_app"
        assert pending.action_target == "Discord"
        assert pending.expires_at == 1060.0
        assert "confirme" in pending.question.casefold() or "?" in pending.question

    def test_has_pending_after_request(self) -> None:
        mgr = ConfirmationManager(clock=lambda: 1000.0)
        mgr.request_confirmation(_blocked_report())
        assert mgr.has_pending is True

    def test_no_pending_initially(self) -> None:
        mgr = ConfirmationManager(clock=lambda: 1000.0)
        assert mgr.has_pending is False
        assert mgr.pending is None


class TestConfirmation:
    """Confirmation vocale par l'utilisateur."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "oui",
            "ok",
            "confirme",
            "vas-y",
            "fais-le",
            "go",
            "d'accord",
            "c'est bon",
            "oui, execute",
            "valide",
        ],
    )
    def test_confirmation_phrases_are_accepted(self, phrase: str) -> None:
        mgr = ConfirmationManager(clock=lambda: 1000.0)
        mgr.request_confirmation(_blocked_report())

        response = mgr.handle_user_reply(phrase)

        assert response is not None
        assert response.verdict == "confirmed"
        assert mgr.has_pending is False

    def test_confirmed_report_is_available(self) -> None:
        mgr = ConfirmationManager(clock=lambda: 1000.0)
        mgr.request_confirmation(_blocked_report())

        pending = mgr.pending
        assert pending is not None
        assert pending.report == _blocked_report()


class TestRejection:
    """Rejet vocal par l'utilisateur."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "non",
            "annule",
            "refuse",
            "stop",
            "laisse tomber",
            "oublie",
            "pas maintenant",
            "ne fais rien",
        ],
    )
    def test_rejection_phrases_are_accepted(self, phrase: str) -> None:
        mgr = ConfirmationManager(clock=lambda: 1000.0)
        mgr.request_confirmation(_blocked_report())

        response = mgr.handle_user_reply(phrase)

        assert response is not None
        assert response.verdict == "rejected"
        assert mgr.has_pending is False


class TestExpiration:
    """TTL depasse."""

    def test_expired_pending_returns_none(self) -> None:
        clock_value = 1000.0

        def clock() -> float:
            return clock_value

        mgr = ConfirmationManager(clock=clock, ttl_s=30.0)
        mgr.request_confirmation(_blocked_report())
        assert mgr.has_pending is True

        clock_value = 1050.0  # 50s > 30s TTL
        assert mgr.has_pending is False
        assert mgr.pending is None

    def test_expire_if_needed_returns_response(self) -> None:
        clock_value = 1000.0

        def clock() -> float:
            return clock_value

        mgr = ConfirmationManager(clock=clock, ttl_s=10.0)
        mgr.request_confirmation(_blocked_report())

        clock_value = 1020.0
        response = mgr.expire_if_needed()

        assert response is not None
        assert response.verdict == "expired"

    def test_expire_if_needed_noop_when_fresh(self) -> None:
        mgr = ConfirmationManager(clock=lambda: 1000.0, ttl_s=60.0)
        mgr.request_confirmation(_blocked_report())
        assert mgr.expire_if_needed() is None


class TestAmbiguousReply:
    """Phrase qui n'est ni oui ni non."""

    def test_ambiguous_reply_returns_none(self) -> None:
        mgr = ConfirmationManager(clock=lambda: 1000.0)
        mgr.request_confirmation(_blocked_report())

        response = mgr.handle_user_reply("je ne sais pas trop")

        assert response is None
        assert mgr.has_pending is True  # toujours en attente

    def test_reply_without_pending_returns_none(self) -> None:
        mgr = ConfirmationManager(clock=lambda: 1000.0)
        assert mgr.handle_user_reply("oui") is None


class TestClear:
    """Clear explicite."""

    def test_clear_removes_pending(self) -> None:
        mgr = ConfirmationManager(clock=lambda: 1000.0)
        mgr.request_confirmation(_blocked_report())
        mgr.clear()
        assert mgr.has_pending is False


class TestClassifyReply:
    """Tests directs de _classify_reply."""

    def test_confirm_word(self) -> None:
        assert _classify_reply("oui bien sur") == "confirmed"

    def test_reject_word(self) -> None:
        assert _classify_reply("non merci") == "rejected"

    def test_neutral_word(self) -> None:
        assert _classify_reply("peut-etre") is None


class TestConfirmationQuestion:
    """Generation de la question adaptee."""

    def test_close_app_question(self) -> None:
        q = _confirmation_question("close_app", "Discord", "test")
        assert "Discord" in q
        assert "ferme" in q.casefold()

    def test_system_command_question(self) -> None:
        q = _confirmation_question("system_command", "shutdown", "test")
        assert "sensible" in q.casefold()

    def test_launch_app_question(self) -> None:
        q = _confirmation_question("launch_app", "Chrome", "test")
        assert "Chrome" in q

    def test_generic_question(self) -> None:
        q = _confirmation_question("unknown_type", None, "raison test")
        assert "raison test" in q
