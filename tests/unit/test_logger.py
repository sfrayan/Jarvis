"""Tests unitaires du module de logging.

On utilise `capsys` de pytest pour capturer stderr (structlog y écrit via son
`PrintLoggerFactory`). Un fixture autouse réinitialise structlog entre chaque
test pour que les reconfigurations soient effectives.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import structlog

from observability.logger import configure_logging, get_logger

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_structlog() -> Any:
    """Remet structlog à un état neutre après chaque test."""
    yield
    structlog.reset_defaults()


def _last_json_line(captured_err: str) -> dict[str, Any]:
    """Parse la dernière ligne non vide de stderr comme JSON."""
    lines = [line for line in captured_err.splitlines() if line.strip()]
    assert lines, "aucune sortie capturée"
    parsed: dict[str, Any] = json.loads(lines[-1])
    return parsed


# ----------------------------------------------------------------------
# configure_logging — validation d'entrée
# ----------------------------------------------------------------------
class TestConfigureLogging:
    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Niveau de log invalide"):
            configure_logging(level="NOPE")

    def test_level_is_case_insensitive(self) -> None:
        # Ne doit pas lever
        configure_logging(level="info", cache_logger=False)
        configure_logging(level="Debug", cache_logger=False)

    def test_idempotent(self) -> None:
        """Deux appels successifs ne provoquent pas d'erreur."""
        configure_logging(level="INFO", log_format="console", cache_logger=False)
        configure_logging(level="DEBUG", log_format="json", cache_logger=False)


# ----------------------------------------------------------------------
# Format console
# ----------------------------------------------------------------------
class TestConsoleFormat:
    def test_event_and_kwargs_appear_in_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level="INFO", log_format="console", cache_logger=False)
        log = get_logger("test.console")
        log.info("hello", foo="bar", number=42)
        err = capsys.readouterr().err
        assert "hello" in err
        assert "foo" in err
        assert "bar" in err
        assert "42" in err


# ----------------------------------------------------------------------
# Format JSON
# ----------------------------------------------------------------------
class TestJsonFormat:
    def test_output_is_valid_json_with_expected_fields(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level="INFO", log_format="json", cache_logger=False)
        log = get_logger("test.json")
        log.info("event_name", key="value", number=42)
        parsed = _last_json_line(capsys.readouterr().err)
        assert parsed["event"] == "event_name"
        assert parsed["key"] == "value"
        assert parsed["number"] == 42
        assert parsed["level"] == "info"
        assert "timestamp" in parsed

    def test_exception_info_serialized(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level="INFO", log_format="json", cache_logger=False)
        log = get_logger("test.exc")
        try:
            raise ValueError("boom")
        except ValueError:
            log.error("failure", exc_info=True)
        parsed = _last_json_line(capsys.readouterr().err)
        assert parsed["event"] == "failure"
        assert "ValueError" in parsed.get("exception", "")
        assert "boom" in parsed.get("exception", "")


# ----------------------------------------------------------------------
# Filtrage par niveau
# ----------------------------------------------------------------------
class TestLogLevelFiltering:
    def test_debug_suppressed_when_level_is_info(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level="INFO", log_format="json", cache_logger=False)
        log = get_logger("test.lvl")
        log.debug("should_not_appear")
        log.info("should_appear")
        err = capsys.readouterr().err
        assert "should_not_appear" not in err
        assert "should_appear" in err

    def test_debug_emitted_when_level_is_debug(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level="DEBUG", log_format="json", cache_logger=False)
        log = get_logger("test.lvl")
        log.debug("should_appear")
        err = capsys.readouterr().err
        assert "should_appear" in err

    def test_warning_still_emitted_when_level_is_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level="WARNING", log_format="json", cache_logger=False)
        log = get_logger("test.lvl")
        log.info("below")
        log.warning("at_level")
        log.error("above")
        err = capsys.readouterr().err
        assert "below" not in err
        assert "at_level" in err
        assert "above" in err


# ----------------------------------------------------------------------
# bind() / contexte
# ----------------------------------------------------------------------
class TestBoundContext:
    def test_bind_adds_keys_to_subsequent_logs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level="INFO", log_format="json", cache_logger=False)
        log = get_logger("test.ctx").bind(request_id="abc123", user="jarvis")
        log.info("action")
        parsed = _last_json_line(capsys.readouterr().err)
        assert parsed["request_id"] == "abc123"
        assert parsed["user"] == "jarvis"
        assert parsed["event"] == "action"
