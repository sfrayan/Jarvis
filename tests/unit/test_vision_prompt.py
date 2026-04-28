"""Tests de garde-fou du prompt système vision."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROMPT_PATH = Path("brain/prompts/vision_system.txt")


def _prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def test_vision_prompt_exists_and_is_non_empty() -> None:
    assert PROMPT_PATH.exists()
    assert len(_prompt().strip()) > 100


@pytest.mark.parametrize(
    "required",
    [
        "JSON valide",
        "confidence < 0.6",
        "actions doit être []",
        "requires_human doit être true",
        "coordonnées x/y",
        "mouse_move",
        "left_click",
        "right_click",
        "double_click",
        "scroll",
        "type_text",
        "key_combo",
        "wait",
        "rm/del/drop/shutdown/taskkill",
        "task_complete",
    ],
)
def test_vision_prompt_contains_required_constraints(required: str) -> None:
    assert required in _prompt()


def test_vision_prompt_forbids_markdown() -> None:
    prompt = _prompt()
    assert "sans Markdown" in prompt
    assert "sans texte avant ou après" in prompt
