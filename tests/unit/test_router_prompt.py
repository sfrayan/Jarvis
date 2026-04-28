"""Tests de garde-fou pour le prompt system du routeur."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROMPT_PATH = Path("brain/prompts/router_system.txt")


def test_router_prompt_exists_and_is_non_empty() -> None:
    assert PROMPT_PATH.exists()
    assert PROMPT_PATH.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize(
    "required_fragment",
    [
        '"chat"',
        '"gui"',
        '"unknown"',
        "JSON valide",
        "confidence",
        "reason",
        "domain",
        "home_assistant",
        "vision",
        "memory",
        "routine",
        "ouvre Chrome",
        "Javis ouvre Discord",
        "opéra",
        "VS Code",
    ],
)
def test_router_prompt_contains_required_constraints(required_fragment: str) -> None:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    assert required_fragment in prompt
