"""Tests unitaires des contrats JSON vision."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from brain.vision_contracts import VisionAction, VisionDecision, human_required_decision

pytestmark = pytest.mark.unit


def _decision(**overrides: object) -> VisionDecision:
    data: dict[str, object] = {
        "thought": "Je vois Chrome ouvert.",
        "confidence": 0.92,
        "speech": "Je clique.",
        "actions": [VisionAction(type="left_click", x=100, y=200)],
        "external_tools": [],
        "requires_human": False,
        "task_complete": False,
    }
    data.update(overrides)
    return VisionDecision.model_validate(data)


class TestVisionAction:
    def test_left_click_requires_coordinates(self) -> None:
        with pytest.raises(ValidationError, match="exige x et y"):
            VisionAction(type="left_click")

    def test_left_click_accepts_coordinates(self) -> None:
        action = VisionAction(type="left_click", x=10, y=20)
        assert action.x == 10
        assert action.y == 20

    def test_scroll_requires_amount(self) -> None:
        with pytest.raises(ValidationError, match="scroll exige amount"):
            VisionAction(type="scroll")

    def test_type_text_requires_text(self) -> None:
        with pytest.raises(ValidationError, match="type_text exige text"):
            VisionAction(type="type_text")

    def test_key_combo_requires_keys(self) -> None:
        with pytest.raises(ValidationError, match="key_combo exige keys"):
            VisionAction(type="key_combo")

    def test_wait_requires_duration(self) -> None:
        with pytest.raises(ValidationError, match="wait exige duration_ms"):
            VisionAction(type="wait")

    def test_destructive_text_detected(self) -> None:
        action = VisionAction(type="type_text", text="rm -rf C:\\Users")
        assert action.contains_destructive_intent() is True


class TestVisionDecision:
    def test_valid_decision(self) -> None:
        decision = _decision()
        assert decision.confidence == pytest.approx(0.92)
        assert decision.requires_human is False

    def test_low_confidence_rejects_actions(self) -> None:
        with pytest.raises(ValidationError, match="actions=\\[\\]"):
            _decision(confidence=0.4, requires_human=True)

    def test_low_confidence_requires_human(self) -> None:
        with pytest.raises(ValidationError, match="requires_human=true"):
            _decision(confidence=0.4, actions=[], requires_human=False)

    def test_low_confidence_safe_shape_is_valid(self) -> None:
        decision = _decision(confidence=0.4, actions=[], requires_human=True)
        assert decision.actions == []
        assert decision.requires_human is True

    def test_destructive_action_requires_human(self) -> None:
        action = VisionAction(type="type_text", text="drop table users")
        with pytest.raises(ValidationError, match="destructive"):
            _decision(actions=[action], requires_human=False)

    def test_destructive_action_with_human_required_is_valid(self) -> None:
        action = VisionAction(type="type_text", text="shutdown /s")
        decision = _decision(actions=[action], requires_human=True)
        assert decision.has_destructive_action() is True

    def test_human_required_factory(self) -> None:
        decision = human_required_decision(
            thought="JSON invalide.",
            speech="Je n'ai pas compris l'écran, peux-tu confirmer ?",
        )
        assert decision.confidence == 0.0
        assert decision.actions == []
        assert decision.requires_human is True
