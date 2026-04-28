"""Contrats JSON du module vision GUI.

Le modèle vision local (Qwen2.5-VL via Ollama) devra répondre avec un JSON
validé par ces schémas avant qu'une action ne puisse être envisagée.

Règles importantes :

- `confidence < 0.6` force `actions=[]` et `requires_human=true`.
- Une action destructive détectée force `requires_human=true`.
- Les coordonnées `x`, `y` sont exprimées dans l'espace de l'image envoyée au
  modèle vision. La conversion vers l'espace écran arrivera dans l'Itération 5
  (`hands/screenshot.py` + `hands/actuators.py`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ActionType = Literal[
    "mouse_move",
    "left_click",
    "right_click",
    "double_click",
    "scroll",
    "type_text",
    "key_combo",
    "wait",
]

CONFIDENCE_THRESHOLD = 0.6

_COORD_ACTIONS = frozenset({"mouse_move", "left_click", "right_click", "double_click"})
_DESTRUCTIVE_PATTERNS = (
    "rm ",
    "rm-",
    "remove-item",
    "del ",
    "delete",
    "format ",
    "drop table",
    "drop database",
    "shutdown",
    "reboot",
    "restart-computer",
    "taskkill",
)


class VisionAction(BaseModel):
    """Action GUI atomique proposée par le modèle vision.

    Les champs sont optionnels au niveau modèle, puis validés selon `type`.
    Cela garde un contrat JSON simple côté LLM tout en étant strict côté code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ActionType
    x: int | None = Field(default=None, ge=0)
    y: int | None = Field(default=None, ge=0)
    text: str | None = Field(default=None, min_length=1)
    keys: list[str] | None = Field(default=None, min_length=1)
    amount: int | None = None
    duration_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_shape(self) -> VisionAction:
        if self.type in _COORD_ACTIONS and (self.x is None or self.y is None):
            raise ValueError(f"L'action {self.type} exige x et y")
        if self.type == "scroll" and self.amount is None:
            raise ValueError("L'action scroll exige amount")
        if self.type == "type_text" and self.text is None:
            raise ValueError("L'action type_text exige text")
        if self.type == "key_combo" and not self.keys:
            raise ValueError("L'action key_combo exige keys")
        if self.type == "wait" and self.duration_ms is None:
            raise ValueError("L'action wait exige duration_ms")
        return self

    def contains_destructive_intent(self) -> bool:
        """Retourne vrai si l'action semble destructive."""
        haystack_parts: list[str] = [self.type]
        if self.text is not None:
            haystack_parts.append(self.text)
        if self.keys is not None:
            haystack_parts.extend(self.keys)
        haystack = " ".join(haystack_parts).casefold()
        return any(pattern in haystack for pattern in _DESTRUCTIVE_PATTERNS)


class VisionDecision(BaseModel):
    """Décision complète retournée par le modèle vision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    thought: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    speech: str = Field(..., min_length=1)
    actions: list[VisionAction] = Field(default_factory=list)
    external_tools: list[str] = Field(default_factory=list)
    requires_human: bool
    task_complete: bool

    @model_validator(mode="after")
    def _enforce_safety_rules(self) -> VisionDecision:
        if self.confidence < CONFIDENCE_THRESHOLD:
            if self.actions:
                raise ValueError("confidence < 0.6 impose actions=[]")
            if not self.requires_human:
                raise ValueError("confidence < 0.6 impose requires_human=true")

        if self.has_destructive_action() and not self.requires_human:
            raise ValueError("Une action destructive impose requires_human=true")

        return self

    def has_destructive_action(self) -> bool:
        """Indique si au moins une action porte un signal destructif."""
        return any(action.contains_destructive_intent() for action in self.actions)


def human_required_decision(*, thought: str, speech: str) -> VisionDecision:
    """Construit une décision sûre quand le modèle est inutilisable."""
    return VisionDecision(
        thought=thought,
        confidence=0.0,
        speech=speech,
        actions=[],
        external_tools=[],
        requires_human=True,
        task_complete=False,
    )
