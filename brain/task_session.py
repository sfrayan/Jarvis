"""Modeles de session de tache pour le dialogue interactif.

Une `TaskSession` garde le contexte temporaire en RAM entre deux phrases
utilisateur. Elle ne decide aucune action GUI : elle decrit seulement ce que
Jarvis a compris, ce qui manque encore, et l'etat courant de la tache.

En 5R, le champ `routine_kind` trace le type de routine associee (work, code,
research, focus, homework) pour permettre l'execution sequentielle des
suggestions d'actions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TaskSessionKind = Literal[
    "homework",
    "coding_project",
    "work_setup",
    "web_video",
    "sensitive_system",
    "general",
]
TaskSessionStatus = Literal["waiting_for_user", "ready", "cancelled", "completed"]


class TaskSlot(BaseModel):
    """Information collectee pendant une session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class TaskSession(BaseModel):
    """Contexte temporaire d'une tache utilisateur."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(..., min_length=1)
    kind: TaskSessionKind
    status: TaskSessionStatus
    original_request: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    slots: tuple[TaskSlot, ...] = ()
    missing_slots: tuple[str, ...] = ()
    plan_steps: tuple[str, ...] = ()
    created_at: float = Field(..., ge=0.0)
    updated_at: float = Field(..., ge=0.0)
    last_question: str | None = Field(default=None, min_length=1)
    last_user_reply: str | None = Field(default=None, min_length=1)
    routine_kind: str | None = Field(
        default=None,
        min_length=1,
        description="Type de routine associee (work, code, research, focus, homework)",
    )

    def slot_value(self, name: str) -> str | None:
        """Retourne la valeur d'un slot par nom, si elle existe."""
        for slot in self.slots:
            if slot.name == name:
                return slot.value
        return None

    def with_update(
        self,
        *,
        now: float,
        status: TaskSessionStatus | None = None,
        slots: tuple[TaskSlot, ...] | None = None,
        missing_slots: tuple[str, ...] | None = None,
        plan_steps: tuple[str, ...] | None = None,
        last_question: str | None = None,
        last_user_reply: str | None = None,
    ) -> TaskSession:
        """Retourne une copie mise a jour sans muter la session existante."""
        return self.model_copy(
            update={
                "status": status if status is not None else self.status,
                "slots": slots if slots is not None else self.slots,
                "missing_slots": (
                    missing_slots if missing_slots is not None else self.missing_slots
                ),
                "plan_steps": plan_steps if plan_steps is not None else self.plan_steps,
                "updated_at": now,
                "last_question": last_question,
                "last_user_reply": last_user_reply,
            }
        )


def make_session_id(timestamp: float) -> str:
    """Construit un identifiant stable pour une session en RAM."""
    return f"task-{int(timestamp * 1000)}"


def merge_slots(
    existing: tuple[TaskSlot, ...],
    updates: dict[str, str],
) -> tuple[TaskSlot, ...]:
    """Fusionne des slots en remplacant les valeurs deja connues."""
    merged = {slot.name: slot.value for slot in existing}
    for name, value in updates.items():
        cleaned = value.strip()
        if cleaned:
            merged[name] = cleaned
    return tuple(TaskSlot(name=name, value=value) for name, value in sorted(merged.items()))
