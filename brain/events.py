"""Evenements publies par le subsystem `brain/`.

En Iteration 4, le cerveau ne produit qu'une decision de routage d'intention :

- `chat` : l'utilisateur attend une reponse conversationnelle.
- `gui` : l'utilisateur demande une action sur l'interface.
- `unknown` : demande ambigue ou confidence trop faible.
- `domain` : sous-domaine interne conservateur, sans execution d'action.

Ces evenements sont immutables pour eviter qu'un handler ne modifie une decision
partagee apres publication sur l'event bus.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from brain.task_session import TaskSession, TaskSessionKind

IntentType = Literal["chat", "gui", "unknown"]
IntentDomain = Literal[
    "general",
    "system",
    "apps",
    "folders",
    "media",
    "home_assistant",
    "vision",
    "memory",
    "web_search",
    "google_workspace",
    "routine",
    "unknown",
]


class IntentRouted(BaseModel):
    """Decision de routage produite depuis une transcription utilisateur."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float = Field(
        ...,
        ge=0.0,
        description="time.time() UNIX au moment de la decision",
    )
    original_text: str = Field(
        ...,
        min_length=1,
        description="Texte brut issu du STT",
    )
    normalized_text: str = Field(
        ...,
        min_length=1,
        description="Texte apres normalisation legere des noms propres",
    )
    intent: IntentType = Field(
        ...,
        description="chat | gui | unknown",
    )
    domain: IntentDomain = Field(
        default="general",
        description="Domaine fonctionnel interne pour preparer le dispatch futur",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confiance du routeur dans sa decision",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Explication courte utile aux logs et tests",
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Modele ayant produit la decision (ex qwen3:latest)",
    )


class ClarificationQuestion(BaseModel):
    """Question posee quand une demande est incomplete."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float = Field(..., ge=0.0)
    session_id: str = Field(..., min_length=1)
    kind: TaskSessionKind
    question: str = Field(..., min_length=1)
    missing_slots: tuple[str, ...] = ()
    reason: str = Field(..., min_length=1)


class AssistantPlan(BaseModel):
    """Plan de travail propose avant toute action sensible ou ambigue."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float = Field(..., ge=0.0)
    session_id: str = Field(..., min_length=1)
    kind: TaskSessionKind
    summary: str = Field(..., min_length=1)
    steps: tuple[str, ...] = Field(..., min_length=1)
    requires_confirmation: bool = True
    reason: str = Field(..., min_length=1)


class AssistantDraft(BaseModel):
    """Brouillon de travail genere en RAM, sans ecriture fichier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float = Field(..., ge=0.0)
    session_id: str = Field(..., min_length=1)
    kind: TaskSessionKind
    title: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)
    sections: tuple[str, ...] = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    next_steps: tuple[str, ...] = ()
    reason: str = Field(..., min_length=1)


class TaskSessionStateChanged(BaseModel):
    """Snapshot publie quand la session de tache evolue."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float = Field(..., ge=0.0)
    session: TaskSession
    reason: str = Field(..., min_length=1)
