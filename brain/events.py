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
