"""Événements publiés par le subsystem `ears/` sur le bus.

Trois types, tous frozen (immutable) pour garantir que les handlers ne
mutent pas un événement partagé :

- `SpeechDetected` : VAD front montant (début de voix).
- `SilenceDetected` : silence prolongé détecté après un segment de parole.
- `Transcription` : résultat faster-whisper pour un segment complet.

Les consommateurs (router en Itération 4, observabilité, tests end-to-end)
s'abonnent via `event_bus.subscribe(<Type>, handler)`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SpeechDetected(BaseModel):
    """Le VAD a détecté un début de parole (probabilité au-dessus du seuil).

    Émis une seule fois par segment (sur le front montant), pas à chaque chunk.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: float = Field(
        ...,
        description="time.time() UNIX à l'instant du front montant",
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilité Silero retournée pour le chunk déclencheur",
    )


class SilenceDetected(BaseModel):
    """Fin d'un segment de parole : silence continu ≥ `silence_timeout_ms`."""

    model_config = ConfigDict(frozen=True)

    timestamp: float = Field(
        ...,
        description="time.time() UNIX au moment où le seuil de silence est atteint",
    )
    silence_duration_ms: int = Field(
        ...,
        ge=0,
        description="Durée de silence mesurée (doit être ≥ config.audio.silence_timeout_ms)",
    )


class Transcription(BaseModel):
    """Résultat faster-whisper d'un segment audio complet."""

    model_config = ConfigDict(frozen=True)

    timestamp: float = Field(
        ...,
        description="time.time() UNIX à la fin de l'inférence",
    )
    text: str = Field(
        ...,
        description="Texte transcrit, espaces en bord strip()-és",
    )
    language: str = Field(
        ...,
        min_length=2,
        max_length=5,
        description="Code langue ISO détecté (ex 'fr'). Forcé si config.stt.language défini",
    )
    language_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confiance faster-whisper dans la détection de langue",
    )
    inference_duration_ms: float = Field(
        ...,
        ge=0.0,
        description="Temps machine consommé par whisper",
    )
    audio_duration_ms: float = Field(
        ...,
        ge=0.0,
        description="Durée du segment audio transcrit (wall time de parole)",
    )
