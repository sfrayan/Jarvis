"""Wrapper `faster-whisper` pour la transcription de segments audio.

Le transcriber reçoit un segment audio complet, mono `int16` à 16 kHz, puis
retourne un événement `Transcription` prêt à publier sur l'event bus.

Le modèle est chargé paresseusement :

- En production : premier appel à `transcribe_chunk()` → chargement Whisper.
- En tests : injection d'un faux modèle via `model=...`, aucun téléchargement.

Choix matériel (machine cible RTX 2060 6 Go) :

- `device=auto` choisit CUDA si `torch.cuda.is_available()`, sinon CPU.
- `compute_type=int8_float16` sur CUDA, `int8` sur CPU.
- En Itération 4, quand la vision locale arrive, on pourra forcer STT sur CPU
  via `config/local.yaml` pour libérer la VRAM.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Iterable
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from config.schema import STTConfig
from ears.events import Transcription
from observability.logger import get_logger

log = get_logger(__name__)

_INT16_SCALE: float = 32768.0
_WHISPER_SAMPLE_RATE: int = 16_000


class WhisperSegment(Protocol):
    """Contrat minimal des segments retournés par faster-whisper."""

    text: str
    start: float
    end: float


class WhisperInfo(Protocol):
    """Contrat minimal des métadonnées retournées par faster-whisper."""

    language: str
    language_probability: float
    duration: float


class WhisperModelLike(Protocol):
    """Contrat minimal du modèle utilisé par `FasterWhisperTranscriber`."""

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        *,
        language: str | None,
        beam_size: int,
        vad_filter: bool,
        condition_on_previous_text: bool,
    ) -> tuple[Iterable[WhisperSegment], WhisperInfo]:
        """Transcrit un waveform float32 mono à 16 kHz."""


ModelFactory = Callable[[str, str, str], WhisperModelLike]


class FasterWhisperTranscriber:
    """Transcription async d'un segment audio complet."""

    def __init__(
        self,
        config: STTConfig,
        *,
        model: WhisperModelLike | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._config = config
        self._model = model
        self._model_factory = model_factory or _default_model_factory
        self._resolved_device = _resolve_device(config.device)
        self._compute_type = _resolve_compute_type(self._resolved_device)

    @property
    def resolved_device(self) -> str:
        """Device effectif (`cuda` ou `cpu`) après résolution de `auto`."""
        return self._resolved_device

    @property
    def compute_type(self) -> str:
        """Quantization CTranslate2 choisie pour le device effectif."""
        return self._compute_type

    async def warm_up(self) -> None:
        """Charge le modèle Whisper sans transcrire d'audio.

        Appelé avant l'ouverture du micro pour éviter que la première phrase
        utilisateur s'accumule puis soit jetée pendant le chargement CUDA.
        """
        start = time.perf_counter()
        await asyncio.to_thread(self._ensure_model)
        log.info(
            "faster_whisper_warmed",
            model=self._config.model,
            device=self._resolved_device,
            compute_type=self._compute_type,
            duration_ms=round((time.perf_counter() - start) * 1000.0, 1),
        )

    async def transcribe_chunk(self, audio: npt.NDArray[np.int16]) -> Transcription:
        """Transcrit un segment audio `int16` complet.

        L'inférence est lancée dans un thread via `asyncio.to_thread`, car
        `faster-whisper` est synchrone.
        """
        self._validate_audio(audio)
        waveform = _int16_to_float32(audio)
        audio_duration_ms = len(audio) / _WHISPER_SAMPLE_RATE * 1000.0

        start = time.perf_counter()
        text, info = await asyncio.to_thread(self._transcribe_sync, waveform)
        cleaned_text = _clean_transcription_text(text)
        inference_duration_ms = (time.perf_counter() - start) * 1000.0

        transcription = Transcription(
            timestamp=time.time(),
            text=cleaned_text,
            language=info.language,
            language_probability=info.language_probability,
            inference_duration_ms=inference_duration_ms,
            audio_duration_ms=audio_duration_ms,
        )
        log.info(
            "stt_transcribed",
            text=transcription.text,
            language=transcription.language,
            language_probability=transcription.language_probability,
            inference_duration_ms=round(transcription.inference_duration_ms, 1),
            audio_duration_ms=round(transcription.audio_duration_ms, 1),
        )
        return transcription

    def _transcribe_sync(self, waveform: npt.NDArray[np.float32]) -> tuple[str, WhisperInfo]:
        model = self._ensure_model()
        segments, info = model.transcribe(
            waveform,
            language=self._config.language,
            beam_size=self._config.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = "".join(segment.text for segment in segments)
        return text, info

    def _ensure_model(self) -> WhisperModelLike:
        if self._model is None:
            self._model = self._model_factory(
                self._config.model,
                self._resolved_device,
                self._compute_type,
            )
            log.info(
                "faster_whisper_loaded",
                model=self._config.model,
                device=self._resolved_device,
                compute_type=self._compute_type,
            )
        return self._model

    @staticmethod
    def _validate_audio(audio: npt.NDArray[np.int16]) -> None:
        if audio.dtype != np.int16:
            raise ValueError(f"Audio STT attendu en int16, reçu {audio.dtype}")
        if audio.ndim != 1:
            raise ValueError(f"Audio STT mono 1D attendu, shape reçue : {audio.shape}")
        if audio.size == 0:
            raise ValueError("Audio STT vide")


def _default_model_factory(model_name: str, device: str, compute_type: str) -> WhisperModelLike:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        num_workers=1,
    )
    return cast(WhisperModelLike, model)


def _resolve_device(config_device: str) -> str:
    if config_device == "auto":
        return "cuda" if _torch_cuda_available() else "cpu"
    if config_device in {"cuda", "cpu"}:
        return config_device
    raise ValueError(f"Device STT non supporté : {config_device}")


def _resolve_compute_type(device: str) -> str:
    return "int8_float16" if device == "cuda" else "int8"


def _clean_transcription_text(text: str) -> str:
    """Nettoie quelques hallucinations STT connues avant publication."""
    cleaned = text.strip()
    if _looks_like_whisper_hallucination(cleaned):
        log.debug("stt_hallucination_filtered", text=cleaned)
        return ""
    return cleaned


def _looks_like_whisper_hallucination(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.casefold().strip())
    known_repeated_fragments = (
        "j'espère que ça vous a plu",
        "merci d'avoir regardé",
        "n'oubliez pas de vous abonner",
    )
    if any(normalized.count(fragment) >= 2 for fragment in known_repeated_fragments):
        return True

    sentences = [
        sentence.strip(" .!?")
        for sentence in re.split(r"[.!?]+", normalized)
        if sentence.strip(" .!?")
    ]
    return len(sentences) >= 3 and len(set(sentences)) == 1


def _torch_cuda_available() -> bool:
    """Détecte CUDA sans importer Torch au chargement du module.

    Sur Windows, `import torch` peut charger beaucoup de DLL CUDA et prendre
    plusieurs secondes. On garde donc cet import local au moment où on en a
    réellement besoin.
    """
    try:
        import torch
    except Exception as exc:
        log.warning(
            "torch_cuda_detection_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return False
    return bool(torch.cuda.is_available())


def _int16_to_float32(audio: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
    """Convertit PCM int16 mono vers waveform float32 dans [-1.0, 1.0]."""
    return audio.astype(np.float32) / _INT16_SCALE
