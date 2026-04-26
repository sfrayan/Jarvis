"""Wrapper Silero VAD pour la détection de parole en streaming.

Ce module expose une API très petite :

    vad = SileroVAD(AudioConfig())
    probability = vad.predict_probability(chunk)
    if vad.is_speech(probability):
        ...

`chunk` doit être un tableau NumPy `int16` mono, issu de `AudioStream` :

- 512 samples à 16 kHz
- 256 samples à 8 kHz

Le modèle est chargé en ONNX CPU par défaut via `silero_vad.load_silero_vad`.
C'est volontaire : sur la machine cible, le GPU 6 Go doit rester disponible
pour faster-whisper puis la vision locale.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import torch
from silero_vad import load_silero_vad

from config.schema import AudioConfig
from observability.logger import get_logger

log = get_logger(__name__)

_INT16_SCALE: float = 32768.0


class SileroModel(Protocol):
    """Contrat minimal du modèle Silero utilisé par ce wrapper."""

    def __call__(self, audio: torch.Tensor, sample_rate: int) -> object:
        """Retourne une probabilité de parole sous forme tensor-like."""

    def reset_states(self) -> None:
        """Réinitialise les états internes du modèle streaming."""


class SileroVAD:
    """Détecteur de parole Silero, adapté aux chunks `AudioStream`."""

    def __init__(
        self,
        config: AudioConfig,
        *,
        model: SileroModel | None = None,
        use_onnx: bool = True,
    ) -> None:
        self._config = config
        self._sample_rate = config.sample_rate
        self._threshold = config.vad_threshold
        self._expected_samples = _expected_samples(self._sample_rate)
        self._model = model if model is not None else self._load_model(use_onnx=use_onnx)

    @property
    def threshold(self) -> float:
        """Seuil au-dessus duquel un chunk est considéré comme parole."""
        return self._threshold

    @property
    def sample_rate(self) -> int:
        """Sample rate utilisé par le modèle."""
        return self._sample_rate

    @property
    def expected_samples(self) -> int:
        """Taille attendue des chunks pour le sample rate courant."""
        return self._expected_samples

    def predict_probability(self, chunk: npt.NDArray[np.int16]) -> float:
        """Retourne la probabilité de parole pour un chunk audio.

        Args:
            chunk: tableau mono `int16` de taille `expected_samples`.

        Raises:
            ValueError: si la taille ou le dtype du chunk ne correspond pas au
                contrat streaming.
        """
        self._validate_chunk(chunk)
        tensor = _int16_to_float_tensor(chunk)

        with torch.no_grad():
            raw_probability = self._model(tensor, self._sample_rate)

        probability = _extract_probability(raw_probability)
        log.debug(
            "vad_probability",
            probability=probability,
            threshold=self._threshold,
        )
        return probability

    def is_speech(self, probability: float) -> bool:
        """Indique si `probability` franchit le seuil configuré."""
        return probability >= self._threshold

    def reset(self) -> None:
        """Réinitialise les états internes du modèle streaming."""
        self._model.reset_states()

    def _validate_chunk(self, chunk: npt.NDArray[np.int16]) -> None:
        if chunk.dtype != np.int16:
            raise ValueError(f"Chunk audio attendu en int16, reçu {chunk.dtype}")
        if chunk.ndim != 1:
            raise ValueError(f"Chunk mono 1D attendu, shape reçue : {chunk.shape}")
        if chunk.shape[0] != self._expected_samples:
            raise ValueError(
                f"Chunk de {self._expected_samples} samples attendu à "
                f"{self._sample_rate} Hz, reçu {chunk.shape[0]}"
            )

    @staticmethod
    def _load_model(*, use_onnx: bool) -> SileroModel:
        model = load_silero_vad(onnx=use_onnx)
        log.info("silero_vad_loaded", onnx=use_onnx)
        return cast(SileroModel, model)


def _expected_samples(sample_rate: int) -> int:
    if sample_rate == 16_000:
        return 512
    if sample_rate == 8_000:
        return 256
    raise ValueError("Silero VAD supporte uniquement 8000 Hz ou 16000 Hz")


def _int16_to_float_tensor(chunk: npt.NDArray[np.int16]) -> torch.Tensor:
    """Convertit PCM int16 mono vers tensor float32 dans [-1.0, 1.0]."""
    audio = chunk.astype(np.float32) / _INT16_SCALE
    return torch.from_numpy(audio)


def _extract_probability(raw_probability: object) -> float:
    """Convertit la sortie Silero en float Python."""
    if isinstance(raw_probability, torch.Tensor):
        return float(raw_probability.item())
    if isinstance(raw_probability, np.ndarray):
        return float(raw_probability.item())
    if isinstance(raw_probability, int | float):
        return float(raw_probability)
    if hasattr(raw_probability, "item"):
        return float(raw_probability.item())
    raise TypeError(f"Sortie VAD non supportée : {type(raw_probability)!r}")
