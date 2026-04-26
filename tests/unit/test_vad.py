"""Tests unitaires du wrapper Silero VAD.

Le vrai modèle Silero n'est pas chargé ici : on injecte un faux modèle pour
garder les tests rapides, déterministes, et indépendants de ONNX Runtime.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
import torch

from config.schema import AudioConfig
from ears.vad import SileroVAD, _extract_probability, _int16_to_float_tensor

pytestmark = pytest.mark.unit


class _FakeModel:
    def __init__(self, probability: float = 0.75) -> None:
        self.probability = probability
        self.calls: list[tuple[torch.Tensor, int]] = []
        self.reset_count = 0

    def __call__(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self.calls.append((audio, sample_rate))
        return torch.tensor([[self.probability]], dtype=torch.float32)

    def reset_states(self) -> None:
        self.reset_count += 1


def _chunk(size: int = 512, fill: int = 0) -> npt.NDArray[np.int16]:
    return np.full((size,), fill, dtype=np.int16)


class TestSileroVAD:
    def test_predict_probability_calls_model_with_normalized_tensor(self) -> None:
        model = _FakeModel(probability=0.73)
        vad = SileroVAD(AudioConfig(sample_rate=16000), model=model)

        chunk = _chunk(fill=16_384)
        probability = vad.predict_probability(chunk)

        assert probability == pytest.approx(0.73)
        assert len(model.calls) == 1
        audio, sample_rate = model.calls[0]
        assert sample_rate == 16_000
        assert audio.dtype == torch.float32
        assert audio.shape == (512,)
        assert float(audio[0]) == pytest.approx(0.5)

    def test_is_speech_uses_configured_threshold(self) -> None:
        vad = SileroVAD(AudioConfig(vad_threshold=0.6), model=_FakeModel())
        assert vad.is_speech(0.6) is True
        assert vad.is_speech(0.59) is False

    def test_reset_delegates_to_model(self) -> None:
        model = _FakeModel()
        vad = SileroVAD(AudioConfig(), model=model)
        vad.reset()
        assert model.reset_count == 1

    def test_rejects_wrong_dtype(self) -> None:
        vad = SileroVAD(AudioConfig(), model=_FakeModel())
        chunk = np.zeros((512,), dtype=np.float32)
        with pytest.raises(ValueError, match="int16"):
            vad.predict_probability(chunk)  # type: ignore[arg-type]

    def test_rejects_non_mono_chunk(self) -> None:
        vad = SileroVAD(AudioConfig(), model=_FakeModel())
        chunk = np.zeros((512, 1), dtype=np.int16)
        with pytest.raises(ValueError, match="1D"):
            vad.predict_probability(chunk)

    def test_rejects_wrong_chunk_size(self) -> None:
        vad = SileroVAD(AudioConfig(sample_rate=16000), model=_FakeModel())
        with pytest.raises(ValueError, match="512"):
            vad.predict_probability(_chunk(size=256))

    def test_8000_hz_expects_256_samples(self) -> None:
        model = _FakeModel()
        vad = SileroVAD(AudioConfig(sample_rate=8000), model=model)
        vad.predict_probability(_chunk(size=256))
        assert vad.expected_samples == 256
        assert model.calls[0][1] == 8_000

    def test_unsupported_sample_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="8000 Hz ou 16000 Hz"):
            SileroVAD(AudioConfig(sample_rate=48000), model=_FakeModel())

    def test_loads_onnx_model_by_default(self) -> None:
        fake_model = _FakeModel()
        with patch("ears.vad.load_silero_vad", return_value=fake_model) as loader:
            vad = SileroVAD(AudioConfig())
        loader.assert_called_once_with(onnx=True)
        assert vad.expected_samples == 512

    def test_can_disable_onnx_loader(self) -> None:
        fake_model = _FakeModel()
        with patch("ears.vad.load_silero_vad", return_value=fake_model) as loader:
            SileroVAD(AudioConfig(), use_onnx=False)
        loader.assert_called_once_with(onnx=False)


class TestHelpers:
    def test_int16_conversion_bounds(self) -> None:
        chunk = np.array([-32768, 0, 32767], dtype=np.int16)
        tensor = _int16_to_float_tensor(chunk)
        assert tensor.dtype == torch.float32
        assert float(tensor[0]) == pytest.approx(-1.0)
        assert float(tensor[1]) == pytest.approx(0.0)
        assert float(tensor[2]) == pytest.approx(32767 / 32768)

    @pytest.mark.parametrize(
        "raw",
        [
            torch.tensor([[0.42]], dtype=torch.float32),
            np.array([[0.42]], dtype=np.float32),
            0.42,
        ],
    )
    def test_extract_probability_supported_outputs(self, raw: object) -> None:
        assert _extract_probability(raw) == pytest.approx(0.42)

    def test_extract_probability_rejects_unknown_type(self) -> None:
        with pytest.raises(TypeError, match="Sortie VAD non supportée"):
            _extract_probability(object())


class TestFixture:
    def test_piper_fixture_exists(self) -> None:
        fixture = Path("tests/fixtures/audio/jarvis_vad_fr.wav")
        assert fixture.exists()
        assert fixture.stat().st_size > 10_000
