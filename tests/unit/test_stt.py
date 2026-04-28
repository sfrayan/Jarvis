"""Tests unitaires du wrapper faster-whisper.

Le vrai modèle `medium` n'est pas chargé ici. On injecte un faux modèle pour
tester le contrat, le choix device/compute_type et la conversion audio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from config.schema import STTConfig
from ears.stt import (
    FasterWhisperTranscriber,
    _clean_transcription_text,
    _int16_to_float32,
    _resolve_compute_type,
    _resolve_device,
)

pytestmark = pytest.mark.unit


@dataclass
class _FakeSegment:
    text: str
    start: float = 0.0
    end: float = 1.0


@dataclass
class _FakeInfo:
    language: str = "fr"
    language_probability: float = 0.98
    duration: float = 1.0


class _FakeWhisperModel:
    def __init__(self) -> None:
        self.calls: list[npt.NDArray[np.float32]] = []
        self.last_language: str | None = None
        self.last_beam_size: int | None = None

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        *,
        language: str | None,
        beam_size: int,
        vad_filter: bool,
        condition_on_previous_text: bool,
    ) -> tuple[list[_FakeSegment], _FakeInfo]:
        self.calls.append(audio)
        self.last_language = language
        self.last_beam_size = beam_size
        assert vad_filter is False
        assert condition_on_previous_text is False
        return [_FakeSegment("Bonjour "), _FakeSegment("Jarvis")], _FakeInfo()


def _audio(samples: int = 16_000, fill: int = 0) -> npt.NDArray[np.int16]:
    return np.full((samples,), fill, dtype=np.int16)


class TestFasterWhisperTranscriber:
    @pytest.mark.asyncio
    async def test_transcribe_chunk_returns_transcription_event(self) -> None:
        model = _FakeWhisperModel()
        transcriber = FasterWhisperTranscriber(STTConfig(model="medium"), model=model)

        result = await transcriber.transcribe_chunk(_audio(fill=123))

        assert result.text == "Bonjour Jarvis"
        assert result.language == "fr"
        assert result.language_probability == pytest.approx(0.98)
        assert result.audio_duration_ms == pytest.approx(1000.0)
        assert result.inference_duration_ms >= 0
        assert len(model.calls) == 1

    @pytest.mark.asyncio
    async def test_passes_language_and_beam_size_to_model(self) -> None:
        model = _FakeWhisperModel()
        cfg = STTConfig(language="fr", beam_size=7)
        transcriber = FasterWhisperTranscriber(cfg, model=model)

        await transcriber.transcribe_chunk(_audio())

        assert model.last_language == "fr"
        assert model.last_beam_size == 7

    @pytest.mark.asyncio
    async def test_rejects_empty_audio(self) -> None:
        transcriber = FasterWhisperTranscriber(STTConfig(), model=_FakeWhisperModel())
        with pytest.raises(ValueError, match="vide"):
            await transcriber.transcribe_chunk(np.array([], dtype=np.int16))

    @pytest.mark.asyncio
    async def test_rejects_wrong_dtype(self) -> None:
        transcriber = FasterWhisperTranscriber(STTConfig(), model=_FakeWhisperModel())
        audio = np.zeros((16000,), dtype=np.float32)
        with pytest.raises(ValueError, match="int16"):
            await transcriber.transcribe_chunk(audio)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_rejects_non_mono_audio(self) -> None:
        transcriber = FasterWhisperTranscriber(STTConfig(), model=_FakeWhisperModel())
        audio = np.zeros((16000, 1), dtype=np.int16)
        with pytest.raises(ValueError, match="1D"):
            await transcriber.transcribe_chunk(audio)

    @pytest.mark.asyncio
    async def test_model_loaded_lazily_via_factory(self) -> None:
        fake_model = _FakeWhisperModel()
        calls: list[tuple[str, str, str]] = []

        def factory(model_name: str, device: str, compute_type: str) -> _FakeWhisperModel:
            calls.append((model_name, device, compute_type))
            return fake_model

        transcriber = FasterWhisperTranscriber(
            STTConfig(model="medium", device="cpu"),
            model_factory=factory,
        )

        assert calls == []
        await transcriber.transcribe_chunk(_audio())
        assert calls == [("medium", "cpu", "int8")]

        await transcriber.transcribe_chunk(_audio())
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_warm_up_loads_model_before_first_transcription(self) -> None:
        fake_model = _FakeWhisperModel()
        calls: list[tuple[str, str, str]] = []

        def factory(model_name: str, device: str, compute_type: str) -> _FakeWhisperModel:
            calls.append((model_name, device, compute_type))
            return fake_model

        transcriber = FasterWhisperTranscriber(
            STTConfig(model="medium", device="cpu"),
            model_factory=factory,
        )

        await transcriber.warm_up()
        assert calls == [("medium", "cpu", "int8")]

        await transcriber.transcribe_chunk(_audio())
        assert len(calls) == 1


class TestDeviceResolution:
    def test_cpu_compute_type(self) -> None:
        assert _resolve_device("cpu") == "cpu"
        assert _resolve_compute_type("cpu") == "int8"

    def test_cuda_compute_type(self) -> None:
        assert _resolve_device("cuda") == "cuda"
        assert _resolve_compute_type("cuda") == "int8_float16"

    def test_auto_uses_torch_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ears.stt._torch_cuda_available", lambda: True)
        assert _resolve_device("auto") == "cuda"

        monkeypatch.setattr("ears.stt._torch_cuda_available", lambda: False)
        assert _resolve_device("auto") == "cpu"

    def test_invalid_device_rejected(self) -> None:
        with pytest.raises(ValueError, match="Device STT non supporté"):
            _resolve_device("gpu")


class TestAudioConversion:
    def test_int16_to_float32_bounds(self) -> None:
        audio = np.array([-32768, 0, 32767], dtype=np.int16)
        waveform = _int16_to_float32(audio)
        assert waveform.dtype == np.float32
        assert float(waveform[0]) == pytest.approx(-1.0)
        assert float(waveform[1]) == pytest.approx(0.0)
        assert float(waveform[2]) == pytest.approx(32767 / 32768)


class TestTranscriptionCleanup:
    def test_filters_repeated_video_outro_hallucination(self) -> None:
        raw = "J'espère que ça vous a plu. J'espère que ça vous a plu. J'espère que ça vous a plu."
        assert _clean_transcription_text(raw) == ""

    def test_keeps_normal_repeated_user_command(self) -> None:
        raw = "ouvre Discord. ouvre Chrome."
        assert _clean_transcription_text(raw) == raw
