"""Tests unitaires d'`EarsService`.

Tout est injecté : faux stream, faux VAD, faux transcriber. Aucun micro, aucun
modèle Silero, aucun Whisper réel.
"""

from __future__ import annotations

import asyncio
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import pytest

from config.schema import AudioConfig, STTConfig
from core.event_bus import EventBus
from core.state_machine import State, StateMachine
from ears.events import SilenceDetected, SpeechDetected, Transcription
from ears.service import EarsService, _chunk_duration_ms

pytestmark = pytest.mark.unit

T = TypeVar("T")


def _chunk(fill: int = 0, size: int = 512) -> npt.NDArray[np.int16]:
    return np.full((size,), fill, dtype=np.int16)


class _FakeVAD:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = probabilities
        self.reset_count = 0

    def predict_probability(self, chunk: npt.NDArray[np.int16]) -> float:
        if not self._probabilities:
            return 0.0
        return self._probabilities.pop(0)

    def is_speech(self, probability: float) -> bool:
        return probability >= 0.5

    def reset(self) -> None:
        self.reset_count += 1


class _FakeTranscriber:
    def __init__(self, text: str = "Bonjour Jarvis") -> None:
        self.text = text
        self.calls: list[npt.NDArray[np.int16]] = []
        self.warm_up_count = 0
        self.events: list[str] | None = None

    async def warm_up(self) -> None:
        self.warm_up_count += 1
        if self.events is not None:
            self.events.append("warm_up")

    async def transcribe_chunk(self, audio: npt.NDArray[np.int16]) -> Transcription:
        self.calls.append(audio)
        return Transcription(
            timestamp=1.0,
            text=self.text,
            language="fr",
            language_probability=0.99,
            inference_duration_ms=10.0,
            audio_duration_ms=len(audio) / 16_000 * 1000.0,
        )


class _FakeStream:
    def __init__(self, chunks: list[npt.NDArray[np.int16]]) -> None:
        self._chunks = list(chunks)
        self.entered = False
        self.exited = False
        self.flush_count = 0
        self.pending_to_flush = 0
        self.events: list[str] | None = None

    async def __aenter__(self) -> _FakeStream:
        if self.events is not None:
            self.events.append("stream_enter")
        self.entered = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.exited = True

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> npt.NDArray[np.int16]:
        await asyncio.sleep(0)
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    def flush_pending(self) -> int:
        self.flush_count += 1
        flushed = self.pending_to_flush
        self.pending_to_flush = 0
        return flushed


def _mk_service(
    *,
    probabilities: list[float],
    stream: _FakeStream | None = None,
    initial_state: State = State.IDLE,
    silence_timeout_ms: int = 100,
) -> tuple[
    EarsService,
    EventBus,
    StateMachine,
    _FakeVAD,
    _FakeTranscriber,
    _FakeStream,
]:
    bus = EventBus()
    sm = StateMachine(bus, initial=initial_state)
    vad = _FakeVAD(probabilities)
    transcriber = _FakeTranscriber()
    fake_stream = stream if stream is not None else _FakeStream([])
    service = EarsService(
        audio_config=AudioConfig(silence_timeout_ms=silence_timeout_ms),
        stt_config=STTConfig(),
        event_bus=bus,
        state_machine=sm,
        stream=fake_stream,
        vad=vad,
        transcriber=transcriber,
    )
    return service, bus, sm, vad, transcriber, fake_stream


class TestProcessChunk:
    @pytest.mark.asyncio
    async def test_idle_speech_publishes_event_and_enters_listening(self) -> None:
        service, bus, sm, _, transcriber, _ = _mk_service(probabilities=[0.9])
        received: list[SpeechDetected] = []

        async def handler(event: SpeechDetected) -> None:
            received.append(event)

        bus.subscribe(SpeechDetected, handler)
        await service.process_chunk(_chunk(fill=1))

        assert sm.state is State.LISTENING
        assert len(received) == 1
        assert received[0].probability == pytest.approx(0.9)
        assert transcriber.calls == []

    @pytest.mark.asyncio
    async def test_idle_silence_is_ignored(self) -> None:
        service, _, sm, _, transcriber, _ = _mk_service(probabilities=[0.1])
        await service.process_chunk(_chunk())
        assert sm.state is State.IDLE
        assert transcriber.calls == []

    @pytest.mark.asyncio
    async def test_silence_threshold_transcribes_and_returns_idle(self) -> None:
        # 1 chunk speech + 4 chunks silence.
        # 4 * 512 / 16000 * 1000 = 128 ms >= silence_timeout_ms=100
        service, bus, sm, vad, transcriber, _ = _mk_service(
            probabilities=[0.9, 0.1, 0.1, 0.1, 0.1],
            silence_timeout_ms=100,
        )
        speech_events: list[SpeechDetected] = []
        silence_events: list[SilenceDetected] = []
        transcriptions: list[Transcription] = []

        bus.subscribe(SpeechDetected, lambda event: _append(speech_events, event))
        bus.subscribe(SilenceDetected, lambda event: _append(silence_events, event))
        bus.subscribe(Transcription, lambda event: _append(transcriptions, event))

        for chunk in [_chunk(1), _chunk(2), _chunk(3), _chunk(4), _chunk(5)]:
            await service.process_chunk(chunk)

        assert sm.state is State.IDLE
        assert len(speech_events) == 1
        assert len(silence_events) == 1
        assert silence_events[0].silence_duration_ms == 128
        assert len(transcriptions) == 1
        assert transcriptions[0].text == "Bonjour Jarvis"
        assert len(transcriber.calls) == 1
        assert len(transcriber.calls[0]) == 512 * 5
        assert vad.reset_count == 1

    @pytest.mark.asyncio
    async def test_flushes_stream_backlog_after_transcription(self) -> None:
        stream = _FakeStream([])
        stream.pending_to_flush = 42
        service, _, _, _, transcriber, fake_stream = _mk_service(
            probabilities=[0.9, 0.1, 0.1, 0.1, 0.1],
            stream=stream,
            silence_timeout_ms=100,
        )

        for chunk in [_chunk(1), _chunk(2), _chunk(3), _chunk(4), _chunk(5)]:
            await service.process_chunk(chunk)

        assert len(transcriber.calls) == 1
        assert fake_stream.flush_count == 1
        assert fake_stream.pending_to_flush == 0

    @pytest.mark.asyncio
    async def test_silence_below_threshold_does_not_transcribe(self) -> None:
        service, _, sm, _, transcriber, _ = _mk_service(
            probabilities=[0.9, 0.1, 0.1],
            silence_timeout_ms=100,
        )

        for chunk in [_chunk(1), _chunk(2), _chunk(3)]:
            await service.process_chunk(chunk)

        assert sm.state is State.LISTENING
        assert transcriber.calls == []

    @pytest.mark.asyncio
    async def test_speech_while_speaking_uses_barge_in_transition(self) -> None:
        service, _, sm, _, _, _ = _mk_service(
            probabilities=[0.9],
            initial_state=State.SPEAKING,
        )
        await service.process_chunk(_chunk())
        assert sm.state is State.LISTENING


class TestRun:
    @pytest.mark.asyncio
    async def test_run_processes_stream_until_exhausted(self) -> None:
        stream = _FakeStream([_chunk(1), _chunk(2), _chunk(3), _chunk(4), _chunk(5)])
        service, _, sm, _, transcriber, fake_stream = _mk_service(
            probabilities=[0.9, 0.1, 0.1, 0.1, 0.1],
            stream=stream,
        )

        await service.run()

        assert fake_stream.entered is True
        assert fake_stream.exited is True
        assert transcriber.warm_up_count == 1
        assert sm.state is State.IDLE
        assert len(transcriber.calls) == 1

    @pytest.mark.asyncio
    async def test_run_warms_transcriber_before_entering_stream(self) -> None:
        stream = _FakeStream([])
        service, _, _, _, transcriber, fake_stream = _mk_service(
            probabilities=[],
            stream=stream,
        )
        events: list[str] = []
        transcriber.events = events
        fake_stream.events = events

        await service.run()

        assert events == ["warm_up", "stream_enter"]

    @pytest.mark.asyncio
    async def test_stop_before_run_does_not_enter_stream(self) -> None:
        stream = _FakeStream([_chunk()])
        service, _, _, _, _, fake_stream = _mk_service(
            probabilities=[0.9],
            stream=stream,
        )

        service.stop()
        await service.run()

        assert fake_stream.entered is False
        assert fake_stream.exited is False


class TestHelpers:
    def test_chunk_duration_ms(self) -> None:
        assert _chunk_duration_ms(_chunk(size=512), sample_rate=16_000) == pytest.approx(32.0)


async def _append(items: list[T], item: T) -> None:
    items.append(item)
