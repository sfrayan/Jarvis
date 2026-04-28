"""Orchestration du pipeline audio d'entrée.

`EarsService` relie :

    AudioStream -> SileroVAD -> FasterWhisperTranscriber -> EventBus/StateMachine

En Itération 3, c'est le premier service réellement "vivant" côté micro. Il
reste volontairement local et testable :

- `run()` gère le cycle de vie du stream et s'arrête via `stop()`.
- `process_chunk()` permet de tester la logique VAD/STT sans ouvrir le micro.
- Les dépendances (`stream`, `vad`, `transcriber`) sont injectables.

Transitions produites :

- `IDLE` ou `SPEAKING` + parole -> `LISTENING`
- `LISTENING` + silence prolongé -> `TRANSCRIBING`
- transcription terminée -> `IDLE` (en Itération 4, ce sera `ROUTING`)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Protocol, Self, cast

import numpy as np
import numpy.typing as npt

from config.schema import AudioConfig, STTConfig
from core.event_bus import EventBus
from core.state_machine import State, StateMachine
from ears.audio_stream import AudioStream
from ears.events import SilenceDetected, SpeechDetected, Transcription
from ears.stt import FasterWhisperTranscriber
from ears.vad import SileroVAD
from observability.logger import get_logger

log = get_logger(__name__)

AudioChunk = npt.NDArray[np.int16]


class AudioChunkStream(Protocol):
    """Contrat minimal d'un stream audio async."""

    async def __aenter__(self) -> Self:
        """Démarre le stream."""

    async def __aexit__(self, *exc: object) -> None:
        """Arrête le stream."""

    def __aiter__(self) -> AsyncIterator[AudioChunk]:
        """Retourne l'itérateur async de chunks audio."""

    def flush_pending(self) -> int:
        """Vide les chunks en attente après une transcription longue."""


class VADLike(Protocol):
    """Contrat minimal du VAD utilisé par le service."""

    def predict_probability(self, chunk: AudioChunk) -> float:
        """Retourne une probabilité de parole."""

    def is_speech(self, probability: float) -> bool:
        """Indique si la probabilité franchit le seuil."""

    def reset(self) -> None:
        """Réinitialise les états internes du VAD."""


class TranscriberLike(Protocol):
    """Contrat minimal du transcriber STT."""

    async def warm_up(self) -> None:
        """Charge les ressources lourdes avant ouverture du micro."""

    async def transcribe_chunk(self, audio: AudioChunk) -> Transcription:
        """Transcrit un segment audio complet."""


class EarsService:
    """Service audio : VAD streaming + STT par segment."""

    def __init__(
        self,
        *,
        audio_config: AudioConfig,
        stt_config: STTConfig,
        event_bus: EventBus,
        state_machine: StateMachine,
        stream: AudioChunkStream | None = None,
        vad: VADLike | None = None,
        transcriber: TranscriberLike | None = None,
    ) -> None:
        self._audio_config = audio_config
        self._bus = event_bus
        self._sm = state_machine
        self._stream = stream if stream is not None else AudioStream(audio_config)
        self._vad = vad if vad is not None else SileroVAD(audio_config)
        self._transcriber = (
            transcriber if transcriber is not None else FasterWhisperTranscriber(stt_config)
        )

        self._stop_event = asyncio.Event()
        self._recording = False
        self._speech_buffer: list[AudioChunk] = []
        self._silence_duration_ms = 0.0

    def stop(self) -> None:
        """Demande l'arrêt propre du service."""
        self._stop_event.set()

    async def run(self) -> None:
        """Consomme le stream audio jusqu'à `stop()` ou fin d'iterator."""
        if self._stop_event.is_set():
            return

        log.info("ears_service_started")
        try:
            await self._warm_up_transcriber()
            if self._stop_event.is_set():
                return
            async with self._stream as stream:
                iterator = stream.__aiter__()
                while not self._stop_event.is_set():
                    chunk = await self._next_chunk_or_stop(iterator)
                    if chunk is None:
                        break
                    await self.process_chunk(chunk)
        finally:
            log.info("ears_service_stopped")

    async def process_chunk(self, chunk: AudioChunk) -> None:
        """Traite un chunk audio unique.

        Cette méthode contient la logique métier et reste indépendante du vrai
        micro, ce qui rend les tests unitaires rapides.
        """
        probability = self._vad.predict_probability(chunk)
        speech = self._vad.is_speech(probability)

        if speech:
            await self._handle_speech_chunk(chunk, probability)
            return

        if self._recording:
            await self._handle_silence_chunk(chunk)

    async def _next_chunk_or_stop(
        self,
        iterator: AsyncIterator[AudioChunk],
    ) -> AudioChunk | None:
        chunk_task = asyncio.create_task(
            self._next_chunk(iterator),
            name="ears_next_chunk",
        )
        stop_task = asyncio.create_task(
            self._stop_event.wait(),
            name="ears_stop_wait",
        )
        done, pending = await asyncio.wait(
            {chunk_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        if stop_task in done:
            return None

        with suppress(StopAsyncIteration):
            return chunk_task.result()
        return None

    async def _next_chunk(self, iterator: AsyncIterator[AudioChunk]) -> AudioChunk:
        """Wrapper coroutine autour de `anext()` pour satisfaire mypy strict."""
        return await anext(iterator)

    async def _handle_speech_chunk(self, chunk: AudioChunk, probability: float) -> None:
        if not self._recording:
            self._recording = True
            self._speech_buffer = []
            self._silence_duration_ms = 0.0
            await self._bus.publish(SpeechDetected(timestamp=time.time(), probability=probability))
            await self._transition_if_allowed(
                State.LISTENING,
                reason="vad_speech",
            )

        self._speech_buffer.append(chunk.copy())
        self._silence_duration_ms = 0.0

    async def _handle_silence_chunk(self, chunk: AudioChunk) -> None:
        self._speech_buffer.append(chunk.copy())
        self._silence_duration_ms += _chunk_duration_ms(
            chunk,
            sample_rate=self._audio_config.sample_rate,
        )

        if self._silence_duration_ms < self._audio_config.silence_timeout_ms:
            return

        silence_duration_ms = int(round(self._silence_duration_ms))
        await self._bus.publish(
            SilenceDetected(
                timestamp=time.time(),
                silence_duration_ms=silence_duration_ms,
            )
        )
        await self._transition_if_allowed(
            State.TRANSCRIBING,
            reason="vad_silence",
        )
        await self._transcribe_current_buffer()

    async def _transcribe_current_buffer(self) -> None:
        if not self._speech_buffer:
            self._reset_segment()
            return

        audio = cast(AudioChunk, np.concatenate(self._speech_buffer))
        self._reset_segment()

        transcription = await self._transcriber.transcribe_chunk(audio)
        self._flush_stream_backlog()
        await self._bus.publish(transcription)
        await self._transition_if_allowed(State.IDLE, reason="awaiting_router")
        self._vad.reset()

    def _reset_segment(self) -> None:
        self._recording = False
        self._speech_buffer = []
        self._silence_duration_ms = 0.0

    async def _transition_if_allowed(self, target: State, *, reason: str) -> None:
        current = self._sm.state
        if target in StateMachine.allowed_from(current):
            await self._sm.transition(target, reason=reason)
        else:
            log.debug(
                "ears_transition_skipped",
                from_state=current.value,
                to_state=target.value,
                reason=reason,
            )

    def _flush_stream_backlog(self) -> None:
        flushed = self._stream.flush_pending()
        if flushed:
            log.info("ears_audio_backlog_flushed", chunks=flushed)

    async def _warm_up_transcriber(self) -> None:
        log.info("ears_stt_warmup_started")
        await self._transcriber.warm_up()
        log.info("ears_stt_warmup_finished")


def _chunk_duration_ms(chunk: AudioChunk, *, sample_rate: int) -> float:
    """Durée d'un chunk en millisecondes."""
    return len(chunk) / sample_rate * 1000.0
