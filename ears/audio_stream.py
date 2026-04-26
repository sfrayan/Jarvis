"""Wrapper async de `sounddevice.InputStream`.

Pattern producteur/consommateur :

- Le callback PortAudio (thread RT) reçoit des chunks fixes de 512 samples
  @ 16 kHz (taille recommandée par Silero VAD) et les push dans une
  `asyncio.Queue` via `call_soon_threadsafe`.
- Le consommateur (EarsService) itère avec `async for chunk in stream:`
  dans la boucle asyncio.

Caractéristiques :

- **Backpressure borné** : si la queue est pleine (consommateur trop lent),
  les nouveaux chunks sont droppés et comptés dans `dropped_chunks` (loggué
  périodiquement).
- **Format figé** : int16 mono 16 kHz. Tout changement impose un ré-échantillonnage
  chez les consommateurs (Silero et whisper attendent ce format).
- **Cycle de vie** : async context manager — `__aenter__` démarre le stream,
  `__aexit__` le stoppe et le ferme proprement.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from config.schema import AudioConfig
from observability.logger import get_logger

log = get_logger(__name__)

# Taille de chunk recommandée par Silero VAD (~32 ms @ 16 kHz).
VAD_BLOCK_SIZE: int = 512


class AudioStream:
    """Producteur async de chunks audio int16 depuis le micro."""

    def __init__(
        self,
        config: AudioConfig,
        *,
        queue_maxsize: int = 100,
    ) -> None:
        self._config = config
        self._queue_maxsize = queue_maxsize

        self._queue: asyncio.Queue[npt.NDArray[np.int16]] | None = None
        self._stream: Any = None  # sd.InputStream (non typé — stubs manquants)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dropped_chunks: int = 0

    @property
    def dropped_chunks(self) -> int:
        """Nombre de chunks droppés depuis le démarrage (queue pleine)."""
        return self._dropped_chunks

    # ------------------------------------------------------------------
    # Cycle de vie (async context manager)
    # ------------------------------------------------------------------
    async def __aenter__(self) -> AudioStream:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._queue_maxsize)

        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=VAD_BLOCK_SIZE,
            callback=self._callback,
            device=self._config.device_index,
        )
        self._stream.start()

        log.info(
            "audio_stream_started",
            sample_rate=self._config.sample_rate,
            block_size=VAD_BLOCK_SIZE,
            device_index=self._config.device_index,
            queue_maxsize=self._queue_maxsize,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log.info(
            "audio_stream_stopped",
            dropped_chunks=self._dropped_chunks,
        )

    # ------------------------------------------------------------------
    # Async iterator
    # ------------------------------------------------------------------
    def __aiter__(self) -> AudioStream:
        return self

    async def __anext__(self) -> npt.NDArray[np.int16]:
        if self._queue is None:
            raise RuntimeError("AudioStream doit être utilisé comme async context manager")
        chunk: npt.NDArray[np.int16] = await self._queue.get()
        return chunk

    def flush_pending(self) -> int:
        """Vide les chunks audio en attente dans la queue.

        À appeler après une transcription longue : pendant que Whisper tourne,
        le callback micro continue d'empiler de l'audio. Ces chunks sont déjà
        anciens au moment où la transcription revient, donc on les jette pour
        repartir sur de l'audio frais.

        Returns:
            Nombre de chunks retirés de la queue.
        """
        if self._queue is None:
            return 0

        flushed = 0
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            flushed += 1

        if flushed:
            log.info("audio_stream_flushed", chunks=flushed)
        return flushed

    # ------------------------------------------------------------------
    # Interne — callback PortAudio
    # ------------------------------------------------------------------
    def _callback(
        self,
        indata: npt.NDArray[np.int16],
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Callback appelé par PortAudio dans un thread RT.

        Doit être court (< 10 ms typiquement). On copie le chunk (indata
        pointe vers un buffer réutilisé) et on le délègue à l'event loop
        via `call_soon_threadsafe`.
        """
        if status:
            # Overflow / underflow / input error. Non fatal mais notable.
            log.warning("audio_stream_callback_status", status=str(status))

        if self._queue is None or self._loop is None:
            return

        # Copie + flatten : shape (blocksize, channels=1) → (blocksize,)
        chunk = indata.copy().reshape(-1)

        # Event loop déjà fermée (shutdown en cours) → ignore.
        with suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._enqueue, chunk)

    def _enqueue(self, chunk: npt.NDArray[np.int16]) -> None:
        """Appelé sur l'event loop. Insère ou drop si la queue est saturée."""
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            self._dropped_chunks += 1
            # Log périodique (1 / 100) pour ne pas saturer stderr.
            if self._dropped_chunks == 1 or self._dropped_chunks % 100 == 0:
                log.warning(
                    "audio_stream_queue_full_drop",
                    total_dropped=self._dropped_chunks,
                )
