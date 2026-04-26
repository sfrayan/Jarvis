"""Tests unitaires d'`AudioStream`.

Stratégie : on mocke `sounddevice.InputStream` pour ne pas ouvrir le vrai
micro pendant les tests. On invoque le callback directement avec des
numpy arrays fabriqués, puis on vérifie que l'async iterator fournit les
chunks dans l'ordre.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from config.schema import AudioConfig
from ears.audio_stream import VAD_BLOCK_SIZE, AudioStream

pytestmark = pytest.mark.unit

_PATCH_TARGET = "ears.audio_stream.sd.InputStream"


def _fake_chunk(fill: int = 0, size: int = VAD_BLOCK_SIZE) -> npt.NDArray[np.int16]:
    """Numpy array 2D (size, 1) int16 — même forme que sounddevice fournit."""
    return np.full((size, 1), fill, dtype=np.int16)


# ---------------------------------------------------------------------
# Cycle de vie
# ---------------------------------------------------------------------
class TestLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager_starts_and_stops(self) -> None:
        with patch(_PATCH_TARGET) as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            stream = AudioStream(AudioConfig())
            async with stream:
                mock_instance.start.assert_called_once()
                mock_instance.stop.assert_not_called()

            mock_instance.stop.assert_called_once()
            mock_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_iterator_before_enter_raises(self) -> None:
        stream = AudioStream(AudioConfig())
        with pytest.raises(RuntimeError, match="async context manager"):
            await stream.__anext__()

    @pytest.mark.asyncio
    async def test_input_stream_configured_from_audio_config(self) -> None:
        with patch(_PATCH_TARGET) as mock_cls:
            mock_cls.return_value = MagicMock()
            cfg = AudioConfig(sample_rate=16000, device_index=None)
            async with AudioStream(cfg):
                pass
            # Vérifie les arguments passés à InputStream
            kwargs = mock_cls.call_args.kwargs
            assert kwargs["samplerate"] == 16000
            assert kwargs["channels"] == 1
            assert kwargs["dtype"] == "int16"
            assert kwargs["blocksize"] == VAD_BLOCK_SIZE
            assert kwargs["device"] is None


# ---------------------------------------------------------------------
# Callback → queue → iterator
# ---------------------------------------------------------------------
class TestCallbackFlow:
    @pytest.mark.asyncio
    async def test_single_chunk_flows_to_iterator(self) -> None:
        with patch(_PATCH_TARGET) as mock_cls:
            mock_cls.return_value = MagicMock()
            stream = AudioStream(AudioConfig())

            async with stream:
                stream._callback(_fake_chunk(fill=42), VAD_BLOCK_SIZE, None, None)
                await asyncio.sleep(0)  # laisse call_soon_threadsafe tourner

                chunk = await asyncio.wait_for(stream.__anext__(), timeout=0.5)
                assert chunk.shape == (VAD_BLOCK_SIZE,)  # flatten 2D → 1D
                assert chunk.dtype == np.int16
                assert np.all(chunk == 42)

    @pytest.mark.asyncio
    async def test_multiple_chunks_preserve_order(self) -> None:
        with patch(_PATCH_TARGET) as mock_cls:
            mock_cls.return_value = MagicMock()
            stream = AudioStream(AudioConfig())

            async with stream:
                for i in range(5):
                    stream._callback(_fake_chunk(fill=i), VAD_BLOCK_SIZE, None, None)
                    await asyncio.sleep(0)

                collected: list[int] = []
                for _ in range(5):
                    c = await asyncio.wait_for(stream.__anext__(), timeout=0.5)
                    collected.append(int(c[0]))

                assert collected == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_chunk_is_copied_not_referenced(self) -> None:
        """Le chunk enfilé ne doit pas partager sa mémoire avec le buffer
        PortAudio (sinon mutation = corruption)."""
        with patch(_PATCH_TARGET) as mock_cls:
            mock_cls.return_value = MagicMock()
            stream = AudioStream(AudioConfig())

            async with stream:
                original = _fake_chunk(fill=5)
                stream._callback(original, VAD_BLOCK_SIZE, None, None)
                await asyncio.sleep(0)

                # Simule la réutilisation du buffer par PortAudio : on
                # modifie `original` après l'appel.
                original.fill(99)

                received = await asyncio.wait_for(stream.__anext__(), timeout=0.5)
                # Le chunk récupéré doit rester à 5, pas 99.
                assert np.all(received == 5)


# ---------------------------------------------------------------------
# Backpressure / drop
# ---------------------------------------------------------------------
class TestBackpressure:
    @pytest.mark.asyncio
    async def test_queue_full_increments_dropped_counter(self) -> None:
        with patch(_PATCH_TARGET) as mock_cls:
            mock_cls.return_value = MagicMock()
            stream = AudioStream(AudioConfig(), queue_maxsize=3)

            async with stream:
                for i in range(10):
                    stream._callback(_fake_chunk(fill=i), VAD_BLOCK_SIZE, None, None)
                    await asyncio.sleep(0)

                # 10 chunks tentés, 3 acceptés → 7 droppés
                assert stream.dropped_chunks == 7

    @pytest.mark.asyncio
    async def test_dropped_counter_zero_when_consumer_keeps_up(self) -> None:
        with patch(_PATCH_TARGET) as mock_cls:
            mock_cls.return_value = MagicMock()
            stream = AudioStream(AudioConfig(), queue_maxsize=5)

            async with stream:
                for i in range(3):
                    stream._callback(_fake_chunk(fill=i), VAD_BLOCK_SIZE, None, None)
                    await asyncio.sleep(0)

                # Consomme immédiatement
                for _ in range(3):
                    await asyncio.wait_for(stream.__anext__(), timeout=0.5)

                assert stream.dropped_chunks == 0


# ---------------------------------------------------------------------
# Flush backlog
# ---------------------------------------------------------------------
class TestFlushPending:
    @pytest.mark.asyncio
    async def test_flush_pending_removes_queued_chunks(self) -> None:
        with patch(_PATCH_TARGET) as mock_cls:
            mock_cls.return_value = MagicMock()
            stream = AudioStream(AudioConfig())

            async with stream:
                for i in range(3):
                    stream._callback(_fake_chunk(fill=i), VAD_BLOCK_SIZE, None, None)
                    await asyncio.sleep(0)

                assert stream.flush_pending() == 3

                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(stream.__anext__(), timeout=0.05)

    def test_flush_pending_before_enter_is_noop(self) -> None:
        stream = AudioStream(AudioConfig())
        assert stream.flush_pending() == 0


# ---------------------------------------------------------------------
# Status non fatal
# ---------------------------------------------------------------------
class TestCallbackStatus:
    @pytest.mark.asyncio
    async def test_nonempty_status_does_not_raise(self) -> None:
        """Un status sounddevice non vide (ex overflow) est loggué mais
        n'interrompt pas le flux."""
        with patch(_PATCH_TARGET) as mock_cls:
            mock_cls.return_value = MagicMock()
            stream = AudioStream(AudioConfig())

            async with stream:
                stream._callback(
                    _fake_chunk(fill=1),
                    VAD_BLOCK_SIZE,
                    None,
                    "input overflow",  # status non vide
                )
                await asyncio.sleep(0)

                chunk = await asyncio.wait_for(stream.__anext__(), timeout=0.5)
                assert np.all(chunk == 1)
