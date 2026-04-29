"""Client Piper local via protocole Wyoming TCP.

Le protocole Wyoming transporte des evenements JSONL suivis d'un payload binaire
optionnel. Piper renvoie des chunks PCM que l'on joue via sounddevice.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, cast

from config.schema import TTSConfig


class PiperUnavailableError(RuntimeError):
    """Piper n'est pas joignable ou n'a pas produit d'audio exploitable."""


class AudioOutputLike(Protocol):
    """Contrat minimal d'une sortie audio PCM."""

    async def play_pcm(
        self,
        payload: bytes,
        *,
        rate: int,
        width: int,
        channels: int,
    ) -> None:
        """Joue un chunk PCM."""


@dataclass(frozen=True)
class _AudioFormat:
    rate: int
    width: int
    channels: int


@dataclass(frozen=True)
class _WyomingEvent:
    type: str
    data: Mapping[str, object]
    payload: bytes


class SoundDeviceAudioOutput:
    """Sortie audio host Windows via sounddevice."""

    async def play_pcm(
        self,
        payload: bytes,
        *,
        rate: int,
        width: int,
        channels: int,
    ) -> None:
        """Joue un chunk PCM sans bloquer la boucle asyncio."""
        await asyncio.to_thread(
            _play_pcm_sync,
            payload,
            rate=rate,
            width=width,
            channels=channels,
        )


class PiperTTSClient:
    """Client TTS Piper parlant Wyoming sur TCP."""

    def __init__(
        self,
        config: TTSConfig,
        *,
        audio_output: AudioOutputLike | None = None,
    ) -> None:
        self._host = config.host
        self._port = config.port
        self._timeout_s = config.timeout_s
        self._audio_output = audio_output or SoundDeviceAudioOutput()

    async def synthesize(self, *, text: str, voice: str) -> None:
        """Synthese puis joue `text` avec Piper."""
        try:
            await asyncio.wait_for(
                self._synthesize_once(text=text, voice=voice),
                timeout=self._timeout_s,
            )
        except PiperUnavailableError:
            raise
        except Exception as exc:
            raise PiperUnavailableError(str(exc) or type(exc).__name__) from exc

    async def _synthesize_once(self, *, text: str, voice: str) -> None:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port),
            timeout=self._timeout_s,
        )
        try:
            writer.write(
                _encode_event(
                    "synthesize",
                    {
                        "text": text,
                        "voice": {"name": voice},
                    },
                )
            )
            await asyncio.wait_for(writer.drain(), timeout=self._timeout_s)
            await self._consume_audio(reader)
        finally:
            writer.close()
            with suppress(Exception):
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)

    async def _consume_audio(self, reader: asyncio.StreamReader) -> None:
        audio_format: _AudioFormat | None = None
        audio_chunks = 0

        while True:
            event = await _read_event(reader, timeout_s=self._timeout_s)
            if event.type == "error":
                raise PiperUnavailableError(_error_message(event))
            if event.type == "audio-start":
                audio_format = _audio_format_from_event(event)
                continue
            if event.type == "audio-chunk":
                chunk_format = _audio_format_from_event(event, fallback=audio_format)
                if not event.payload:
                    continue
                await self._audio_output.play_pcm(
                    event.payload,
                    rate=chunk_format.rate,
                    width=chunk_format.width,
                    channels=chunk_format.channels,
                )
                audio_chunks += 1
                continue
            if event.type == "audio-stop":
                break

        if audio_chunks == 0:
            raise PiperUnavailableError("Piper n'a renvoye aucun chunk audio")


async def _read_event(reader: asyncio.StreamReader, *, timeout_s: float) -> _WyomingEvent:
    header_line = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
    if not header_line:
        raise PiperUnavailableError("Connexion Piper fermee sans reponse")

    header = _json_object(header_line.decode("utf-8"))
    event_type = header.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise PiperUnavailableError("Evenement Wyoming sans type valide")

    data = _object_map(header.get("data"))
    data_length = _int_field(header, "data_length")
    payload_length = _int_field(header, "payload_length")

    if data_length:
        raw_data = await asyncio.wait_for(reader.readexactly(data_length), timeout=timeout_s)
        data = {**data, **_json_object(raw_data.decode("utf-8"))}

    payload = b""
    if payload_length:
        payload = await asyncio.wait_for(reader.readexactly(payload_length), timeout=timeout_s)

    return _WyomingEvent(type=event_type, data=data, payload=payload)


def _encode_event(event_type: str, data: Mapping[str, object]) -> bytes:
    header = {
        "type": event_type,
        "data": dict(data),
    }
    return (json.dumps(header, ensure_ascii=False) + "\n").encode("utf-8")


def _audio_format_from_event(
    event: _WyomingEvent,
    *,
    fallback: _AudioFormat | None = None,
) -> _AudioFormat:
    rate = _int_field(event.data, "rate", default=fallback.rate if fallback else 0)
    width = _int_field(event.data, "width", default=fallback.width if fallback else 0)
    channels = _int_field(
        event.data,
        "channels",
        default=fallback.channels if fallback else 0,
    )
    if rate <= 0 or width <= 0 or channels <= 0:
        raise PiperUnavailableError("Format audio Piper incomplet")
    return _AudioFormat(rate=rate, width=width, channels=channels)


def _error_message(event: _WyomingEvent) -> str:
    message = event.data.get("message")
    if isinstance(message, str) and message:
        return message
    return "Erreur Piper inconnue"


def _json_object(raw: str) -> dict[str, object]:
    parsed = cast(object, json.loads(raw))
    if not isinstance(parsed, dict):
        raise PiperUnavailableError("JSON Wyoming invalide")
    return {str(key): value for key, value in parsed.items()}


def _object_map(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _int_field(
    data: Mapping[str, object],
    field: str,
    *,
    default: int = 0,
) -> int:
    value = data.get(field, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _play_pcm_sync(
    payload: bytes,
    *,
    rate: int,
    width: int,
    channels: int,
) -> None:
    import sounddevice as sd

    dtype = _sounddevice_dtype(width)
    with sd.RawOutputStream(samplerate=rate, channels=channels, dtype=dtype) as stream:
        stream.write(payload)


def _sounddevice_dtype(width: int) -> str:
    if width == 1:
        return "int8"
    if width == 2:
        return "int16"
    if width == 4:
        return "int32"
    raise PiperUnavailableError(f"Largeur PCM non supportee: {width}")
