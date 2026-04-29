"""Tests unitaires du service Voice 5G-I."""

from __future__ import annotations

import pytest

from config.schema import TTSConfig
from core.event_bus import EventBus
from voice.feedback import AssistantUtterance
from voice.piper import PiperUnavailableError
from voice.service import PiperVoiceSpeaker, VoiceFeedbackService

pytestmark = pytest.mark.unit


class _FakeSpeaker:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[AssistantUtterance] = []

    async def speak(self, utterance: AssistantUtterance) -> None:
        self.calls.append(utterance)
        if self.fail:
            raise RuntimeError("tts indisponible")


class _FakePiperClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    async def synthesize(self, *, text: str, voice: str) -> None:
        self.calls.append((text, voice))
        if self.fail:
            raise PiperUnavailableError("piper indisponible")


def _utterance(text: str = "J'ouvre Antigravity.") -> AssistantUtterance:
    return AssistantUtterance(
        timestamp=1.0,
        text=text,
        source="hands",
        priority="info",
        reason="test",
    )


class TestVoiceFeedbackService:
    def test_tts_config_defaults_to_safe_log_backend(self) -> None:
        config = TTSConfig()

        assert config.backend == "log"
        assert config.host == "127.0.0.1"
        assert config.port == 10200
        assert config.fallback_to_log is True

    @pytest.mark.asyncio
    async def test_publishes_utterance_to_speaker(self) -> None:
        bus = EventBus()
        speaker = _FakeSpeaker()
        service = VoiceFeedbackService(event_bus=bus, speaker=speaker)
        service.start()

        await bus.publish(_utterance())

        assert [call.text for call in speaker.calls] == ["J'ouvre Antigravity."]

    def test_start_is_idempotent(self) -> None:
        bus = EventBus()
        service = VoiceFeedbackService(event_bus=bus, speaker=_FakeSpeaker())

        service.start()
        service.start()

        assert bus.subscriber_count(AssistantUtterance) == 1

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self) -> None:
        bus = EventBus()
        speaker = _FakeSpeaker()
        service = VoiceFeedbackService(event_bus=bus, speaker=speaker)
        service.start()
        service.stop()

        await bus.publish(_utterance())

        assert bus.subscriber_count(AssistantUtterance) == 0
        assert speaker.calls == []

    @pytest.mark.asyncio
    async def test_speaker_failure_is_swallowed(self) -> None:
        bus = EventBus()
        speaker = _FakeSpeaker(fail=True)
        service = VoiceFeedbackService(event_bus=bus, speaker=speaker)
        service.start()

        await bus.publish(_utterance("Action bloquee."))

        assert [call.text for call in speaker.calls] == ["Action bloquee."]
        assert bus.subscriber_count(AssistantUtterance) == 1

    @pytest.mark.asyncio
    async def test_piper_speaker_calls_client(self) -> None:
        client = _FakePiperClient()
        speaker = PiperVoiceSpeaker(
            TTSConfig(backend="piper", voice="fr_FR-siwis-medium"),
            client=client,
        )

        await speaker.speak(_utterance("J'ouvre Antigravity."))

        assert client.calls == [("J'ouvre Antigravity.", "fr_FR-siwis-medium")]

    @pytest.mark.asyncio
    async def test_piper_speaker_falls_back_when_piper_is_unavailable(self) -> None:
        client = _FakePiperClient(fail=True)
        fallback = _FakeSpeaker()
        speaker = PiperVoiceSpeaker(
            TTSConfig(backend="piper"),
            client=client,
            fallback=fallback,
        )

        await speaker.speak(_utterance("Piper indisponible."))

        assert client.calls == [("Piper indisponible.", "fr_FR-siwis-medium")]
        assert [call.text for call in fallback.calls] == ["Piper indisponible."]

    def test_create_default_accepts_piper_backend(self) -> None:
        bus = EventBus()
        service = VoiceFeedbackService.create_default(
            event_bus=bus,
            tts_config=TTSConfig(backend="piper"),
        )

        service.start()

        assert bus.subscriber_count(AssistantUtterance) == 1
