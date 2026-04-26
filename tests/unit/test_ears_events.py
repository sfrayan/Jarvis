"""Tests unitaires des événements `ears/`.

Vérifie : validation Pydantic, immutabilité (frozen), intégration avec
l'event bus (publish → handler reçoit bien l'instance typée).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.event_bus import EventBus
from ears.events import SilenceDetected, SpeechDetected, Transcription

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------
# SpeechDetected
# ---------------------------------------------------------------------
class TestSpeechDetected:
    def test_construct_valid(self) -> None:
        event = SpeechDetected(timestamp=1.0, probability=0.8)
        assert event.timestamp == 1.0
        assert event.probability == 0.8

    @pytest.mark.parametrize("bad_prob", [-0.01, 1.01, 2.0, -1.0])
    def test_probability_bounds_rejected(self, bad_prob: float) -> None:
        with pytest.raises(ValidationError):
            SpeechDetected(timestamp=1.0, probability=bad_prob)

    def test_is_frozen(self) -> None:
        event = SpeechDetected(timestamp=1.0, probability=0.8)
        with pytest.raises(ValidationError):
            event.probability = 0.5  # type: ignore[misc]

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            SpeechDetected(timestamp=1.0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------
# SilenceDetected
# ---------------------------------------------------------------------
class TestSilenceDetected:
    def test_construct_valid(self) -> None:
        event = SilenceDetected(timestamp=2.0, silence_duration_ms=800)
        assert event.silence_duration_ms == 800

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SilenceDetected(timestamp=2.0, silence_duration_ms=-1)

    def test_zero_duration_accepted(self) -> None:
        # Cas limite : autorisé (peut arriver si le chunk courant matche
        # pile le seuil avec 0 ms de silence mesurable).
        event = SilenceDetected(timestamp=2.0, silence_duration_ms=0)
        assert event.silence_duration_ms == 0


# ---------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------
class TestTranscription:
    def _mk(self, **overrides: object) -> Transcription:
        defaults: dict[str, object] = {
            "timestamp": 3.0,
            "text": "Bonjour Jarvis",
            "language": "fr",
            "language_probability": 0.99,
            "inference_duration_ms": 120.0,
            "audio_duration_ms": 1500.0,
        }
        defaults.update(overrides)
        return Transcription(**defaults)  # type: ignore[arg-type]

    def test_construct_valid(self) -> None:
        event = self._mk()
        assert event.text == "Bonjour Jarvis"
        assert event.language == "fr"

    def test_language_code_length_bounds(self) -> None:
        # "f" trop court
        with pytest.raises(ValidationError):
            self._mk(language="f")
        # "french" trop long (>5)
        with pytest.raises(ValidationError):
            self._mk(language="french")
        # "fr-FR" OK (5 chars)
        event = self._mk(language="fr-FR")
        assert event.language == "fr-FR"

    def test_negative_durations_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._mk(inference_duration_ms=-1.0)
        with pytest.raises(ValidationError):
            self._mk(audio_duration_ms=-1.0)

    def test_language_probability_bounds(self) -> None:
        with pytest.raises(ValidationError):
            self._mk(language_probability=1.5)

    def test_is_frozen(self) -> None:
        event = self._mk()
        with pytest.raises(ValidationError):
            event.text = "hacked"  # type: ignore[misc]


# ---------------------------------------------------------------------
# Intégration event bus : publish → handler reçoit bien l'instance typée
# ---------------------------------------------------------------------
class TestBusIntegration:
    @pytest.mark.asyncio
    async def test_speech_detected_routed_correctly(self) -> None:
        bus = EventBus()
        received: list[SpeechDetected] = []

        async def handler(event: SpeechDetected) -> None:
            received.append(event)

        bus.subscribe(SpeechDetected, handler)
        await bus.publish(SpeechDetected(timestamp=1.0, probability=0.75))

        assert len(received) == 1
        assert received[0].probability == 0.75

    @pytest.mark.asyncio
    async def test_transcription_not_received_by_speech_handler(self) -> None:
        """Isolation : un handler abonné à SpeechDetected ne reçoit pas les
        Transcription, même si les deux circulent."""
        bus = EventBus()
        speech_received: list[SpeechDetected] = []
        trans_received: list[Transcription] = []

        async def speech_handler(event: SpeechDetected) -> None:
            speech_received.append(event)

        async def trans_handler(event: Transcription) -> None:
            trans_received.append(event)

        bus.subscribe(SpeechDetected, speech_handler)
        bus.subscribe(Transcription, trans_handler)

        await bus.publish(SpeechDetected(timestamp=1.0, probability=0.9))
        await bus.publish(
            Transcription(
                timestamp=2.0,
                text="test",
                language="fr",
                language_probability=1.0,
                inference_duration_ms=10.0,
                audio_duration_ms=100.0,
            )
        )

        assert len(speech_received) == 1
        assert len(trans_received) == 1
