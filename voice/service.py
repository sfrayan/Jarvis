"""Service de feedback vocal de Jarvis.

Iteration 5G-I : le service ecoute les phrases `AssistantUtterance` publiees
sur l'EventBus. Le speaker par defaut journalise la phrase comme sortie vocale
Jarvis ; Piper sera branche derriere la meme interface dans une iteration
dediee.
"""

from __future__ import annotations

from typing import Protocol

from config.schema import TTSConfig
from core.event_bus import EventBus, SubscriptionHandle
from observability.logger import get_logger
from voice.feedback import AssistantUtterance
from voice.piper import PiperTTSClient, PiperUnavailableError

log = get_logger(__name__)


class VoiceSpeakerLike(Protocol):
    """Contrat minimal d'une sortie vocale."""

    async def speak(self, utterance: AssistantUtterance) -> None:
        """Dit ou journalise une phrase assistant."""


class PiperClientLike(Protocol):
    """Contrat minimal du client Piper."""

    async def synthesize(self, *, text: str, voice: str) -> None:
        """Synthese une phrase via Piper."""


class LoggingVoiceSpeaker:
    """Speaker temporaire : log explicite en attendant Piper."""

    def __init__(self, config: TTSConfig) -> None:
        self._config = config

    async def speak(self, utterance: AssistantUtterance) -> None:
        """Journalise la phrase comme sortie vocale Jarvis."""
        log.info(
            "jarvis_voice_feedback",
            text=utterance.text,
            priority=utterance.priority,
            source=utterance.source,
            voice=self._config.voice,
            speed=self._config.speed,
            reason=utterance.reason,
        )


class PiperVoiceSpeaker:
    """Speaker Piper avec fallback optionnel vers un speaker de secours."""

    def __init__(
        self,
        config: TTSConfig,
        *,
        client: PiperClientLike | None = None,
        fallback: VoiceSpeakerLike | None = None,
    ) -> None:
        self._config = config
        self._client = client or PiperTTSClient(config)
        self._fallback = fallback

    async def speak(self, utterance: AssistantUtterance) -> None:
        """Envoie la phrase a Piper ou bascule vers le fallback."""
        try:
            await self._client.synthesize(text=utterance.text, voice=self._config.voice)
        except PiperUnavailableError as exc:
            if self._fallback is None:
                raise
            log.warning(
                "jarvis_voice_piper_unavailable",
                text=utterance.text,
                voice=self._config.voice,
                error=str(exc) or type(exc).__name__,
                fallback="log",
            )
            await self._fallback.speak(utterance)
            return

        log.info(
            "jarvis_voice_piper_sent",
            text=utterance.text,
            voice=self._config.voice,
            speed=self._config.speed,
            reason=utterance.reason,
        )


class VoiceFeedbackService:
    """Service reactif : AssistantUtterance -> speaker vocal."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        speaker: VoiceSpeakerLike,
    ) -> None:
        self._bus = event_bus
        self._speaker = speaker
        self._subscription: SubscriptionHandle | None = None

    @classmethod
    def create_default(
        cls,
        *,
        event_bus: EventBus,
        tts_config: TTSConfig,
    ) -> VoiceFeedbackService:
        """Factory utilisee par `main.py`."""
        fallback = LoggingVoiceSpeaker(tts_config)
        if tts_config.backend == "piper":
            return cls(
                event_bus=event_bus,
                speaker=PiperVoiceSpeaker(
                    tts_config,
                    fallback=fallback if tts_config.fallback_to_log else None,
                ),
            )
        return cls(
            event_bus=event_bus,
            speaker=fallback,
        )

    def start(self) -> None:
        """S'abonne aux phrases assistant. Idempotent."""
        if self._subscription is not None and self._subscription.active:
            return
        self._subscription = self._bus.subscribe(AssistantUtterance, self._on_utterance)
        log.info("voice_feedback_service_started")

    def stop(self) -> None:
        """Retire l'abonnement. Idempotent."""
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None
        log.info("voice_feedback_service_stopped")

    async def _on_utterance(self, event: AssistantUtterance) -> None:
        try:
            await self._speaker.speak(event)
        except Exception as exc:
            log.warning(
                "voice_feedback_failed",
                text=event.text,
                error=str(exc) or type(exc).__name__,
                error_type=type(exc).__name__,
            )
