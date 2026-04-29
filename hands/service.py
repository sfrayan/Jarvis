"""Pipeline GUI dry-run : screenshot -> vision -> hands.

Iteration 5C : ce service branche les briques 5A/5B sans execution reelle. Une
intention GUI locale declenche une capture d'ecran, une analyse vision locale,
puis un rapport `HandsExecutionReport` publie sur l'EventBus.
"""

from __future__ import annotations

from typing import Protocol

from brain.events import IntentRouted
from brain.vision_contracts import VisionDecision, human_required_decision
from brain.vision_local import LocalVisionClient
from config.schema import JarvisConfig
from core.event_bus import EventBus, SubscriptionHandle
from core.state_machine import State, StateMachine
from hands.executor import DryRunHandsExecutor, HandsExecutionReport
from hands.local_actions import LocalActionPlanner
from hands.screenshot import ScreenshotCapture, ScreenshotFrame
from observability.logger import get_logger
from voice.feedback import (
    AssistantUtterance,
    feedback_for_unhandled_local_intent,
    feedback_from_hands_report,
)

log = get_logger(__name__)


class ScreenshotCaptureLike(Protocol):
    """Contrat minimal de capture d'ecran."""

    def capture(self, *, monitor_index: int = 0) -> ScreenshotFrame:
        """Capture un ecran et retourne une image exploitable par la vision."""


class VisionAnalyzerLike(Protocol):
    """Contrat minimal du client vision local."""

    async def analyze(self, *, user_request: str, image_base64: str) -> VisionDecision:
        """Produit une decision GUI validee."""


class HandsExecutorLike(Protocol):
    """Contrat minimal de l'executeur Hands."""

    def execute(
        self,
        decision: VisionDecision,
        *,
        frame: ScreenshotFrame | None = None,
    ) -> HandsExecutionReport:
        """Retourne le rapport dry-run pour une decision vision."""


class LocalActionPlannerLike(Protocol):
    """Contrat minimal du planificateur local sans vision."""

    def plan(self, event: IntentRouted) -> HandsExecutionReport | None:
        """Retourne un rapport local si l'intention est supportee."""


class HandsPipelineService:
    """Service reactif : IntentRouted(gui) -> rapport Hands dry-run."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        state_machine: StateMachine,
        screenshot: ScreenshotCaptureLike,
        vision: VisionAnalyzerLike,
        executor: HandsExecutorLike,
        local_actions: LocalActionPlannerLike,
        monitor_index: int = 0,
    ) -> None:
        self._bus = event_bus
        self._sm = state_machine
        self._screenshot = screenshot
        self._vision = vision
        self._executor = executor
        self._local_actions = local_actions
        self._monitor_index = monitor_index
        self._subscription: SubscriptionHandle | None = None

    @classmethod
    def create_default(
        cls,
        *,
        config: JarvisConfig,
        event_bus: EventBus,
        state_machine: StateMachine,
    ) -> HandsPipelineService:
        """Factory utilisee par `main.py`."""
        return cls(
            event_bus=event_bus,
            state_machine=state_machine,
            screenshot=ScreenshotCapture(config.vision),
            vision=LocalVisionClient(config.vision),
            executor=DryRunHandsExecutor(config.safety),
            local_actions=LocalActionPlanner(config.safety),
        )

    def start(self) -> None:
        """S'abonne aux intentions routees. Idempotent."""
        if self._subscription is not None and self._subscription.active:
            return
        self._subscription = self._bus.subscribe(IntentRouted, self._on_intent_routed)
        log.info("hands_pipeline_started")

    def stop(self) -> None:
        """Retire l'abonnement. Idempotent."""
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None
        log.info("hands_pipeline_stopped")

    async def _on_intent_routed(self, event: IntentRouted) -> None:
        local_report = self._local_actions.plan(event)
        if local_report is not None:
            await self._bus.publish(local_report)
            await self._publish_feedback(
                feedback_from_hands_report(local_report),
                reason=f"local_action_{local_report.status}",
            )
            log.info(
                "hands_local_action_reported",
                domain=event.domain,
                status=local_report.status,
                actions=len(local_report.actions),
                executed=local_report.executed,
                requires_human=local_report.requires_human,
            )
            return

        feedback = feedback_for_unhandled_local_intent(event)
        if feedback is not None:
            await self._publish_feedback(
                feedback,
                reason="local_action_unhandled",
            )
            log.info(
                "hands_local_action_unhandled",
                domain=event.domain,
                text=event.normalized_text,
            )
            return

        if not _should_use_vision_pipeline(event):
            return

        if not await self._transition_if_allowed(
            State.SCREENSHOT,
            reason="gui_intent_routed",
        ):
            return

        frame = self._capture_frame()
        if not await self._transition_if_allowed(
            State.VISION,
            reason="screenshot_captured",
        ):
            return

        decision = await self._decide(event, frame)
        if not await self._transition_if_allowed(
            State.ACTING,
            reason="vision_decision_ready",
        ):
            return

        report = self._executor.execute(decision, frame=frame)
        await self._bus.publish(report)
        log.info(
            "hands_pipeline_reported",
            domain=event.domain,
            status=report.status,
            actions=len(report.actions),
            executed=report.executed,
            requires_human=report.requires_human,
        )

        await self._finish_dry_run_cycle(report)

    def _capture_frame(self) -> ScreenshotFrame | None:
        try:
            frame = self._screenshot.capture(monitor_index=self._monitor_index)
            log.info(
                "hands_screenshot_captured",
                width=frame.width,
                height=frame.height,
                original_width=frame.original_width,
                original_height=frame.original_height,
                resized=frame.resized,
            )
            return frame
        except Exception as exc:
            log.warning(
                "hands_screenshot_failed",
                error=str(exc) or type(exc).__name__,
                error_type=type(exc).__name__,
            )
            return None

    async def _decide(
        self,
        event: IntentRouted,
        frame: ScreenshotFrame | None,
    ) -> VisionDecision:
        if frame is None:
            return human_required_decision(
                thought="Capture ecran impossible.",
                speech="Je ne peux pas capturer l'ecran pour le moment.",
            )

        try:
            decision = await self._vision.analyze(
                user_request=event.normalized_text,
                image_base64=frame.image_base64,
            )
        except Exception as exc:
            log.warning(
                "hands_vision_failed",
                error=str(exc) or type(exc).__name__,
                error_type=type(exc).__name__,
            )
            return human_required_decision(
                thought="Vision locale indisponible.",
                speech="Je ne peux pas analyser l'ecran pour le moment.",
            )

        log.info(
            "hands_vision_decision",
            confidence=round(decision.confidence, 3),
            actions=len(decision.actions),
            requires_human=decision.requires_human,
            task_complete=decision.task_complete,
        )
        return decision

    async def _finish_dry_run_cycle(self, report: HandsExecutionReport) -> None:
        if not await self._transition_if_allowed(
            State.VERIFYING,
            reason="hands_dry_run_reported",
        ):
            return
        if not await self._transition_if_allowed(
            State.SPEAKING,
            reason=f"dry_run_{report.status}",
        ):
            return
        await self._transition_if_allowed(State.IDLE, reason="dry_run_speaking_skipped")

    async def _publish_feedback(
        self,
        utterance: AssistantUtterance,
        *,
        reason: str,
    ) -> None:
        await self._transition_if_allowed(State.CHAT_ANSWER, reason=reason)
        await self._transition_if_allowed(State.SPEAKING, reason=reason)
        await self._bus.publish(utterance)
        log.info(
            "assistant_feedback_published",
            source=utterance.source,
            priority=utterance.priority,
            text=utterance.text,
            reason=utterance.reason,
        )
        await self._transition_if_allowed(State.IDLE, reason="assistant_feedback_spoken")

    async def _transition_if_allowed(self, target: State, *, reason: str) -> bool:
        current = self._sm.state
        if target in StateMachine.allowed_from(current):
            await self._sm.transition(target, reason=reason)
            return True
        log.debug(
            "hands_pipeline_transition_skipped",
            from_state=current.value,
            to_state=target.value,
            reason=reason,
        )
        return False


def _should_use_vision_pipeline(event: IntentRouted) -> bool:
    return event.intent == "gui" and event.domain == "vision"
