"""Pipeline GUI dry-run : screenshot -> vision -> hands.

Iteration 5C : ce service branche les briques 5A/5B sans execution reelle. Une
intention GUI locale declenche une capture d'ecran, une analyse vision locale,
puis un rapport `HandsExecutionReport` publie sur l'EventBus.

Iteration 5Q : les rapports bloques (`requires_human=True`) declenchent une
demande de confirmation vocale. L'utilisateur dit "oui" ou "non" et l'action
est executee ou rejetee.
"""

from __future__ import annotations

from typing import Protocol

from brain.confirmation import ConfirmationManager
from brain.events import ConfirmationResponse, IntentRouted, PendingConfirmation
from brain.vision_contracts import VisionDecision, human_required_decision
from brain.vision_local import LocalVisionClient
from config.schema import JarvisConfig
from core.event_bus import EventBus, SubscriptionHandle
from core.state_machine import State, StateMachine
from hands.browser_actions import BrowserActionPlanner
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


class BrowserActionPlannerLike(Protocol):
    """Contrat minimal du planificateur navigateur sans vision."""

    def plan(self, event: IntentRouted) -> HandsExecutionReport | None:
        """Retourne un rapport navigateur si l'intention est supportee."""


class _NoopBrowserActionPlanner:
    def plan(self, event: IntentRouted) -> HandsExecutionReport | None:
        """Ne planifie aucune action navigateur."""
        _ = event
        return None


class HandsPipelineService:
    """Service reactif : IntentRouted(gui) -> rapport Hands dry-run.

    En 5Q, les rapports bloques declenchent une confirmation vocale.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        state_machine: StateMachine,
        screenshot: ScreenshotCaptureLike,
        vision: VisionAnalyzerLike,
        executor: HandsExecutorLike,
        local_actions: LocalActionPlannerLike,
        browser_actions: BrowserActionPlannerLike | None = None,
        confirmation: ConfirmationManager | None = None,
        monitor_index: int = 0,
    ) -> None:
        self._bus = event_bus
        self._sm = state_machine
        self._screenshot = screenshot
        self._vision = vision
        self._executor = executor
        self._local_actions = local_actions
        self._browser_actions = browser_actions or _NoopBrowserActionPlanner()
        self._confirmation = confirmation or ConfirmationManager()
        self._monitor_index = monitor_index
        self._subscription: SubscriptionHandle | None = None
        self._confirm_subscription: SubscriptionHandle | None = None

    @classmethod
    def create_default(
        cls,
        *,
        config: JarvisConfig,
        event_bus: EventBus,
        state_machine: StateMachine,
        confirmation: ConfirmationManager | None = None,
    ) -> HandsPipelineService:
        """Factory utilisee par `main.py`."""
        return cls(
            event_bus=event_bus,
            state_machine=state_machine,
            screenshot=ScreenshotCapture(config.vision),
            vision=LocalVisionClient(config.vision),
            executor=DryRunHandsExecutor(config.safety),
            local_actions=LocalActionPlanner(config.safety),
            browser_actions=BrowserActionPlanner(config.safety),
            confirmation=confirmation,
        )

    def start(self) -> None:
        """S'abonne aux intentions routees et aux confirmations. Idempotent."""
        if self._subscription is not None and self._subscription.active:
            return
        self._subscription = self._bus.subscribe(IntentRouted, self._on_intent_routed)
        self._confirm_subscription = self._bus.subscribe(
            ConfirmationResponse, self._on_confirmation_response,
        )
        log.info("hands_pipeline_started")

    def stop(self) -> None:
        """Retire les abonnements. Idempotent."""
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None
        if self._confirm_subscription is not None:
            self._confirm_subscription.unsubscribe()
            self._confirm_subscription = None
        self._confirmation.clear()
        log.info("hands_pipeline_stopped")

    async def _on_intent_routed(self, event: IntentRouted) -> None:
        local_report = self._local_actions.plan(event)
        if local_report is not None:
            if local_report.requires_human:
                await self._request_confirmation(local_report)
                return
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

        browser_report = self._browser_actions.plan(event)
        if browser_report is not None:
            await self._bus.publish(browser_report)
            await self._publish_feedback(
                feedback_from_hands_report(browser_report),
                reason=f"browser_action_{browser_report.status}",
            )
            log.info(
                "hands_browser_action_reported",
                domain=event.domain,
                status=browser_report.status,
                actions=len(browser_report.actions),
                executed=browser_report.executed,
                requires_human=browser_report.requires_human,
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

    # ------------------------------------------------------------------
    # Confirmations explicites (5Q)
    # ------------------------------------------------------------------
    async def _request_confirmation(self, report: HandsExecutionReport) -> None:
        """Enregistre l'action et publie la demande de confirmation."""
        pending = self._confirmation.request_confirmation(report)
        await self._bus.publish(pending)
        await self._publish_feedback(
            AssistantUtterance(
                timestamp=pending.timestamp,
                text=pending.question,
                source="hands",
                priority="warning",
                reason=pending.reason,
            ),
            reason="confirmation_requested",
        )

    async def _on_confirmation_response(self, event: ConfirmationResponse) -> None:
        """Traite la reponse a une confirmation pendante."""
        pending = self._confirmation.pending
        if pending is None:
            log.debug(
                "confirmation_response_no_pending",
                confirmation_id=event.confirmation_id,
            )
            return

        if event.confirmation_id != pending.confirmation_id:
            log.debug(
                "confirmation_response_id_mismatch",
                expected=pending.confirmation_id,
                received=event.confirmation_id,
            )
            return

        self._confirmation.clear()

        if event.verdict == "confirmed":
            await self._execute_confirmed(pending.report)
        else:
            reason = "Confirmation rejetee" if event.verdict == "rejected" else "Confirmation expiree"
            await self._publish_feedback(
                AssistantUtterance(
                    timestamp=event.timestamp,
                    text=f"D'accord, j'annule l'action. {reason}.",
                    source="hands",
                    priority="info",
                    reason=reason,
                ),
                reason=f"confirmation_{event.verdict}",
            )

    async def _execute_confirmed(self, report: HandsExecutionReport) -> None:
        """Execute l'action apres confirmation explicite."""
        import time as _time

        from hands.local_actions import PyAutoGuiLocalActionBackend

        if not report.actions:
            return

        action = report.actions[0]
        if report.mode not in {"assisted", "autonomous"}:
            # En dry_run/observe, on ne fait que publier le rapport initial
            await self._bus.publish(report)
            await self._publish_feedback(
                feedback_from_hands_report(report),
                reason="confirmed_but_mode_prevents_execution",
            )
            return

        backend = PyAutoGuiLocalActionBackend()
        try:
            backend.perform(action)
        except Exception as exc:
            log.warning(
                "hands_confirmed_execution_failed",
                error=str(exc),
                action_type=action.type,
            )
            await self._publish_feedback(
                AssistantUtterance(
                    timestamp=_time.time(),
                    text=f"L'action a echoue apres confirmation: {type(exc).__name__}.",
                    source="hands",
                    priority="error",
                    reason=f"confirmed_execution_failed: {exc}",
                ),
                reason="confirmed_execution_failed",
            )
            return

        executed_report = HandsExecutionReport(
            status="completed",
            mode=report.mode,
            actions=report.actions,
            executed=True,
            requires_human=False,
            reason="Action executee apres confirmation explicite",
        )
        await self._bus.publish(executed_report)
        await self._publish_feedback(
            feedback_from_hands_report(executed_report),
            reason="confirmed_execution_completed",
        )


def _should_use_vision_pipeline(event: IntentRouted) -> bool:
    return event.intent == "gui" and event.domain == "vision"
