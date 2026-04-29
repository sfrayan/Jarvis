"""Tests unitaires du pipeline Hands 5C."""

from __future__ import annotations

import pytest

from brain.events import IntentDomain, IntentRouted, IntentType
from brain.vision_contracts import VisionAction, VisionDecision
from core.event_bus import EventBus
from core.state_machine import State, StateMachine, StateTransition
from hands.executor import HandsExecutionReport, HandsExecutionStatus, PlannedGuiAction
from hands.screenshot import ScreenshotFrame
from hands.service import HandsPipelineService
from voice.feedback import AssistantUtterance

pytestmark = pytest.mark.unit


class _FakeScreenshot:
    def __init__(self, *, frame: ScreenshotFrame | None = None, fail: bool = False) -> None:
        self.frame = frame or _frame()
        self.fail = fail
        self.calls: list[int] = []

    def capture(self, *, monitor_index: int = 0) -> ScreenshotFrame:
        self.calls.append(monitor_index)
        if self.fail:
            raise RuntimeError("capture impossible")
        return self.frame


class _FakeVision:
    def __init__(self, decision: VisionDecision | None = None) -> None:
        self.decision = decision or _decision()
        self.requests: list[tuple[str, str]] = []

    async def analyze(self, *, user_request: str, image_base64: str) -> VisionDecision:
        self.requests.append((user_request, image_base64))
        return self.decision


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[VisionDecision, ScreenshotFrame | None]] = []

    def execute(
        self,
        decision: VisionDecision,
        *,
        frame: ScreenshotFrame | None = None,
    ) -> HandsExecutionReport:
        self.calls.append((decision, frame))
        status: HandsExecutionStatus = "blocked" if decision.requires_human else "dry_run"
        return HandsExecutionReport(
            status=status,
            mode="dry_run",
            actions=(),
            executed=False,
            requires_human=decision.requires_human,
            reason="rapport fake",
        )


class _FakeLocalActions:
    def __init__(self, report: HandsExecutionReport | None = None) -> None:
        self.report = report
        self.calls: list[IntentRouted] = []

    def plan(self, event: IntentRouted) -> HandsExecutionReport | None:
        self.calls.append(event)
        return self.report


class _FakeBrowserActions:
    def __init__(self, report: HandsExecutionReport | None = None) -> None:
        self.report = report
        self.calls: list[IntentRouted] = []

    def plan(self, event: IntentRouted) -> HandsExecutionReport | None:
        self.calls.append(event)
        return self.report


def _intent(
    *,
    intent: IntentType = "gui",
    domain: IntentDomain = "vision",
    text: str = "clique sur enregistrer",
) -> IntentRouted:
    return IntentRouted(
        timestamp=1.0,
        original_text=text,
        normalized_text=text,
        intent=intent,
        domain=domain,
        confidence=0.95,
        reason="test",
        model="qwen3:latest",
    )


def _frame() -> ScreenshotFrame:
    return ScreenshotFrame(
        image_base64="image-base64",
        width=100,
        height=50,
        original_width=200,
        original_height=100,
        left=10,
        top=20,
    )


def _decision() -> VisionDecision:
    return VisionDecision(
        thought="Je vois le bouton.",
        confidence=0.9,
        speech="Je prepare le clic.",
        actions=[VisionAction(type="left_click", x=10, y=20)],
        external_tools=[],
        requires_human=False,
        task_complete=False,
    )


def _report() -> HandsExecutionReport:
    return HandsExecutionReport(
        status="dry_run",
        mode="dry_run",
        actions=(),
        executed=False,
        requires_human=False,
        reason="rapport local fake",
    )


def _browser_report() -> HandsExecutionReport:
    return HandsExecutionReport(
        status="dry_run",
        mode="dry_run",
        actions=(PlannedGuiAction(type="browser_navigate", text="https://www.youtube.com"),),
        executed=False,
        requires_human=False,
        reason="rapport navigateur fake",
    )


class TestHandsPipelineService:
    @pytest.mark.asyncio
    async def test_pipeline_publishes_dry_run_report_and_returns_idle(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        screenshot = _FakeScreenshot()
        vision = _FakeVision()
        executor = _FakeExecutor()
        service = HandsPipelineService(
            event_bus=bus,
            state_machine=sm,
            screenshot=screenshot,
            vision=vision,
            executor=executor,
            local_actions=_FakeLocalActions(),
        )
        reports: list[HandsExecutionReport] = []
        transitions: list[StateTransition] = []

        async def report_handler(event: HandsExecutionReport) -> None:
            reports.append(event)

        async def transition_handler(event: StateTransition) -> None:
            transitions.append(event)

        bus.subscribe(HandsExecutionReport, report_handler)
        bus.subscribe(StateTransition, transition_handler)
        service.start()

        await bus.publish(_intent())

        assert screenshot.calls == [0]
        assert vision.requests == [("clique sur enregistrer", "image-base64")]
        assert executor.calls[0][1] == screenshot.frame
        assert reports[0].status == "dry_run"
        assert sm.state is State.IDLE
        assert [(t.from_state, t.to_state) for t in transitions] == [
            (State.ROUTING, State.SCREENSHOT),
            (State.SCREENSHOT, State.VISION),
            (State.VISION, State.ACTING),
            (State.ACTING, State.VERIFYING),
            (State.VERIFYING, State.SPEAKING),
            (State.SPEAKING, State.IDLE),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain", ["apps", "folders", "media", "system"])
    async def test_local_domains_publish_dry_run_report_without_vision(
        self,
        domain: IntentDomain,
    ) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        screenshot = _FakeScreenshot()
        local_actions = _FakeLocalActions(_report())
        service = HandsPipelineService(
            event_bus=bus,
            state_machine=sm,
            screenshot=screenshot,
            vision=_FakeVision(),
            executor=_FakeExecutor(),
            local_actions=local_actions,
        )
        reports: list[HandsExecutionReport] = []
        utterances: list[AssistantUtterance] = []

        async def report_handler(event: HandsExecutionReport) -> None:
            reports.append(event)

        async def utterance_handler(event: AssistantUtterance) -> None:
            utterances.append(event)

        bus.subscribe(HandsExecutionReport, report_handler)
        bus.subscribe(AssistantUtterance, utterance_handler)
        service.start()

        await bus.publish(_intent(domain=domain, text="ouvre Chrome"))

        assert local_actions.calls[0].domain == domain
        assert screenshot.calls == []
        assert reports[0].status == "dry_run"
        assert utterances[0].text == "En mode dry run, je n'exécute rien pour l'instant."
        assert sm.state is State.IDLE

    @pytest.mark.asyncio
    async def test_web_search_domain_publishes_browser_report_without_vision(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        screenshot = _FakeScreenshot()
        vision = _FakeVision()
        local_actions = _FakeLocalActions()
        browser_actions = _FakeBrowserActions(_browser_report())
        service = HandsPipelineService(
            event_bus=bus,
            state_machine=sm,
            screenshot=screenshot,
            vision=vision,
            executor=_FakeExecutor(),
            local_actions=local_actions,
            browser_actions=browser_actions,
        )
        reports: list[HandsExecutionReport] = []
        utterances: list[AssistantUtterance] = []

        async def report_handler(event: HandsExecutionReport) -> None:
            reports.append(event)

        async def utterance_handler(event: AssistantUtterance) -> None:
            utterances.append(event)

        bus.subscribe(HandsExecutionReport, report_handler)
        bus.subscribe(AssistantUtterance, utterance_handler)
        service.start()

        await bus.publish(_intent(domain="web_search", text="ouvre YouTube"))

        assert local_actions.calls[0].domain == "web_search"
        assert browser_actions.calls[0].domain == "web_search"
        assert screenshot.calls == []
        assert vision.requests == []
        assert reports[0].actions[0].type == "browser_navigate"
        assert utterances[0].text == "Mode dry run: je n'exécute pas encore ouvrir YouTube."
        assert sm.state is State.IDLE

    @pytest.mark.asyncio
    async def test_web_search_without_browser_plan_keeps_existing_fallback(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        screenshot = _FakeScreenshot()
        vision = _FakeVision()
        browser_actions = _FakeBrowserActions()
        service = HandsPipelineService(
            event_bus=bus,
            state_machine=sm,
            screenshot=screenshot,
            vision=vision,
            executor=_FakeExecutor(),
            local_actions=_FakeLocalActions(),
            browser_actions=browser_actions,
        )
        reports: list[HandsExecutionReport] = []
        utterances: list[AssistantUtterance] = []

        async def report_handler(event: HandsExecutionReport) -> None:
            reports.append(event)

        async def utterance_handler(event: AssistantUtterance) -> None:
            utterances.append(event)

        bus.subscribe(HandsExecutionReport, report_handler)
        bus.subscribe(AssistantUtterance, utterance_handler)
        service.start()

        await bus.publish(_intent(domain="web_search", text="commande navigateur inconnue"))

        assert browser_actions.calls[0].domain == "web_search"
        assert screenshot.calls == []
        assert vision.requests == []
        assert reports == []
        assert utterances == []
        assert sm.state is State.ROUTING

    @pytest.mark.asyncio
    async def test_local_domain_without_action_publishes_feedback(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        screenshot = _FakeScreenshot()
        local_actions = _FakeLocalActions()
        service = HandsPipelineService(
            event_bus=bus,
            state_machine=sm,
            screenshot=screenshot,
            vision=_FakeVision(),
            executor=_FakeExecutor(),
            local_actions=local_actions,
        )
        utterances: list[AssistantUtterance] = []

        async def utterance_handler(event: AssistantUtterance) -> None:
            utterances.append(event)

        bus.subscribe(AssistantUtterance, utterance_handler)
        service.start()

        await bus.publish(_intent(domain="apps", text="ouvre Obsidian"))

        assert local_actions.calls[0].domain == "apps"
        assert screenshot.calls == []
        assert utterances[0].text == "Je ne trouve pas cette application dans ton inventaire local."
        assert utterances[0].priority == "warning"
        assert sm.state is State.IDLE

    @pytest.mark.asyncio
    async def test_ignores_non_local_non_vision_gui_domains(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        screenshot = _FakeScreenshot()
        local_actions = _FakeLocalActions()
        service = HandsPipelineService(
            event_bus=bus,
            state_machine=sm,
            screenshot=screenshot,
            vision=_FakeVision(),
            executor=_FakeExecutor(),
            local_actions=local_actions,
        )
        service.start()

        await bus.publish(_intent(domain="google_workspace", text="ouvre Gmail"))

        assert local_actions.calls[0].domain == "google_workspace"
        assert screenshot.calls == []
        assert sm.state is State.ROUTING

    @pytest.mark.asyncio
    async def test_ignores_non_gui_intents(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        screenshot = _FakeScreenshot()
        service = HandsPipelineService(
            event_bus=bus,
            state_machine=sm,
            screenshot=screenshot,
            vision=_FakeVision(),
            executor=_FakeExecutor(),
            local_actions=_FakeLocalActions(),
        )
        service.start()

        await bus.publish(_intent(intent="chat", domain="general", text="bonjour"))

        assert screenshot.calls == []
        assert sm.state is State.ROUTING

    @pytest.mark.asyncio
    async def test_screenshot_failure_publishes_blocked_report(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        vision = _FakeVision()
        executor = _FakeExecutor()
        service = HandsPipelineService(
            event_bus=bus,
            state_machine=sm,
            screenshot=_FakeScreenshot(fail=True),
            vision=vision,
            executor=executor,
            local_actions=_FakeLocalActions(),
        )
        reports: list[HandsExecutionReport] = []

        async def report_handler(event: HandsExecutionReport) -> None:
            reports.append(event)

        bus.subscribe(HandsExecutionReport, report_handler)
        service.start()

        await bus.publish(_intent())

        assert vision.requests == []
        assert executor.calls[0][0].requires_human is True
        assert executor.calls[0][1] is None
        assert reports[0].status == "blocked"
        assert sm.state is State.IDLE

    def test_start_and_stop_manage_subscription(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus)
        service = HandsPipelineService(
            event_bus=bus,
            state_machine=sm,
            screenshot=_FakeScreenshot(),
            vision=_FakeVision(),
            executor=_FakeExecutor(),
            local_actions=_FakeLocalActions(),
        )

        service.start()
        service.start()
        assert bus.subscriber_count(IntentRouted) == 1

        service.stop()
        assert bus.subscriber_count(IntentRouted) == 0
