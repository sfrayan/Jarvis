"""Tests unitaires du service de dialogue 5H."""

from __future__ import annotations

import pytest

from brain.dialogue_service import DialogueService
from brain.events import (
    AssistantDraft,
    AssistantPlan,
    ClarificationQuestion,
    IntentDomain,
    IntentRouted,
    IntentType,
)
from core.event_bus import EventBus
from core.state_machine import State, StateMachine
from voice.feedback import AssistantUtterance

pytestmark = pytest.mark.unit


def _intent(
    text: str,
    *,
    intent: IntentType = "chat",
    domain: IntentDomain = "general",
) -> IntentRouted:
    return IntentRouted(
        timestamp=1.0,
        original_text=text,
        normalized_text=text,
        intent=intent,
        domain=domain,
        confidence=0.9,
        reason="test",
        model="test",
    )


class TestDialogueService:
    @pytest.mark.asyncio
    async def test_incomplete_task_publishes_question_without_relaying_intent(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        service = DialogueService(event_bus=bus, state_machine=sm)
        intents: list[IntentRouted] = []
        utterances: list[AssistantUtterance] = []
        questions: list[ClarificationQuestion] = []

        async def intent_handler(event: IntentRouted) -> None:
            intents.append(event)

        async def utterance_handler(event: AssistantUtterance) -> None:
            utterances.append(event)

        async def question_handler(event: ClarificationQuestion) -> None:
            questions.append(event)

        bus.subscribe(IntentRouted, intent_handler)
        bus.subscribe(AssistantUtterance, utterance_handler)
        bus.subscribe(ClarificationQuestion, question_handler)

        await service.process(_intent("j'ai un devoir a faire"))

        assert intents == []
        assert len(questions) == 1
        assert questions[0].kind == "homework"
        assert utterances[0].source == "dialogue"
        assert sm.state is State.IDLE

    @pytest.mark.asyncio
    async def test_clear_web_search_is_relayed_to_hands_channel(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        service = DialogueService(event_bus=bus, state_machine=sm)
        intents: list[IntentRouted] = []
        utterances: list[AssistantUtterance] = []

        async def intent_handler(event: IntentRouted) -> None:
            intents.append(event)

        async def utterance_handler(event: AssistantUtterance) -> None:
            utterances.append(event)

        bus.subscribe(IntentRouted, intent_handler)
        bus.subscribe(AssistantUtterance, utterance_handler)

        await service.process(
            _intent(
                "cherche sur YouTube les chats",
                intent="gui",
                domain="web_search",
            )
        )

        assert len(intents) == 1
        assert intents[0].domain == "web_search"
        assert utterances == []
        assert sm.state is State.IDLE

    @pytest.mark.asyncio
    async def test_routine_plan_is_not_relayed_to_hands_channel(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        service = DialogueService(event_bus=bus, state_machine=sm)
        intents: list[IntentRouted] = []
        utterances: list[AssistantUtterance] = []

        async def intent_handler(event: IntentRouted) -> None:
            intents.append(event)

        async def utterance_handler(event: AssistantUtterance) -> None:
            utterances.append(event)

        bus.subscribe(IntentRouted, intent_handler)
        bus.subscribe(AssistantUtterance, utterance_handler)

        await service.process(_intent("mode code", intent="gui", domain="routine"))

        assert intents == []
        assert utterances[0].source == "dialogue"
        assert "Mode code" in utterances[0].text
        assert sm.state is State.IDLE

    @pytest.mark.asyncio
    async def test_homework_demo_relay_search_after_clarification_and_plan(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        service = DialogueService(event_bus=bus, state_machine=sm)
        intents: list[IntentRouted] = []
        questions: list[ClarificationQuestion] = []
        plans: list[AssistantPlan] = []

        async def intent_handler(event: IntentRouted) -> None:
            intents.append(event)

        async def question_handler(event: ClarificationQuestion) -> None:
            questions.append(event)

        async def plan_handler(event: AssistantPlan) -> None:
            plans.append(event)

        bus.subscribe(IntentRouted, intent_handler)
        bus.subscribe(ClarificationQuestion, question_handler)
        bus.subscribe(AssistantPlan, plan_handler)

        await service.process(_intent("j'ai un devoir a faire"))
        await service.process(
            _intent("Consigne: exercice sur les fonctions en maths niveau seconde pour demain")
        )
        await service.process(_intent("commence par une recherche Google"))

        assert questions[0].kind == "homework"
        assert plans[0].kind == "homework"
        assert len(intents) == 1
        assert intents[0].domain == "web_search"
        assert intents[0].normalized_text.startswith("cherche sur Google exercice")

    @pytest.mark.asyncio
    async def test_homework_draft_is_published_without_relaying_intent(self) -> None:
        bus = EventBus()
        sm = StateMachine(bus, initial=State.ROUTING)
        service = DialogueService(event_bus=bus, state_machine=sm)
        intents: list[IntentRouted] = []
        drafts: list[AssistantDraft] = []
        utterances: list[AssistantUtterance] = []

        async def intent_handler(event: IntentRouted) -> None:
            intents.append(event)

        async def draft_handler(event: AssistantDraft) -> None:
            drafts.append(event)

        async def utterance_handler(event: AssistantUtterance) -> None:
            utterances.append(event)

        bus.subscribe(IntentRouted, intent_handler)
        bus.subscribe(AssistantDraft, draft_handler)
        bus.subscribe(AssistantUtterance, utterance_handler)

        await service.process(_intent("j'ai un devoir a faire"))
        await service.process(
            _intent("Consigne: exercice sur les fonctions en maths niveau seconde pour demain")
        )
        await service.process(_intent("commence par le brouillon"))

        assert intents == []
        assert len(drafts) == 1
        assert drafts[0].kind == "homework"
        assert drafts[0].title.startswith("Brouillon de maths")
        assert "Premiere version" in drafts[0].body
        assert utterances[-1].source == "dialogue"
        assert "brouillon structure" in utterances[-1].text
        assert sm.state is State.IDLE
