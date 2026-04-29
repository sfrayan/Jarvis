"""Dialogue interactif et sessions de tache.

Le `DialogueManager` reste une brique pure : pas d'EventBus, pas de GUI, pas de
TTS direct. Il decide seulement si une intention routee doit etre relayee vers
Hands, clarifiee, transformee en plan, ou annulee.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from brain.events import (
    AssistantPlan,
    ClarificationQuestion,
    IntentRouted,
    TaskSessionStateChanged,
)
from brain.routines import RoutinePlan, plan_routine
from brain.task_session import (
    TaskSession,
    TaskSessionKind,
    TaskSlot,
    make_session_id,
    merge_slots,
)
from voice.feedback import AssistantFeedbackPriority, AssistantUtterance

DialogueDecision = Literal["pass_through", "clarify", "plan", "respond", "cancel"]

_HOMEWORK_REQUIRED = ("instruction", "subject", "level", "deadline")
_CODING_REQUIRED = ("app_type",)
_WORK_SETUP_REQUIRED = ("work_type",)
_VIDEO_REQUIRED = ("topic",)
_SENSITIVE_REQUIRED = ("service_name", "reason")


class DialogueTurn(BaseModel):
    """Decision produite pour une phrase utilisateur."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: DialogueDecision
    reason: str = Field(..., min_length=1)
    intent: IntentRouted | None = None
    utterance: AssistantUtterance | None = None
    clarification: ClarificationQuestion | None = None
    plan: AssistantPlan | None = None
    session_event: TaskSessionStateChanged | None = None


@dataclass(frozen=True)
class _SessionTemplate:
    kind: TaskSessionKind
    summary: str
    missing_slots: tuple[str, ...]
    question: str
    reason: str


class DialogueManager:
    """Gere une session de tache temporaire en RAM."""

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.time
        self._session: TaskSession | None = None

    @property
    def active_session(self) -> TaskSession | None:
        """Retourne la session courante, si elle est encore utile."""
        if self._session is None or self._session.status in {"cancelled", "completed"}:
            return None
        return self._session

    def handle(self, routed: IntentRouted) -> DialogueTurn:
        """Retourne la decision dialogue pour une intention routee."""
        now = self._clock()
        text = _fold(routed.normalized_text)

        if _is_cancel_request(text):
            return self._cancel_session(now=now)

        active = self.active_session
        if active is not None and active.status == "waiting_for_user":
            return self._continue_session(active, routed, now=now)
        if active is not None and active.status == "ready":
            ready_turn = self._continue_ready_session(active, routed, text, now=now)
            if ready_turn is not None:
                return ready_turn

        template = _detect_incomplete_request(routed, text)
        if template is not None:
            return self._start_session(routed, template, now=now)

        routine = plan_routine(routed.normalized_text)
        if routine is not None and (
            routed.domain == "routine" or _looks_like_routine_request(text)
        ):
            return self._routine_plan_turn(routed, routine, now=now)

        if routed.intent == "chat":
            return self._chat_response(routed, text, now=now)

        if routed.intent == "unknown":
            return self._unknown_response(routed, now=now)

        return DialogueTurn(
            decision="pass_through",
            intent=routed,
            reason="Intention claire relayee vers Hands",
        )

    def _start_session(
        self,
        routed: IntentRouted,
        template: _SessionTemplate,
        *,
        now: float,
    ) -> DialogueTurn:
        session = TaskSession(
            session_id=make_session_id(now),
            kind=template.kind,
            status="waiting_for_user",
            original_request=routed.normalized_text,
            summary=template.summary,
            missing_slots=template.missing_slots,
            created_at=now,
            updated_at=now,
            last_question=template.question,
        )
        self._session = session
        return self._clarification_turn(
            session,
            question=template.question,
            reason=template.reason,
            now=now,
            priority="warning" if template.kind == "sensitive_system" else "info",
        )

    def _continue_session(
        self,
        session: TaskSession,
        routed: IntentRouted,
        *,
        now: float,
    ) -> DialogueTurn:
        text = _fold(routed.normalized_text)
        if session.kind == "homework":
            return self._continue_homework(session, routed, text, now=now)
        if session.kind == "coding_project":
            return self._continue_coding_project(session, routed, text, now=now)
        if session.kind == "work_setup":
            return self._continue_work_setup(session, routed, text, now=now)
        if session.kind == "web_video":
            return self._continue_web_video(session, routed, text, now=now)
        if session.kind == "sensitive_system":
            return self._continue_sensitive_system(session, routed, text, now=now)
        return self._unknown_response(routed, now=now)

    def _continue_ready_session(
        self,
        session: TaskSession,
        routed: IntentRouted,
        text: str,
        *,
        now: float,
    ) -> DialogueTurn | None:
        if session.kind == "homework":
            return self._continue_ready_homework(session, routed, text, now=now)
        return None

    def _continue_ready_homework(
        self,
        session: TaskSession,
        routed: IntentRouted,
        text: str,
        *,
        now: float,
    ) -> DialogueTurn | None:
        if _looks_like_homework_research_choice(text):
            return self._homework_research_turn(session, routed, text, now=now)
        if _looks_like_homework_draft_choice(text):
            return self._homework_draft_turn(session, routed, now=now)
        return None

    def _continue_homework(
        self,
        session: TaskSession,
        routed: IntentRouted,
        text: str,
        *,
        now: float,
    ) -> DialogueTurn:
        slots = merge_slots(session.slots, _extract_homework_slots(routed.normalized_text, text))
        missing = _missing_slots(slots, _HOMEWORK_REQUIRED)
        if missing:
            question = _homework_missing_question(missing)
            updated = session.with_update(
                now=now,
                slots=slots,
                missing_slots=missing,
                last_question=question,
                last_user_reply=routed.normalized_text,
            )
            self._session = updated
            return self._clarification_turn(
                updated,
                question=question,
                reason="Session devoir encore incomplete",
                now=now,
            )

        steps = (
            "Reformuler la consigne et verifier les criteres attendus.",
            "Decouper le devoir en parties courtes et ordonnees.",
            "Construire un brouillon, puis relire avec la date limite en tete.",
        )
        updated = session.with_update(
            now=now,
            status="ready",
            slots=slots,
            missing_slots=(),
            plan_steps=steps,
            last_user_reply=routed.normalized_text,
        )
        self._session = updated
        return self._plan_turn(
            updated,
            steps=steps,
            reason="Informations devoir suffisantes pour proposer un plan",
            now=now,
        )

    def _continue_coding_project(
        self,
        session: TaskSession,
        routed: IntentRouted,
        text: str,
        *,
        now: float,
    ) -> DialogueTurn:
        slots = merge_slots(session.slots, _extract_coding_slots(text))
        missing = _missing_slots(slots, _CODING_REQUIRED)
        if missing:
            question = "Quel type d'application veux-tu creer : web, desktop, bot, script Python ou autre ?"
            updated = session.with_update(
                now=now,
                slots=slots,
                missing_slots=missing,
                last_question=question,
                last_user_reply=routed.normalized_text,
            )
            self._session = updated
            return self._clarification_turn(
                updated,
                question=question,
                reason="Type d'application encore manquant",
                now=now,
            )

        steps = (
            "Definir l'objectif de l'application et son utilisateur principal.",
            "Lister les trois premieres fonctionnalites utiles.",
            "Choisir la structure de projet avant d'ouvrir les outils.",
        )
        updated = session.with_update(
            now=now,
            status="ready",
            slots=slots,
            missing_slots=(),
            plan_steps=steps,
            last_user_reply=routed.normalized_text,
        )
        self._session = updated
        return self._plan_turn(
            updated,
            steps=steps,
            reason="Type de projet code renseigne",
            now=now,
        )

    def _continue_work_setup(
        self,
        session: TaskSession,
        routed: IntentRouted,
        text: str,
        *,
        now: float,
    ) -> DialogueTurn:
        slots = merge_slots(session.slots, _extract_work_setup_slots(text))
        missing = _missing_slots(slots, _WORK_SETUP_REQUIRED)
        if missing:
            question = "Tu veux travailler sur quoi : code, devoir, documents, recherche web ou organisation ?"
            updated = session.with_update(
                now=now,
                slots=slots,
                missing_slots=missing,
                last_question=question,
                last_user_reply=routed.normalized_text,
            )
            self._session = updated
            return self._clarification_turn(
                updated,
                question=question,
                reason="Type de routine de travail manquant",
                now=now,
            )

        steps = (
            "Preparer uniquement les outils utiles au type de travail choisi.",
            "Proposer les ouvertures d'applications avant de les executer.",
            "Garder la session active pour ajuster l'environnement.",
        )
        updated = session.with_update(
            now=now,
            status="ready",
            slots=slots,
            missing_slots=(),
            plan_steps=steps,
            last_user_reply=routed.normalized_text,
        )
        self._session = updated
        return self._plan_turn(
            updated,
            steps=steps,
            reason="Routine de travail suffisamment precise",
            now=now,
        )

    def _continue_web_video(
        self,
        session: TaskSession,
        routed: IntentRouted,
        text: str,
        *,
        now: float,
    ) -> DialogueTurn:
        slots = merge_slots(session.slots, _extract_video_slots(text))
        missing = _missing_slots(slots, _VIDEO_REQUIRED)
        if missing:
            question = "Sur quel sujet veux-tu que je cherche une video YouTube ?"
            updated = session.with_update(
                now=now,
                slots=slots,
                missing_slots=missing,
                last_question=question,
                last_user_reply=routed.normalized_text,
            )
            self._session = updated
            return self._clarification_turn(
                updated,
                question=question,
                reason="Sujet de recherche video manquant",
                now=now,
            )

        topic = _slot_value(slots, "topic") or routed.normalized_text
        updated = session.with_update(
            now=now,
            status="completed",
            slots=slots,
            missing_slots=(),
            last_user_reply=routed.normalized_text,
        )
        self._session = updated
        return DialogueTurn(
            decision="pass_through",
            intent=IntentRouted(
                timestamp=now,
                original_text=routed.original_text,
                normalized_text=f"ouvre une recherche YouTube sur {topic}",
                intent="gui",
                domain="web_search",
                confidence=0.92,
                reason="Session video completee par clarification",
                model="dialogue",
            ),
            session_event=_session_event(updated, reason="Session video completee", now=now),
            reason="Sujet video clarifie, recherche YouTube sure",
        )

    def _continue_sensitive_system(
        self,
        session: TaskSession,
        routed: IntentRouted,
        text: str,
        *,
        now: float,
    ) -> DialogueTurn:
        slots = merge_slots(session.slots, _extract_sensitive_slots(text))
        missing = _missing_slots(slots, _SENSITIVE_REQUIRED)
        if missing:
            question = (
                "Action sensible. Dis-moi le service exact et pourquoi tu veux le modifier. "
                "Je demanderai ensuite une confirmation explicite avant toute action."
            )
        else:
            question = (
                "J'ai assez d'informations, mais je ne toucherai pas aux services Windows "
                "sans confirmation explicite et justification admin."
            )
        updated = session.with_update(
            now=now,
            slots=slots,
            missing_slots=missing,
            last_question=question,
            last_user_reply=routed.normalized_text,
        )
        self._session = updated
        return self._clarification_turn(
            updated,
            question=question,
            reason="Action systeme sensible bloquee par dialogue",
            now=now,
            priority="warning",
        )

    def _chat_response(self, routed: IntentRouted, text: str, *, now: float) -> DialogueTurn:
        if re.search(r"\b(salut|bonjour|bonsoir|coucou)\b", text):
            message = "Salut. Je suis pret. Dis-moi ce que tu veux faire."
        elif re.search(r"\b(merci)\b", text):
            message = "Avec plaisir."
        else:
            message = (
                "Je peux t'aider. Donne-moi l'objectif, le contexte et ce que tu veux "
                "obtenir, puis je te proposerai la prochaine etape."
            )
        return DialogueTurn(
            decision="respond",
            utterance=_utterance(message, reason=routed.reason, now=now),
            reason="Reponse conversationnelle courte",
        )

    def _unknown_response(self, routed: IntentRouted, *, now: float) -> DialogueTurn:
        message = (
            "Je n'ai pas assez compris. Reformule avec l'objectif, la cible, et ce que "
            "tu veux que je fasse."
        )
        return DialogueTurn(
            decision="clarify",
            utterance=_utterance(message, reason=routed.reason, now=now, priority="warning"),
            reason="Intention inconnue clarifiee sans action",
        )

    def _cancel_session(self, *, now: float) -> DialogueTurn:
        session = self.active_session
        if session is None:
            return DialogueTurn(
                decision="respond",
                utterance=_utterance(
                    "Aucune session active a annuler.",
                    reason="Annulation sans session",
                    now=now,
                ),
                reason="Annulation sans session active",
            )

        updated = session.with_update(now=now, status="cancelled", missing_slots=())
        self._session = updated
        return DialogueTurn(
            decision="cancel",
            utterance=_utterance(
                "D'accord, j'annule cette tache et je repars proprement.",
                reason="Session annulee par l'utilisateur",
                now=now,
            ),
            session_event=_session_event(updated, reason="Session annulee", now=now),
            reason="Session annulee",
        )

    def _clarification_turn(
        self,
        session: TaskSession,
        *,
        question: str,
        reason: str,
        now: float,
        priority: AssistantFeedbackPriority = "info",
    ) -> DialogueTurn:
        return DialogueTurn(
            decision="clarify",
            utterance=_utterance(question, reason=reason, now=now, priority=priority),
            clarification=ClarificationQuestion(
                timestamp=now,
                session_id=session.session_id,
                kind=session.kind,
                question=question,
                missing_slots=session.missing_slots,
                reason=reason,
            ),
            session_event=_session_event(session, reason=reason, now=now),
            reason=reason,
        )

    def _plan_turn(
        self,
        session: TaskSession,
        *,
        steps: tuple[str, ...],
        reason: str,
        now: float,
    ) -> DialogueTurn:
        message = _plan_message(session, steps)
        plan = AssistantPlan(
            timestamp=now,
            session_id=session.session_id,
            kind=session.kind,
            summary=session.summary,
            steps=steps,
            requires_confirmation=True,
            reason=reason,
        )
        return DialogueTurn(
            decision="plan",
            utterance=_utterance(message, reason=reason, now=now),
            plan=plan,
            session_event=_session_event(session, reason=reason, now=now),
            reason=reason,
        )

    def _routine_plan_turn(
        self,
        routed: IntentRouted,
        routine: RoutinePlan,
        *,
        now: float,
    ) -> DialogueTurn:
        session = TaskSession(
            session_id=make_session_id(now),
            kind="work_setup",
            status="ready",
            original_request=routed.normalized_text,
            summary=routine.summary,
            missing_slots=(),
            plan_steps=routine.steps,
            created_at=now,
            updated_at=now,
        )
        self._session = session
        plan = AssistantPlan(
            timestamp=now,
            session_id=session.session_id,
            kind=session.kind,
            summary=routine.summary,
            steps=routine.steps,
            requires_confirmation=True,
            reason=f"Routine sure: {routine.title}",
        )
        return DialogueTurn(
            decision="plan",
            utterance=_utterance(
                _routine_message(routine),
                reason=plan.reason,
                now=now,
            ),
            plan=plan,
            session_event=_session_event(session, reason=plan.reason, now=now),
            reason=plan.reason,
        )

    def _homework_research_turn(
        self,
        session: TaskSession,
        routed: IntentRouted,
        text: str,
        *,
        now: float,
    ) -> DialogueTurn:
        target = "youtube" if _prefers_youtube(text) else "google"
        query = _homework_search_query(session)
        normalized_text = (
            f"ouvre une recherche YouTube sur {query}"
            if target == "youtube"
            else f"cherche sur Google {query}"
        )
        updated = session.with_update(
            now=now,
            last_user_reply=routed.normalized_text,
        )
        self._session = updated
        return DialogueTurn(
            decision="pass_through",
            intent=IntentRouted(
                timestamp=now,
                original_text=routed.original_text,
                normalized_text=normalized_text,
                intent="gui",
                domain="web_search",
                confidence=0.93,
                reason="Session devoir: recherche demandee apres le plan",
                model="dialogue",
            ),
            session_event=_session_event(
                updated,
                reason="Recherche devoir preparee",
                now=now,
            ),
            reason="Recherche devoir relayee vers BrowserActionPlanner",
        )

    def _homework_draft_turn(
        self,
        session: TaskSession,
        routed: IntentRouted,
        *,
        now: float,
    ) -> DialogueTurn:
        steps = _homework_draft_steps(session)
        updated = session.with_update(
            now=now,
            plan_steps=steps,
            last_user_reply=routed.normalized_text,
        )
        self._session = updated
        plan = AssistantPlan(
            timestamp=now,
            session_id=updated.session_id,
            kind=updated.kind,
            summary="Brouillon guide du devoir",
            steps=steps,
            requires_confirmation=False,
            reason="Brouillon devoir demande apres le plan",
        )
        return DialogueTurn(
            decision="plan",
            utterance=_utterance(
                _homework_draft_message(updated, steps),
                reason=plan.reason,
                now=now,
            ),
            plan=plan,
            session_event=_session_event(updated, reason=plan.reason, now=now),
            reason=plan.reason,
        )


def _plan_message(session: TaskSession, steps: tuple[str, ...]) -> str:
    numbered = " ".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    return f"D'accord. Je te propose {len(steps)} etapes: {numbered} {_next_step_question(session)}"


def _next_step_question(session: TaskSession) -> str:
    if session.kind == "homework":
        return "Tu veux que je commence par le brouillon ou par une recherche ?"
    if session.kind == "coding_project":
        return "Donne-moi les fonctionnalites principales, et je structure le projet."
    if session.kind == "work_setup":
        return "Dis-moi les outils a ouvrir, et je les preparerai prudemment."
    return "Dis-moi par quoi tu veux commencer."


def _routine_message(routine: RoutinePlan) -> str:
    steps = " ".join(f"{index}. {step}" for index, step in enumerate(routine.steps, start=1))
    suggestions = _routine_suggestions_text(routine)
    return (
        f"{routine.title}. Je propose un plan prudent: {steps}{suggestions} {routine.next_question}"
    )


def _routine_suggestions_text(routine: RoutinePlan) -> str:
    if not routine.suggestions:
        return ""
    labels = ", ".join(suggestion.label for suggestion in routine.suggestions)
    return f" Options possibles ensuite: {labels}."


def _looks_like_homework_research_choice(text: str) -> bool:
    return bool(
        re.search(
            r"\b(recherche|cherche|google|youtube|video|documentation|source|sources)\b",
            text,
        )
        and re.search(
            r"\b(commence|lance|ouvre|fais|fait|cherche|recherche|google|youtube)\b",
            text,
        )
    )


def _looks_like_homework_draft_choice(text: str) -> bool:
    return bool(
        re.search(
            r"\b(brouillon|redige|rediger|redaction|structure|plan detaille|introduction)\b",
            text,
        )
        or re.search(r"\bcommence\b", text)
    )


def _prefers_youtube(text: str) -> bool:
    return bool(re.search(r"\b(youtube|video|videos)\b", text))


def _homework_search_query(session: TaskSession) -> str:
    instruction = session.slot_value("instruction")
    subject = session.slot_value("subject")
    level = session.slot_value("level")

    parts: list[str] = []
    if instruction is not None:
        parts.append(_clean_homework_query_part(instruction))
    else:
        parts.append(_clean_homework_query_part(session.original_request))

    _append_query_context(parts, subject)
    _append_query_context(parts, level)

    query = re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip(" .")
    return query[:160].rstrip(" .") or "devoir scolaire"


def _append_query_context(parts: list[str], value: str | None) -> None:
    if value is None:
        return
    folded = _fold(" ".join(parts))
    if _fold(value) not in folded:
        parts.append(value)


def _clean_homework_query_part(text: str) -> str:
    cleaned = re.sub(r"\bconsigne\s*:?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(jarvis|aide-moi|aide moi|j'ai un devoir a faire)\b",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .,!?:;")


def _homework_draft_steps(session: TaskSession) -> tuple[str, ...]:
    instruction = session.slot_value("instruction") or "la consigne exacte"
    subject = session.slot_value("subject") or "la matiere"
    level = session.slot_value("level")
    deadline = session.slot_value("deadline")
    level_text = f"niveau {level}" if level is not None else "au bon niveau"
    deadline_text = f"avant {deadline}" if deadline is not None else "avant la date limite"
    return (
        f"Reformuler la consigne en une phrase claire: {instruction}.",
        f"Lister les idees ou notions utiles en {subject}, avec un langage {level_text}.",
        f"Rediger une premiere reponse courte, puis relire {deadline_text}.",
    )


def _homework_draft_message(session: TaskSession, steps: tuple[str, ...]) -> str:
    subject = session.slot_value("subject")
    subject_text = f" pour {subject}" if subject is not None else ""
    numbered = " ".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    return (
        f"Je commence par un brouillon structure{subject_text}: {numbered} "
        "Ensuite, donne-moi tes idees ou demande-moi une recherche ciblee."
    )


def _detect_incomplete_request(
    routed: IntentRouted,
    text: str,
) -> _SessionTemplate | None:
    if _looks_like_sensitive_service_request(text):
        return _SessionTemplate(
            kind="sensitive_system",
            summary="Controle de service Windows demande",
            missing_slots=_SENSITIVE_REQUIRED,
            question=(
                "Action sensible. Dis-moi quel service, pourquoi tu veux le modifier, "
                "et je te demanderai une confirmation explicite avant toute action."
            ),
            reason="Demande service Windows incomplete ou sensible",
        )

    if _looks_like_homework_request(text):
        return _SessionTemplate(
            kind="homework",
            summary="Aide a un devoir",
            missing_slots=_HOMEWORK_REQUIRED,
            question=(
                "D'accord. Donne-moi la consigne exacte, la matiere, ton niveau et la "
                "date limite. Si le sujet est deja ouvert a l'ecran, je peux aussi "
                "l'analyser."
            ),
            reason="Demande de devoir incomplete",
        )

    if _looks_like_work_setup_request(text):
        return _SessionTemplate(
            kind="work_setup",
            summary="Preparation d'un environnement de travail",
            missing_slots=_WORK_SETUP_REQUIRED,
            question=(
                "Je peux preparer un environnement de travail. Tu veux travailler sur "
                "quoi : code, devoir, documents, recherche web ou organisation ?"
            ),
            reason="Routine de travail incomplete",
        )

    if _looks_like_coding_project_request(text):
        return _SessionTemplate(
            kind="coding_project",
            summary="Aide a un projet de code",
            missing_slots=_CODING_REQUIRED,
            question=(
                "Je peux t'aider. Quel type d'application veux-tu creer : web, desktop, "
                "bot, script Python ou autre ?"
            ),
            reason="Demande de projet code incomplete",
        )

    if _looks_like_vague_video_request(routed, text):
        return _SessionTemplate(
            kind="web_video",
            summary="Recherche de video YouTube",
            missing_slots=_VIDEO_REQUIRED,
            question="Sur quel sujet veux-tu que je cherche une video YouTube ?",
            reason="Sujet de recherche video manquant",
        )

    if _looks_like_generic_project_request(text):
        return _SessionTemplate(
            kind="general",
            summary="Aide a un projet",
            missing_slots=("goal",),
            question="Quel est l'objectif du projet, et qu'est-ce que tu veux obtenir a la fin ?",
            reason="Projet general incomplet",
        )

    return None


def _looks_like_sensitive_service_request(text: str) -> bool:
    return bool(
        re.search(r"\b(desactive|arrete|stop|modifie|change)\b.*\b(service|windows)\b", text)
        or re.search(r"\bservice\s+windows\b", text)
    )


def _looks_like_homework_request(text: str) -> bool:
    return bool(
        re.search(r"\b(devoir|exercice|expose|redaction|dissertation)\b", text)
        and re.search(r"\b(aide|faire|j'ai|je dois|organise|prepare)\b", text)
    )


def _looks_like_work_setup_request(text: str) -> bool:
    return bool(
        re.search(r"\b(ouvre|prepare|organise)\b", text)
        and re.search(r"\b(ce qu'il faut|environnement|espace|routine|travail|bosser)\b", text)
    )


def _looks_like_routine_request(text: str) -> bool:
    return bool(
        re.search(
            r"\b(mode travail|mode devoir|mode code|mode recherche|mode concentration|routine|au boulot)\b",
            text,
        )
    )


def _looks_like_coding_project_request(text: str) -> bool:
    return bool(
        re.search(r"\b(aide|idee|projet|creer|coder|developper)\b", text)
        and re.search(r"\b(app|application|code|coder|script|bot)\b", text)
        and not _extract_coding_slots(text)
    )


def _looks_like_vague_video_request(routed: IntentRouted, text: str) -> bool:
    if "video" not in text:
        return False
    if "youtube" in text and _extract_video_slots(text):
        return False
    return bool(
        routed.intent == "unknown"
        or re.search(r"\b(ca|cela|ce sujet|ce truc|la dessus|l[a'] dessus)\b", text)
        or not _extract_video_slots(text)
    )


def _looks_like_generic_project_request(text: str) -> bool:
    return bool(
        re.search(r"\b(aide|organise|prepare)\b", text) and re.search(r"\b(projet|travail)\b", text)
    )


def _extract_homework_slots(raw_text: str, text: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    subject = _extract_subject(text)
    if subject is not None:
        slots["subject"] = subject

    level = _extract_level(text)
    if level is not None:
        slots["level"] = level

    deadline = _extract_deadline(text)
    if deadline is not None:
        slots["deadline"] = deadline

    instruction = _extract_instruction(raw_text, text)
    if instruction is not None:
        slots["instruction"] = instruction

    return slots


def _extract_coding_slots(text: str) -> dict[str, str]:
    if re.search(r"\b(web|site)\b", text):
        return {"app_type": "web"}
    if re.search(r"\b(desktop|bureau)\b", text):
        return {"app_type": "desktop"}
    if re.search(r"\b(bot|assistant)\b", text):
        return {"app_type": "bot"}
    if re.search(r"\b(script|python)\b", text):
        return {"app_type": "script Python"}
    if re.search(r"\b(autre|outil)\b", text):
        return {"app_type": "autre"}
    return {}


def _extract_work_setup_slots(text: str) -> dict[str, str]:
    for work_type in ("code", "devoir", "documents", "recherche web", "organisation"):
        if work_type in text:
            return {"work_type": work_type}
    if "recherche" in text:
        return {"work_type": "recherche web"}
    return {}


def _extract_video_slots(text: str) -> dict[str, str]:
    topic = _topic_after_marker(text)
    if topic is None and _looks_like_short_topic(text):
        topic = text
    if topic is None:
        return {}
    return {"topic": topic}


def _extract_sensitive_slots(text: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    service_match = re.search(r"\bservice\s+(?P<service>[a-z0-9 ._-]{2,})", text)
    if service_match is not None:
        slots["service_name"] = service_match.group("service").strip(" .")
    reason_match = re.search(r"\b(?:parce que|car|pour)\s+(?P<reason>.+)$", text)
    if reason_match is not None:
        slots["reason"] = reason_match.group("reason").strip(" .")
    return slots


def _extract_subject(text: str) -> str | None:
    subjects = (
        "mathematiques",
        "maths",
        "francais",
        "histoire",
        "geographie",
        "anglais",
        "physique",
        "chimie",
        "svt",
        "philosophie",
        "nsi",
        "ses",
    )
    for subject in subjects:
        if re.search(rf"\b{re.escape(subject)}\b", text):
            return subject
    return None


def _extract_level(text: str) -> str | None:
    match = re.search(
        r"\b(6e|5e|4e|3e|seconde|premiere|terminale|college|lycee|licence|master)\b",
        text,
    )
    if match is None:
        return None
    return match.group(1)


def _extract_deadline(text: str) -> str | None:
    relative_match = re.search(r"\b(demain|ce soir|aujourd'hui)\b", text)
    if relative_match is not None:
        return relative_match.group(1)
    match = re.search(
        r"\b(?:pour|avant|date limite|deadline)\s+(?P<deadline>[a-z0-9 /-]{2,30})",
        text,
    )
    if match is None:
        return None
    return match.group("deadline").strip()


def _extract_instruction(raw_text: str, text: str) -> str | None:
    match = re.search(r"\bconsigne\s*:?\s*(?P<instruction>.+)$", raw_text, flags=re.IGNORECASE)
    if match is not None:
        return match.group("instruction").strip(" .")
    if re.search(r"\b(sur|theme|sujet|chapitre|redaction|dissertation|exercice)\b", text):
        cleaned = raw_text.strip(" .")
        if len(cleaned) >= 12:
            return cleaned
    return None


def _topic_after_marker(text: str) -> str | None:
    match = re.search(r"\b(?:sur|a propos de|concernant)\s+(?P<topic>.+)$", text)
    if match is None:
        return None
    topic = match.group("topic")
    topic = re.sub(r"\b(youtube|google|video|videos?)\b", "", topic)
    topic = topic.strip(" .,!?:;")
    if not topic or topic in {"ca", "cela", "ce sujet", "ce truc", "la dessus"}:
        return None
    return topic


def _looks_like_short_topic(text: str) -> bool:
    return 2 <= len(text) <= 80 and not re.search(
        r"\b(cherche|recherche|ouvre|video|youtube|google|devoir|projet)\b",
        text,
    )


def _missing_slots(slots: tuple[TaskSlot, ...], required: tuple[str, ...]) -> tuple[str, ...]:
    known = {slot.name for slot in slots}
    return tuple(slot for slot in required if slot not in known)


def _slot_value(slots: tuple[TaskSlot, ...], name: str) -> str | None:
    for slot in slots:
        if slot.name == name:
            return slot.value
    return None


def _homework_missing_question(missing: tuple[str, ...]) -> str:
    labels = {
        "instruction": "la consigne exacte",
        "subject": "la matiere",
        "level": "ton niveau",
        "deadline": "la date limite",
    }
    readable = ", ".join(labels[item] for item in missing)
    return f"Il me manque encore {readable}. Donne-moi ces elements et je te fais un plan."


def _session_event(
    session: TaskSession,
    *,
    reason: str,
    now: float,
) -> TaskSessionStateChanged:
    return TaskSessionStateChanged(timestamp=now, session=session, reason=reason)


def _utterance(
    text: str,
    *,
    reason: str,
    now: float,
    priority: AssistantFeedbackPriority = "info",
) -> AssistantUtterance:
    return AssistantUtterance(
        timestamp=now,
        text=text,
        source="dialogue",
        priority=priority,
        reason=reason,
    )


def _is_cancel_request(text: str) -> bool:
    return bool(
        re.search(
            r"^(annule|stop|reset|recommence|oublie)"
            r"(?:\s+(la|cette|ma)\s+(tache|session|demande))?$",
            text,
        )
        or re.search(r"\b(annule|oublie)\b.*\b(tache|session|demande)\b", text)
    )


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold().replace("\u2019", "'"))
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", stripped).strip(" .!?")
