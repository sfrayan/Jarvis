"""Plans de routines locales sures.

Une routine n'execute rien directement. Elle decrit un petit plan de travail et
les actions locales ou navigateur que Jarvis pourra proposer ensuite, une par
une, via les garde-fous existants.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RoutineKind = Literal["work", "homework", "code", "research", "focus"]
SuggestedActionKind = Literal["local", "browser", "dialogue"]


class RoutineActionSuggestion(BaseModel):
    """Action non executee suggeree par une routine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: SuggestedActionKind
    label: str = Field(..., min_length=1)
    command: str | None = Field(default=None, min_length=1)
    requires_confirmation: bool = False


class RoutinePlan(BaseModel):
    """Plan de routine court, sans effet de bord."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: RoutineKind
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    steps: tuple[str, ...] = Field(..., min_length=1)
    suggestions: tuple[RoutineActionSuggestion, ...] = ()
    next_question: str = Field(..., min_length=1)


def plan_routine(text: str) -> RoutinePlan | None:
    """Retourne un plan de routine si `text` correspond a un mode connu."""
    folded = _fold(text)
    kind = _infer_routine_kind(folded)
    if kind is None:
        return None
    return _ROUTINES[kind]


def _infer_routine_kind(text: str) -> RoutineKind | None:
    if re.search(r"\b(devoir|exercice|revision|revisions)\b", text):
        return "homework"
    if re.search(r"\b(code|coder|dev|developpement|app|application|script|python)\b", text):
        return "code"
    if re.search(r"\b(recherche|chercher|web|google|youtube|documentation|veille)\b", text):
        return "research"
    if re.search(r"\b(focus|concentration|profond|profondement)\b", text):
        return "focus"
    if re.search(r"\b(mode travail|au boulot|travail|bosser|routine)\b", text):
        return "work"
    return None


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold().replace("\u2019", "'"))
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", stripped).strip(" .!?")


_ROUTINES: dict[RoutineKind, RoutinePlan] = {
    "work": RoutinePlan(
        kind="work",
        title="Mode travail",
        summary="Preparation generale pour travailler sans lancer d'action automatique",
        steps=(
            "Clarifier le type de travail et le resultat attendu.",
            "Proposer uniquement les outils utiles.",
            "Ouvrir les outils un par un apres accord ou en mode autorise.",
        ),
        suggestions=(
            RoutineActionSuggestion(
                kind="dialogue",
                label="Choisir le type de travail",
                command="code, devoir, documents, recherche web ou organisation",
            ),
        ),
        next_question="Tu veux travailler sur quoi : code, devoir, documents, recherche web ou organisation ?",
    ),
    "homework": RoutinePlan(
        kind="homework",
        title="Mode devoir",
        summary="Preparation d'une session de devoir avec consigne, niveau et date limite",
        steps=(
            "Recueillir la consigne exacte, la matiere, le niveau et la date limite.",
            "Proposer un plan court avant toute recherche.",
            "Ouvrir une recherche web seulement si tu le demandes.",
        ),
        suggestions=(
            RoutineActionSuggestion(
                kind="dialogue",
                label="Demander les informations du devoir",
                command="consigne, matiere, niveau, date limite",
            ),
        ),
        next_question="Donne-moi la consigne, la matiere, ton niveau et la date limite.",
    ),
    "code": RoutinePlan(
        kind="code",
        title="Mode code",
        summary="Preparation prudente d'un environnement de developpement",
        steps=(
            "Identifier le type de projet et son objectif.",
            "Proposer les outils locaux utiles comme VS Code ou un dossier projet.",
            "Attendre ton accord avant d'ouvrir ou de creer quoi que ce soit.",
        ),
        suggestions=(
            RoutineActionSuggestion(
                kind="local",
                label="Ouvrir VS Code",
                command="ouvre VS Code",
            ),
            RoutineActionSuggestion(
                kind="dialogue",
                label="Choisir le type de projet",
                command="web, desktop, bot, script Python ou autre",
            ),
        ),
        next_question="Quel type de projet veux-tu coder : web, desktop, bot, script Python ou autre ?",
    ),
    "research": RoutinePlan(
        kind="research",
        title="Mode recherche",
        summary="Preparation d'une recherche web ou YouTube ciblee",
        steps=(
            "Preciser le sujet et le niveau de detail voulu.",
            "Choisir Google, YouTube ou une recherche plus ciblee.",
            "Ouvrir la recherche seulement quand le sujet est clair.",
        ),
        suggestions=(
            RoutineActionSuggestion(
                kind="browser",
                label="Ouvrir une recherche Google",
                command="cherche sur Google <sujet>",
            ),
            RoutineActionSuggestion(
                kind="browser",
                label="Ouvrir une recherche YouTube",
                command="cherche sur YouTube <sujet>",
            ),
        ),
        next_question="Quel sujet veux-tu chercher, et tu preferes Google ou YouTube ?",
    ),
    "focus": RoutinePlan(
        kind="focus",
        title="Mode concentration",
        summary="Preparation calme d'une session de concentration",
        steps=(
            "Definir l'objectif unique de la session.",
            "Limiter les ouvertures d'applications au strict necessaire.",
            "Garder une prochaine action simple et verifiable.",
        ),
        suggestions=(
            RoutineActionSuggestion(
                kind="dialogue",
                label="Definir l'objectif de focus",
                command="objectif unique de la session",
            ),
        ),
        next_question="Quel est l'objectif unique de cette session ?",
    ),
}
