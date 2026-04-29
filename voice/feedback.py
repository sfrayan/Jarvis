"""Feedback assistant textuel avant le TTS reel.

Iteration 5G-H : cette couche transforme les rapports Hands en phrases courtes
que Jarvis pourra ensuite envoyer a Piper. Elle reste pure et testable : aucun
son n'est joue ici.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from brain.events import IntentRouted
from hands.executor import HandsExecutionReport, PlannedGuiAction

AssistantFeedbackPriority = Literal["info", "warning", "error"]
AssistantFeedbackSource = Literal["hands", "dialogue"]

_LOCAL_DOMAINS = frozenset({"apps", "folders", "media", "system"})


class AssistantUtterance(BaseModel):
    """Phrase que Jarvis doit dire ou afficher a l'utilisateur."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float = Field(..., ge=0.0)
    text: str = Field(..., min_length=1)
    source: AssistantFeedbackSource = "hands"
    priority: AssistantFeedbackPriority = "info"
    reason: str = Field(..., min_length=1)


def feedback_from_hands_report(report: HandsExecutionReport) -> AssistantUtterance:
    """Construit une phrase utilisateur depuis un rapport Hands."""
    action = _primary_feedback_action(report.actions)

    if report.status == "blocked":
        return _utterance(
            text=_blocked_text(report),
            priority="warning",
            reason=report.reason,
        )

    if report.status == "dry_run":
        return _utterance(
            text=_dry_run_text(action),
            reason=report.reason,
        )

    if report.status == "observe":
        return _utterance(
            text=_observe_text(action),
            reason=report.reason,
        )

    return _utterance(
        text=_completed_text(action, executed=report.executed),
        reason=report.reason,
    )


def feedback_for_unhandled_local_intent(event: IntentRouted) -> AssistantUtterance | None:
    """Retourne une phrase quand une demande locale n'a pas d'action connue."""
    if event.intent != "gui" or event.domain not in _LOCAL_DOMAINS:
        return None

    if event.domain == "apps":
        text = "Je ne trouve pas cette application dans ton inventaire local."
    elif event.domain == "folders":
        text = "Je ne trouve pas ce dossier dans ton inventaire local."
    elif event.domain == "media":
        text = "Je ne sais pas encore piloter cette action média."
    else:
        text = "Je ne sais pas encore exécuter cette action système."

    return _utterance(
        text=text,
        priority="warning",
        reason=f"Action locale non prise en charge: {event.normalized_text}",
    )


def _completed_text(action: PlannedGuiAction | None, *, executed: bool) -> str:
    if action is None:
        return "C'est déjà terminé."
    if _is_browser_action(action):
        return _executed_action_sentence(action) if executed else _planned_action_sentence(action)
    if executed:
        return f"Je l'ai trouvée dans ton PC. {_executed_action_sentence(action)}"
    return f"Action terminée. {_planned_action_sentence(action)}"


def _dry_run_text(action: PlannedGuiAction | None) -> str:
    if action is None:
        return "En mode dry run, je n'exécute rien pour l'instant."
    return f"Je l'ai trouvée dans ton PC. En mode dry run, je n'exécute pas encore: {_action_label(action)}."


def _observe_text(action: PlannedGuiAction | None) -> str:
    if action is None:
        return "Je reste en observation, aucune action n'est exécutée."
    return f"Je vois l'action à faire, mais le mode observe bloque l'exécution: {_action_label(action)}."


def _blocked_text(report: HandsExecutionReport) -> str:
    if report.requires_human:
        return f"J'ai besoin de ta confirmation avant de continuer: {report.reason}."
    return f"Action bloquée: {report.reason}."


def _executed_action_sentence(action: PlannedGuiAction) -> str:
    target = _target_or_default(action)
    if action.type in {"launch_app", "system_tool"}:
        return f"J'ai ouvert {target}."
    if action.type == "open_folder":
        return f"J'ai ouvert le dossier {target}."
    if action.type == "close_app":
        return f"J'ai fermé {target}."
    if action.type == "system_volume":
        return _volume_sentence(action.text)
    if action.type == "media_control":
        return _media_sentence(action.text)
    if action.type == "browser_open_chrome":
        return "J'ai ouvert Chrome."
    if action.type == "browser_new_tab":
        return "J'ai ouvert un nouvel onglet."
    if action.type == "browser_navigate":
        return f"J'ai ouvert la page {target}."
    return f"J'ai exécuté l'action {action.type}."


def _planned_action_sentence(action: PlannedGuiAction) -> str:
    target = _target_or_default(action)
    if action.type in {"launch_app", "system_tool"}:
        return f"Je peux ouvrir {target}."
    if action.type == "open_folder":
        return f"Je peux ouvrir le dossier {target}."
    if action.type == "close_app":
        return f"Je peux fermer {target}."
    if action.type == "system_volume":
        return f"Je peux régler le volume: {target}."
    if action.type == "media_control":
        return f"Je peux envoyer la commande média: {target}."
    if action.type == "browser_open_chrome":
        return "Je peux ouvrir Chrome."
    if action.type == "browser_new_tab":
        return "Je peux ouvrir un nouvel onglet."
    if action.type == "browser_navigate":
        return f"Je peux ouvrir la page {target}."
    return f"Je peux exécuter l'action {action.type}."


def _action_label(action: PlannedGuiAction) -> str:
    target = action.text
    if target is None:
        return action.type
    if action.type == "launch_app":
        return f"ouvrir {target}"
    if action.type == "open_folder":
        return f"ouvrir le dossier {target}"
    if action.type == "close_app":
        return f"fermer {target}"
    if action.type == "browser_open_chrome":
        return "ouvrir Chrome"
    if action.type == "browser_new_tab":
        return "ouvrir un nouvel onglet"
    if action.type == "browser_navigate":
        return f"ouvrir la page {target}"
    return f"{action.type} {target}"


def _target_or_default(action: PlannedGuiAction) -> str:
    return action.text or action.type


def _primary_feedback_action(
    actions: tuple[PlannedGuiAction, ...],
) -> PlannedGuiAction | None:
    if not actions:
        return None
    for action in actions:
        if action.type == "browser_navigate":
            return action
    return actions[0]


def _is_browser_action(action: PlannedGuiAction) -> bool:
    return action.type.startswith("browser_")


def _volume_sentence(target: str | None) -> str:
    if target == "volume_up":
        return "J'ai monté le volume."
    if target == "volume_down":
        return "J'ai baissé le volume."
    return "J'ai coupé ou rétabli le son."


def _media_sentence(target: str | None) -> str:
    if target == "next":
        return "Je passe au morceau suivant."
    if target == "previous":
        return "Je reviens au morceau précédent."
    return "J'ai envoyé la commande lecture pause."


def _utterance(
    *,
    text: str,
    reason: str,
    priority: AssistantFeedbackPriority = "info",
) -> AssistantUtterance:
    return AssistantUtterance(
        timestamp=time.time(),
        text=text,
        priority=priority,
        reason=reason,
    )
