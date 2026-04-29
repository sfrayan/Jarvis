"""Gestionnaire de confirmations explicites pour actions sensibles.

Le `ConfirmationManager` :
- enregistre une action bloquee en attente de confirmation ;
- verifie si une phrase utilisateur est une confirmation ou un rejet ;
- expire les demandes apres un TTL configurable (defaut 60s) ;
- produit un `ConfirmationResponse` pour relayer la decision.

Il reste pur (pas d'EventBus, pas de GUI). Le service qui l'utilise
(`DialogueManager` ou `HandsPipelineService`) publie les evenements.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from brain.events import (
    ConfirmationResponse,
    ConfirmationVerdict,
    PendingConfirmation,
)
from hands.executor import HandsExecutionReport, PlannedGuiAction
from observability.logger import get_logger

log = get_logger(__name__)

_DEFAULT_TTL_S: float = 60.0


class PendingAction(BaseModel):
    """Action en attente de confirmation stockee en RAM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    confirmation_id: str = Field(..., min_length=1)
    report: HandsExecutionReport
    question: str = Field(..., min_length=1)
    created_at: float = Field(..., ge=0.0)
    expires_at: float = Field(..., ge=0.0)


class ConfirmationManager:
    """Gere une seule action pendante a la fois (FIFO simple)."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        ttl_s: float = _DEFAULT_TTL_S,
    ) -> None:
        self._clock = clock or time.time
        self._ttl_s = ttl_s
        self._pending: PendingAction | None = None

    @property
    def pending(self) -> PendingAction | None:
        """Retourne l'action pendante si elle n'est pas expiree."""
        if self._pending is None:
            return None
        if self._clock() > self._pending.expires_at:
            log.info(
                "confirmation_expired",
                confirmation_id=self._pending.confirmation_id,
            )
            self._pending = None
            return None
        return self._pending

    @property
    def has_pending(self) -> bool:
        """Indique si une confirmation est en attente."""
        return self.pending is not None

    def request_confirmation(
        self,
        report: HandsExecutionReport,
    ) -> PendingConfirmation:
        """Enregistre une action bloquee et retourne l'evenement de demande."""
        now = self._clock()
        action = _primary_action(report.actions)
        action_type = action.type if action is not None else "unknown"
        action_target = action.text if action is not None else None

        confirmation_id = f"confirm-{int(now * 1000)}"
        question = _confirmation_question(action_type, action_target, report.reason)

        pending = PendingAction(
            confirmation_id=confirmation_id,
            report=report,
            question=question,
            created_at=now,
            expires_at=now + self._ttl_s,
        )
        self._pending = pending

        log.info(
            "confirmation_requested",
            confirmation_id=confirmation_id,
            action_type=action_type,
            action_target=action_target,
            expires_in_s=self._ttl_s,
        )

        return PendingConfirmation(
            confirmation_id=confirmation_id,
            timestamp=now,
            action_type=action_type,
            action_target=action_target,
            question=question,
            reason=report.reason,
            expires_at=pending.expires_at,
        )

    def handle_user_reply(self, text: str) -> ConfirmationResponse | None:
        """Traite la reponse utilisateur.

        Retourne un ConfirmationResponse si la phrase est une confirmation ou
        un rejet. Retourne None si ce n'est ni l'un ni l'autre, ou s'il n'y a
        rien en attente.
        """
        current = self.pending
        if current is None:
            return None

        folded = _fold(text)
        verdict = _classify_reply(folded)
        if verdict is None:
            return None

        now = self._clock()
        self._pending = None

        log.info(
            "confirmation_verdict",
            confirmation_id=current.confirmation_id,
            verdict=verdict,
        )

        return ConfirmationResponse(
            confirmation_id=current.confirmation_id,
            timestamp=now,
            verdict=verdict,
            reason=f"Utilisateur a {'confirme' if verdict == 'confirmed' else 'refuse'} l'action",
        )

    def expire_if_needed(self) -> ConfirmationResponse | None:
        """Force l'expiration si le TTL est depasse. Retourne l'evenement."""
        if self._pending is None:
            return None
        now = self._clock()
        if now <= self._pending.expires_at:
            return None

        expired_id = self._pending.confirmation_id
        self._pending = None
        log.info("confirmation_expired_explicit", confirmation_id=expired_id)

        return ConfirmationResponse(
            confirmation_id=expired_id,
            timestamp=now,
            verdict="expired",
            reason="Delai de confirmation depasse",
        )

    def clear(self) -> None:
        """Annule toute confirmation pendante."""
        self._pending = None


def _classify_reply(text: str) -> ConfirmationVerdict | None:
    """Classe une phrase en confirmation, rejet, ou None (pas de rapport)."""
    if re.search(
        r"\b(oui|ok|confirme|confirmer|vas-y|vas y|fais-le|fais le|go|d'accord|"
        r"c'est bon|execute|executer|valide|valider|accepte|accepter)\b",
        text,
    ):
        return "confirmed"

    if re.search(
        r"\b(non|annule|annuler|refuse|refuser|stop|arrete|laisse tomber|"
        r"oublie|pas maintenant|ne fais rien|ne fait rien)\b",
        text,
    ):
        return "rejected"

    return None


def _confirmation_question(
    action_type: str,
    action_target: str | None,
    reason: str,
) -> str:
    """Construit la question de confirmation adaptee."""
    target_text = action_target or "cette action"

    if action_type == "close_app":
        return f"Tu veux que je ferme {target_text} ? Dis oui pour confirmer."
    if action_type == "system_command":
        return (
            f"Attention, action systeme sensible: {target_text}. "
            f"Dis oui explicitement pour que j'execute."
        )
    if action_type in {"launch_app", "system_tool"}:
        return f"Je peux ouvrir {target_text}. Tu confirmes ?"

    return f"Action sensible detectee: {reason}. Tu veux que je continue ? Dis oui ou non."


def _primary_action(actions: tuple[PlannedGuiAction, ...]) -> PlannedGuiAction | None:
    """Premiere action du rapport, si elle existe."""
    return actions[0] if actions else None


def _fold(text: str) -> str:
    """Normalise pour la detection de confirmation/rejet."""
    decomposed = unicodedata.normalize("NFKD", text.casefold().replace("\u2019", "'"))
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", stripped).strip(" .!?")
