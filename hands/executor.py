"""Execution GUI en dry-run pour les decisions vision.

Iteration 5B : ce module ne pilote pas encore `pyautogui`. Il transforme une
`VisionDecision` en actions GUI planifiees, applique les garde-fous de securite,
et journalise le resultat. Les clics reels arriveront dans un lot separe.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from brain.vision_contracts import VisionAction, VisionDecision
from config.schema import SafetyConfig, SafetyMode
from hands.screenshot import ScreenshotFrame
from observability.logger import get_logger

log = get_logger(__name__)

HandsExecutionStatus = Literal["completed", "observe", "dry_run", "blocked"]


class PlannedGuiAction(BaseModel):
    """Action GUI convertie dans l'espace ecran natif."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    x: int | None = None
    y: int | None = None
    text: str | None = None
    keys: tuple[str, ...] = ()
    amount: int | None = None
    duration_ms: int | None = None
    destructive: bool = False


class HandsExecutionReport(BaseModel):
    """Rapport d'execution ou de non-execution Hands."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: HandsExecutionStatus
    mode: SafetyMode
    actions: tuple[PlannedGuiAction, ...]
    executed: bool
    requires_human: bool
    reason: str


class DryRunHandsExecutor:
    """Planifie les actions GUI sans les executer reellement."""

    def __init__(self, safety: SafetyConfig) -> None:
        self._safety = safety

    def execute(
        self,
        decision: VisionDecision,
        *,
        frame: ScreenshotFrame | None = None,
    ) -> HandsExecutionReport:
        """Retourne ce qui serait execute par Hands.

        Les modes `observe` et `dry_run` n'executent jamais d'action. Les modes
        `assisted` et `autonomous` restent volontairement bloques en 5B : on veut
        d'abord stabiliser le plan et les tests avant de brancher `pyautogui`.
        """
        planned_actions = tuple(
            _plan_action(
                action,
                frame=frame,
                destructive_patterns=self._safety.allowlist_destructive_patterns,
            )
            for action in decision.actions
        )

        report = self._build_report(decision=decision, actions=planned_actions)
        log.info(
            "hands_execution_report",
            mode=report.mode,
            status=report.status,
            actions=len(report.actions),
            executed=report.executed,
            requires_human=report.requires_human,
            reason=report.reason,
        )
        return report

    def _build_report(
        self,
        *,
        decision: VisionDecision,
        actions: tuple[PlannedGuiAction, ...],
    ) -> HandsExecutionReport:
        if decision.task_complete:
            return HandsExecutionReport(
                status="completed",
                mode=self._safety.mode,
                actions=(),
                executed=False,
                requires_human=False,
                reason="Tache deja terminee selon la vision",
            )

        if decision.requires_human:
            return _blocked_report(
                mode=self._safety.mode,
                actions=actions,
                reason="Confirmation humaine requise par la vision",
            )

        if any(action.destructive for action in actions):
            return _blocked_report(
                mode=self._safety.mode,
                actions=actions,
                reason="Action potentiellement destructive detectee",
            )

        if self._safety.mode == "observe":
            return HandsExecutionReport(
                status="observe",
                mode=self._safety.mode,
                actions=actions,
                executed=False,
                requires_human=False,
                reason="Mode observe: aucune action GUI planifiee n'est executee",
            )

        if self._safety.mode == "dry_run":
            return HandsExecutionReport(
                status="dry_run",
                mode=self._safety.mode,
                actions=actions,
                executed=False,
                requires_human=False,
                reason="Mode dry_run: actions journalisees sans execution",
            )

        return _blocked_report(
            mode=self._safety.mode,
            actions=actions,
            reason="Actuators reels non branches en iteration 5B",
        )


def _blocked_report(
    *,
    mode: SafetyMode,
    actions: tuple[PlannedGuiAction, ...],
    reason: str,
) -> HandsExecutionReport:
    return HandsExecutionReport(
        status="blocked",
        mode=mode,
        actions=actions,
        executed=False,
        requires_human=True,
        reason=reason,
    )


def _plan_action(
    action: VisionAction,
    *,
    frame: ScreenshotFrame | None,
    destructive_patterns: list[str],
) -> PlannedGuiAction:
    x, y = _native_point(action, frame=frame)
    return PlannedGuiAction(
        type=action.type,
        x=x,
        y=y,
        text=action.text,
        keys=tuple(action.keys or ()),
        amount=action.amount,
        duration_ms=action.duration_ms,
        destructive=action.contains_destructive_intent()
        or _matches_destructive_pattern(action, destructive_patterns),
    )


def _native_point(
    action: VisionAction,
    *,
    frame: ScreenshotFrame | None,
) -> tuple[int | None, int | None]:
    if action.x is None or action.y is None:
        return None, None
    if frame is None:
        return action.x, action.y
    return (
        frame.left + round(action.x * frame.scale_x),
        frame.top + round(action.y * frame.scale_y),
    )


def _matches_destructive_pattern(action: VisionAction, patterns: list[str]) -> bool:
    if not patterns:
        return False
    haystack_parts: list[str] = [action.type]
    if action.text is not None:
        haystack_parts.append(action.text)
    if action.keys is not None:
        haystack_parts.extend(action.keys)
    haystack = " ".join(haystack_parts).casefold()
    return any(pattern.casefold() in haystack for pattern in patterns)
