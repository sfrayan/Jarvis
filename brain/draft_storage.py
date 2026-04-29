"""Sauvegarde locale controlee des brouillons assistant.

Le service ecoute `AssistantDraft` et publie un `DraftSaveReport`. Il n'ecrit
un fichier que si la sauvegarde est activee et que le mode de securite autorise
l'execution reelle. En `observe` et `dry_run`, il planifie seulement.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

from brain.events import AssistantDraft, DraftSaveReport, DraftSaveStatus
from config.schema import DraftStorageConfig, SafetyConfig
from core.event_bus import EventBus, SubscriptionHandle
from observability.logger import get_logger

log = get_logger(__name__)


class DraftStorage:
    """Stockage fichier Markdown limite a un dossier configure."""

    def __init__(
        self,
        config: DraftStorageConfig,
        safety: SafetyConfig,
        *,
        base_dir: Path | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._config = config
        self._safety = safety
        self._base_dir = (base_dir or Path.cwd()).resolve()
        self._clock = clock or time.time

    def save(self, draft: AssistantDraft) -> DraftSaveReport:
        """Sauvegarde ou planifie la sauvegarde d'un brouillon."""
        now = self._clock()
        if not self._config.enabled:
            return _report(
                draft,
                timestamp=now,
                status="disabled",
                reason="Sauvegarde des brouillons desactivee",
            )

        target_dir = self._resolve_directory()
        if target_dir is None:
            return _report(
                draft,
                timestamp=now,
                status="blocked",
                requires_human=True,
                reason="Dossier de brouillons hors du scope autorise",
            )

        target_path = _unique_path(target_dir, _draft_filename(draft))
        if self._safety.mode == "observe":
            return _report(
                draft,
                timestamp=now,
                status="observe",
                path=str(target_path),
                reason="Mode observe: brouillon note sans ecriture fichier",
            )

        if self._safety.mode == "dry_run":
            return _report(
                draft,
                timestamp=now,
                status="dry_run",
                path=str(target_path),
                reason="Mode dry_run: brouillon planifie sans ecriture fichier",
            )

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path.write_text(_draft_markdown(draft), encoding="utf-8")
        except OSError as exc:
            return _report(
                draft,
                timestamp=now,
                status="blocked",
                path=str(target_path),
                requires_human=True,
                reason=f"Sauvegarde du brouillon impossible: {type(exc).__name__}",
            )

        return _report(
            draft,
            timestamp=now,
            status="saved",
            path=str(target_path),
            saved=True,
            reason="Brouillon sauvegarde localement",
        )

    def _resolve_directory(self) -> Path | None:
        directory = Path(self._config.directory)
        if directory.is_absolute():
            return None
        target = (self._base_dir / directory).resolve()
        if not target.is_relative_to(self._base_dir):
            return None
        return target


class DraftStorageService:
    """Service reactif : AssistantDraft -> DraftSaveReport."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        storage: DraftStorage,
    ) -> None:
        self._bus = event_bus
        self._storage = storage
        self._subscription: SubscriptionHandle | None = None

    @classmethod
    def create_default(
        cls,
        *,
        event_bus: EventBus,
        config: DraftStorageConfig,
        safety: SafetyConfig,
    ) -> DraftStorageService:
        """Factory utilisee par `main.py`."""
        return cls(
            event_bus=event_bus,
            storage=DraftStorage(config, safety),
        )

    def start(self) -> None:
        """S'abonne aux brouillons. Idempotent."""
        if self._subscription is not None and self._subscription.active:
            return
        self._subscription = self._bus.subscribe(AssistantDraft, self._on_draft)
        log.info("draft_storage_service_started")

    def stop(self) -> None:
        """Retire l'abonnement. Idempotent."""
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None
        log.info("draft_storage_service_stopped")

    async def _on_draft(self, event: AssistantDraft) -> None:
        report = self._storage.save(event)
        await self._bus.publish(report)
        log.info(
            "draft_storage_reported",
            status=report.status,
            saved=report.saved,
            path=report.path,
            reason=report.reason,
        )


def _report(
    draft: AssistantDraft,
    *,
    timestamp: float,
    status: DraftSaveStatus,
    reason: str,
    path: str | None = None,
    saved: bool = False,
    requires_human: bool = False,
) -> DraftSaveReport:
    return DraftSaveReport(
        timestamp=timestamp,
        session_id=draft.session_id,
        status=status,
        path=path,
        saved=saved,
        requires_human=requires_human,
        reason=reason,
    )


def _unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        numbered = directory / f"{stem}-{index}{suffix}"
        if not numbered.exists():
            return numbered
        index += 1


def _draft_filename(draft: AssistantDraft) -> str:
    session = _slug(draft.session_id) or "session"
    title = _slug(draft.title) or "brouillon"
    return f"{session}-{title}.md"


def _draft_markdown(draft: AssistantDraft) -> str:
    sections = "\n".join(f"{index}. {section}" for index, section in enumerate(draft.sections, 1))
    next_steps = "\n".join(f"- {step}" for step in draft.next_steps)
    return "\n".join(
        (
            f"# {draft.title}",
            "",
            "## Contexte",
            draft.context,
            "",
            "## Sections",
            sections,
            "",
            "## Brouillon",
            draft.body,
            "",
            "## Prochaines etapes",
            next_steps or "- Relire et completer le brouillon.",
            "",
        )
    )


def _slug(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char))
    cleaned = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return cleaned[:80].strip("-")
