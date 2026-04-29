"""Tests unitaires de la sauvegarde controlee des brouillons 5N."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.draft_storage import DraftStorage, DraftStorageService
from brain.events import AssistantDraft, DraftSaveReport
from config.schema import DraftStorageConfig, SafetyConfig, SafetyMode
from core.event_bus import EventBus

pytestmark = pytest.mark.unit


def _draft() -> AssistantDraft:
    return AssistantDraft(
        timestamp=1.0,
        session_id="task-123",
        kind="homework",
        title="Brouillon de maths: fonctions",
        context="Consigne: exercice sur les fonctions. Matiere: maths.",
        sections=("Comprendre", "Rediger"),
        body="Premiere version du brouillon.",
        next_steps=("Relire", "Completer les calculs"),
        reason="test",
    )


def _storage(
    tmp_path: Path,
    *,
    enabled: bool = True,
    mode: SafetyMode = "dry_run",
    directory: str = "drafts",
) -> DraftStorage:
    return DraftStorage(
        DraftStorageConfig(enabled=enabled, directory=directory),
        SafetyConfig(mode=mode),
        base_dir=tmp_path,
        clock=lambda: 10.0,
    )


def test_draft_storage_defaults_to_disabled() -> None:
    config = DraftStorageConfig()

    assert config.enabled is False
    assert config.directory == "data/drafts"


def test_disabled_storage_does_not_write_file(tmp_path: Path) -> None:
    storage = _storage(tmp_path, enabled=False, mode="assisted")

    report = storage.save(_draft())

    assert report.status == "disabled"
    assert report.saved is False
    assert report.path is None
    assert list(tmp_path.iterdir()) == []


def test_dry_run_plans_path_without_writing(tmp_path: Path) -> None:
    storage = _storage(tmp_path, mode="dry_run")

    report = storage.save(_draft())

    assert report.status == "dry_run"
    assert report.saved is False
    assert report.path is not None
    assert report.path.endswith(".md")
    assert not Path(report.path).exists()


def test_observe_plans_path_without_writing(tmp_path: Path) -> None:
    storage = _storage(tmp_path, mode="observe")

    report = storage.save(_draft())

    assert report.status == "observe"
    assert report.saved is False
    assert report.path is not None
    assert not Path(report.path).exists()


def test_assisted_mode_writes_markdown_inside_configured_directory(tmp_path: Path) -> None:
    storage = _storage(tmp_path, mode="assisted")

    report = storage.save(_draft())

    assert report.status == "saved"
    assert report.saved is True
    assert report.path is not None
    saved = Path(report.path)
    assert saved.is_relative_to(tmp_path)
    assert saved.read_text(encoding="utf-8").startswith("# Brouillon de maths")
    assert "Premiere version du brouillon." in saved.read_text(encoding="utf-8")


def test_existing_file_is_not_overwritten(tmp_path: Path) -> None:
    storage = _storage(tmp_path, mode="assisted")

    first = storage.save(_draft())
    second = storage.save(_draft())

    assert first.path is not None
    assert second.path is not None
    assert first.path != second.path
    assert Path(first.path).exists()
    assert Path(second.path).exists()


@pytest.mark.parametrize("directory", ["../outside", "C:/outside"])
def test_out_of_scope_directory_is_blocked(tmp_path: Path, directory: str) -> None:
    storage = _storage(tmp_path, mode="assisted", directory=directory)

    report = storage.save(_draft())

    assert report.status == "blocked"
    assert report.saved is False
    assert report.requires_human is True
    assert report.path is None


@pytest.mark.asyncio
async def test_service_publishes_save_report(tmp_path: Path) -> None:
    bus = EventBus()
    storage = _storage(tmp_path, mode="assisted")
    service = DraftStorageService(event_bus=bus, storage=storage)
    reports: list[DraftSaveReport] = []

    async def report_handler(event: DraftSaveReport) -> None:
        reports.append(event)

    bus.subscribe(DraftSaveReport, report_handler)
    service.start()

    await bus.publish(_draft())

    assert len(reports) == 1
    assert reports[0].status == "saved"
    assert reports[0].path is not None
    assert Path(reports[0].path).exists()


@pytest.mark.asyncio
async def test_service_stop_unsubscribes(tmp_path: Path) -> None:
    bus = EventBus()
    storage = _storage(tmp_path, mode="assisted")
    service = DraftStorageService(event_bus=bus, storage=storage)
    reports: list[DraftSaveReport] = []

    async def report_handler(event: DraftSaveReport) -> None:
        reports.append(event)

    bus.subscribe(DraftSaveReport, report_handler)
    service.start()
    service.stop()

    await bus.publish(_draft())

    assert reports == []
    assert bus.subscriber_count(AssistantDraft) == 0
