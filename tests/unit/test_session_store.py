"""Tests unitaires pour brain/session_store.py.

Couvre :
- save/load round-trip normal ;
- expiration TTL ;
- sessions terminales ignorées au save et au load ;
- fichier corrompu ;
- mode disabled (no-op complet) ;
- clear ;
- écriture atomique (pas de corruption si dossier absent).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.session_store import SessionSnapshot, SessionStore
from brain.task_session import TaskSession, TaskSlot
from config.schema import SessionMemoryConfig


def _make_session(
    *,
    status: str = "waiting_for_user",
    kind: str = "homework",
    slots: tuple[TaskSlot, ...] = (),
    now: float = 1000.0,
) -> TaskSession:
    """Fabrique une session de test minimale."""
    return TaskSession(
        session_id=f"task-{int(now * 1000)}",
        kind=kind,
        status=status,
        original_request="aide-moi pour un devoir",
        summary="Aide a un devoir",
        slots=slots,
        missing_slots=("instruction", "subject"),
        created_at=now,
        updated_at=now,
        last_question="Donne-moi la consigne exacte.",
    )


def _make_store(
    tmp_path: Path,
    *,
    enabled: bool = True,
    expiry_hours: float = 4.0,
    clock: float = 1000.0,
) -> SessionStore:
    """Construit un store pointant sur un dossier temporaire."""
    config = SessionMemoryConfig(
        enabled=enabled,
        directory=str(tmp_path / "sessions"),
        expiry_hours=expiry_hours,
    )
    return SessionStore(config, clock=lambda: clock)


class TestSaveLoad:
    """Round-trip save → load."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, clock=1000.0)
        session = _make_session(now=1000.0)

        assert store.save(session) is True
        loaded = store.load()

        assert loaded is not None
        assert loaded.session_id == session.session_id
        assert loaded.kind == session.kind
        assert loaded.status == session.status
        assert loaded.missing_slots == session.missing_slots
        assert loaded.last_question == session.last_question

    def test_save_with_slots_preserves_data(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, clock=1000.0)
        session = _make_session(
            now=1000.0,
            slots=(
                TaskSlot(name="subject", value="maths"),
                TaskSlot(name="level", value="terminale"),
            ),
        )

        store.save(session)
        loaded = store.load()

        assert loaded is not None
        assert loaded.slot_value("subject") == "maths"
        assert loaded.slot_value("level") == "terminale"

    def test_load_returns_none_when_no_file(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, clock=1000.0)
        assert store.load() is None

    def test_successive_saves_overwrite(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, clock=1000.0)
        session_1 = _make_session(now=1000.0)
        session_2 = _make_session(now=2000.0, kind="coding_project")

        store.save(session_1)
        store.save(session_2)

        loaded = store.load()
        assert loaded is not None
        assert loaded.kind == "coding_project"


class TestExpiry:
    """Vérification du TTL."""

    def test_expired_session_returns_none(self, tmp_path: Path) -> None:
        # Sauvegarde à t=1000
        store_save = _make_store(tmp_path, clock=1000.0, expiry_hours=1.0)
        session = _make_session(now=1000.0)
        store_save.save(session)

        # Chargement à t=1000 + 2h = 3600*2 secondes plus tard
        store_load = _make_store(tmp_path, clock=1000.0 + 7200.0, expiry_hours=1.0)
        assert store_load.load() is None

    def test_fresh_session_loads_fine(self, tmp_path: Path) -> None:
        store_save = _make_store(tmp_path, clock=1000.0, expiry_hours=4.0)
        session = _make_session(now=1000.0)
        store_save.save(session)

        # 30 min plus tard
        store_load = _make_store(tmp_path, clock=1000.0 + 1800.0, expiry_hours=4.0)
        loaded = store_load.load()
        assert loaded is not None
        assert loaded.session_id == session.session_id

    def test_expired_session_is_cleared(self, tmp_path: Path) -> None:
        store_save = _make_store(tmp_path, clock=1000.0, expiry_hours=0.5)
        session = _make_session(now=1000.0)
        store_save.save(session)

        store_load = _make_store(tmp_path, clock=1000.0 + 7200.0, expiry_hours=0.5)
        store_load.load()

        # Le fichier doit avoir été supprimé
        assert not store_load.session_path.exists()


class TestTerminalSessions:
    """Les sessions terminées/annulées ne sont pas persistées."""

    def test_cancelled_session_not_saved(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, clock=1000.0)
        session = _make_session(now=1000.0, status="cancelled")
        assert store.save(session) is False
        assert store.load() is None

    def test_completed_session_not_saved(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, clock=1000.0)
        session = _make_session(now=1000.0, status="completed")
        assert store.save(session) is False
        assert store.load() is None

    def test_terminal_save_clears_existing(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, clock=1000.0)
        # D'abord, sauvegarder une session active
        active = _make_session(now=1000.0, status="waiting_for_user")
        store.save(active)
        assert store.load() is not None

        # Ensuite, sauvegarder la session comme annulée → supprime le fichier
        cancelled = active.with_update(now=1001.0, status="cancelled")
        store.save(cancelled)
        assert store.load() is None

    def test_terminal_on_load_is_cleared(self, tmp_path: Path) -> None:
        """Si le fichier contient une session terminal (manuellement), load la rejette."""
        store = _make_store(tmp_path, clock=1000.0)
        session = _make_session(now=1000.0, status="waiting_for_user")
        # Sauvegarder manuellement avec statut completed dans le snapshot
        snapshot = SessionSnapshot(
            saved_at=1000.0,
            session=session.model_copy(update={"status": "completed"}),
        )
        (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
        store.session_path.write_text(
            snapshot.model_dump_json(indent=2), encoding="utf-8"
        )
        assert store.load() is None


class TestCorruptFile:
    """Fichier corrompu ou JSON invalide."""

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, clock=1000.0)
        (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
        store.session_path.write_text("{not valid json!!!", encoding="utf-8")
        assert store.load() is None

    def test_corrupt_json_clears_file(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, clock=1000.0)
        (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
        store.session_path.write_text("garbage", encoding="utf-8")
        store.load()
        assert not store.session_path.exists()


class TestDisabled:
    """Quand enabled=False, tout est no-op."""

    def test_save_returns_false_when_disabled(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, enabled=False)
        session = _make_session(now=1000.0)
        assert store.save(session) is False

    def test_load_returns_none_when_disabled(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, enabled=False)
        assert store.load() is None

    def test_no_file_written_when_disabled(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, enabled=False)
        session = _make_session(now=1000.0)
        store.save(session)
        assert not (tmp_path / "sessions").exists()


class TestClear:
    """Suppression explicite."""

    def test_clear_removes_file(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, clock=1000.0)
        session = _make_session(now=1000.0)
        store.save(session)
        assert store.session_path.exists()
        store.clear()
        assert not store.session_path.exists()

    def test_clear_noop_when_no_file(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, clock=1000.0)
        store.clear()  # Ne doit pas lever d'exception


class TestSnapshotModel:
    """Validation du modèle SessionSnapshot."""

    def test_snapshot_is_frozen(self) -> None:
        session = _make_session(now=1000.0)
        snapshot = SessionSnapshot(saved_at=1000.0, session=session)
        with pytest.raises(Exception):
            snapshot.saved_at = 2000.0  # type: ignore[misc]
