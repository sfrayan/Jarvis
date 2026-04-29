"""Tests unitaires des sessions de tache 5H."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from brain.task_session import TaskSession, TaskSlot, make_session_id, merge_slots

pytestmark = pytest.mark.unit


def test_make_session_id_is_deterministic_from_timestamp() -> None:
    assert make_session_id(12.345) == "task-12345"


def test_merge_slots_replaces_existing_value_and_sorts_names() -> None:
    slots = (TaskSlot(name="subject", value="maths"),)

    merged = merge_slots(slots, {"deadline": "demain", "subject": "francais"})

    assert [slot.name for slot in merged] == ["deadline", "subject"]
    assert [slot.value for slot in merged] == ["demain", "francais"]


def test_task_session_rejects_empty_identifier() -> None:
    with pytest.raises(ValidationError):
        TaskSession(
            session_id="",
            kind="homework",
            status="waiting_for_user",
            original_request="devoir",
            summary="Aide devoir",
            created_at=1.0,
            updated_at=1.0,
        )
