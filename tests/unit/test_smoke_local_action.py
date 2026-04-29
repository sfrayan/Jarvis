"""Tests unitaires du smoke test local action."""

from __future__ import annotations

import pytest

from tools.smoke_local_action import parse_args

pytestmark = pytest.mark.unit


def test_parse_args_defaults_to_dry_run() -> None:
    args = parse_args(["ouvre Obsidian"])

    assert args.text == "ouvre Obsidian"
    assert args.assisted is False
    assert args.log_level == "INFO"


def test_parse_args_accepts_assisted_and_log_level() -> None:
    args = parse_args(["ouvre Obsidian", "--assisted", "--log-level", "DEBUG"])

    assert args.text == "ouvre Obsidian"
    assert args.assisted is True
    assert args.log_level == "DEBUG"


def test_parse_args_accepts_list_apps_without_text() -> None:
    args = parse_args(["--list-apps", "--limit", "5"])

    assert args.text is None
    assert args.list_apps is True
    assert args.limit == 5


def test_parse_args_requires_text_without_list_apps() -> None:
    with pytest.raises(SystemExit):
        parse_args([])
