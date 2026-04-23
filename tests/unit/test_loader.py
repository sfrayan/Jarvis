"""Tests unitaires du loader de configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from config.loader import load_config

pytestmark = pytest.mark.unit


@pytest.fixture
def default_yaml(tmp_path: Path) -> Path:
    """Fichier YAML par défaut minimal mais valide."""
    f = tmp_path / "default.yaml"
    f.write_text(
        """
safety:
  mode: dry_run
  kill_switch:
    hotkey: f12
    escape_long_ms: 1000
    corner_trigger: true
stt:
  model: medium
  language: fr
observability:
  log_format: console
""",
        encoding="utf-8",
    )
    return f


# ---------------------------------------------------------------------
# Chargement par défaut
# ---------------------------------------------------------------------
class TestLoadDefaultOnly:
    def test_loads_default_without_local(
        self, default_yaml: Path, tmp_path: Path
    ) -> None:
        config = load_config(
            default_path=default_yaml,
            local_path=tmp_path / "absent.yaml",
        )
        assert config.safety.mode == "dry_run"
        assert config.stt.model == "medium"
        assert config.observability.log_format == "console"

    def test_missing_default_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Config par défaut"):
            load_config(default_path=tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------
# Override par local.yaml
# ---------------------------------------------------------------------
class TestLocalOverride:
    def test_local_overrides_top_level(
        self, default_yaml: Path, tmp_path: Path
    ) -> None:
        local = tmp_path / "local.yaml"
        local.write_text(
            """
safety:
  mode: autonomous
stt:
  model: large-v3
""",
            encoding="utf-8",
        )
        config = load_config(default_path=default_yaml, local_path=local)
        assert config.safety.mode == "autonomous"
        assert config.stt.model == "large-v3"
        # Valeurs non-overridées conservées
        assert config.safety.kill_switch.hotkey == "f12"

    def test_deep_merge_preserves_sibling_keys(
        self, default_yaml: Path, tmp_path: Path
    ) -> None:
        """Override d'un sous-champ préserve les autres clés du sous-arbre."""
        local = tmp_path / "local.yaml"
        local.write_text(
            """
safety:
  kill_switch:
    escape_long_ms: 2000
""",
            encoding="utf-8",
        )
        config = load_config(default_path=default_yaml, local_path=local)
        # Valeurs soeurs préservées
        assert config.safety.mode == "dry_run"
        assert config.safety.kill_switch.hotkey == "f12"
        assert config.safety.kill_switch.corner_trigger is True
        # Override appliqué
        assert config.safety.kill_switch.escape_long_ms == 2000

    def test_empty_local_yaml_is_noop(
        self, default_yaml: Path, tmp_path: Path
    ) -> None:
        local = tmp_path / "local.yaml"
        local.write_text("", encoding="utf-8")
        config = load_config(default_path=default_yaml, local_path=local)
        assert config.safety.mode == "dry_run"

    def test_local_yaml_with_only_comments_is_noop(
        self, default_yaml: Path, tmp_path: Path
    ) -> None:
        local = tmp_path / "local.yaml"
        local.write_text("# rien à override\n", encoding="utf-8")
        config = load_config(default_path=default_yaml, local_path=local)
        assert config.safety.mode == "dry_run"


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
class TestValidation:
    def test_invalid_enum_raises_validation_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            """
safety:
  mode: INVALID_MODE
""",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            load_config(default_path=bad)

    def test_extra_key_raises_validation_error(self, tmp_path: Path) -> None:
        """Les schemas ont `extra=forbid` — une clé inconnue doit échouer."""
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            """
safety:
  mode: dry_run
  unknown_key: 42
""",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            load_config(default_path=bad)

    def test_empty_yaml_uses_all_defaults(self, tmp_path: Path) -> None:
        """Un YAML vide produit une config entièrement par défaut."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        config = load_config(default_path=empty)
        # Défauts du schema
        assert config.safety.mode == "dry_run"
        assert config.stt.model == "medium"
