"""Chargement de la configuration Jarvis depuis YAML.

Pipeline :

1. Lit `config/default.yaml` (obligatoire, commité).
2. Si `config/local.yaml` existe (gitignore), applique un deep-merge récursif
   dessus. Les dicts imbriqués sont fusionnés ; listes et scalaires sont
   remplacés en bloc.
3. Valide le tout contre `JarvisConfig` (Pydantic v2).

Les overrides depuis variables d'env (`JARVIS_SAFETY_MODE`, etc.) ne sont pas
gérés ici — ils sont appliqués directement dans `main.py` là où c'est utile,
ou ignorés pour l'Itération 2 (aucun subsystem ne les consomme encore).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config.schema import JarvisConfig


def load_config(
    default_path: str | Path = "config/default.yaml",
    local_path: str | Path = "config/local.yaml",
) -> JarvisConfig:
    """Charge et valide la configuration Jarvis.

    Args:
        default_path: Chemin du YAML par défaut (obligatoire).
        local_path: Chemin du YAML override (optionnel).

    Returns:
        Instance `JarvisConfig` validée.

    Raises:
        FileNotFoundError: si `default_path` n'existe pas.
        yaml.YAMLError: si un fichier YAML est mal formé.
        pydantic.ValidationError: si la config finale ne valide pas le schema.
    """
    default_file = Path(default_path)
    if not default_file.exists():
        raise FileNotFoundError(f"Config par défaut manquante : {default_file}")

    with default_file.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    data: dict[str, Any] = raw if isinstance(raw, dict) else {}

    local_file = Path(local_path)
    if local_file.exists():
        with local_file.open(encoding="utf-8") as f:
            local_raw = yaml.safe_load(f)
        if isinstance(local_raw, dict):
            data = _deep_merge(data, local_raw)

    return JarvisConfig.model_validate(data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge récursif. `override` a priorité.

    - Dicts imbriqués des deux côtés → fusionnés récursivement.
    - Sinon → `override` remplace.
    """
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
