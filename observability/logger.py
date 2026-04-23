"""Configuration de logging pour Jarvis (structlog + stdlib).

Appelée une fois au bootstrap (`main.py`). Les modules applicatifs se bornent
ensuite à :

    from observability.logger import get_logger
    log = get_logger(__name__)
    log.info("event", key=value)

Deux formats sont supportés :

- `console` : coloré, humain, pour le dev.
- `json`    : structuré, parsable, pour la production et les agrégateurs.

Le niveau minimum de log vient de la variable d'env `JARVIS_LOG_LEVEL` en
prod (défaut INFO). `configure_logging` accepte aussi un override explicite.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog

LogFormat = Literal["console", "json"]

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def configure_logging(
    *,
    level: str = "INFO",
    log_format: LogFormat = "console",
    cache_logger: bool = True,
) -> None:
    """Configure structlog globalement. Idempotent.

    Args:
        level: Niveau minimum de log (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        log_format: "console" (dev) ou "json" (prod).
        cache_logger: Si True (défaut prod), les loggers créés via
            `get_logger` sont mis en cache après première utilisation.
            Mettre à False dans les tests pour que les reconfigurations
            soient effectives sur les loggers déjà obtenus.

    Raises:
        ValueError: si `level` n'est pas reconnu.
    """
    level_upper = level.upper()
    if level_upper not in _VALID_LEVELS:
        raise ValueError(
            f"Niveau de log invalide : {level!r}. "
            f"Valeurs acceptées : {sorted(_VALID_LEVELS)}"
        )
    level_int: int = getattr(logging, level_upper)

    # Configure stdlib logging : utile si une lib tierce passe par
    # `logging.getLogger`. On utilise `force=True` pour remplacer
    # d'éventuels handlers pré-existants (tests, reconfigurations).
    logging.basicConfig(
        level=level_int,
        format="%(message)s",
        stream=sys.stderr,
        force=True,
    )

    processors: list[Any] = [
        # Inclut les variables liées par `structlog.contextvars.bind_contextvars`
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        # ConsoleRenderer auto-désactive les couleurs si stderr n'est pas un TTY.
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level_int),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=cache_logger,
    )


def get_logger(name: str | None = None) -> Any:
    """Retourne un logger structlog lié au module appelant.

    À utiliser avec `__name__` pour identifier la source :

        log = get_logger(__name__)
    """
    return structlog.get_logger(name)
