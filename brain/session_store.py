"""Persistance légère de la session de dialogue entre deux lancements.

Le `SessionStore` sauvegarde la `TaskSession` courante dans un fichier JSON
atomique. Au prochain lancement, `DialogueManager` peut recharger la session
si elle n'a pas expiré (TTL configurable via `SessionMemoryConfig.expiry_hours`).

Principes :
- Écriture atomique (write → rename) pour éviter la corruption.
- Un seul fichier de session active (`_SESSION_FILENAME`).
- Pydantic v2 `.model_dump_json()` / `.model_validate_json()` pour la sérialisation.
- Quand `enabled=False` (défaut), toutes les opérations sont des no-ops.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from brain.task_session import TaskSession
from config.schema import SessionMemoryConfig
from observability.logger import get_logger

log = get_logger(__name__)

_SESSION_FILENAME: Final[str] = "current_session.json"
_TEMP_SUFFIX: Final[str] = ".tmp"


class SessionSnapshot(BaseModel):
    """Enveloppe JSON stockée sur disque : session + métadonnées de fraîcheur."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    saved_at: float = Field(..., ge=0.0, description="Timestamp UNIX de la sauvegarde")
    session: TaskSession


class SessionStore:
    """Lecture/écriture d'une session de dialogue sur disque.

    Toutes les méthodes sont synchrones (I/O fichier local négligeable).
    Quand `config.enabled` est False, aucun accès disque n'est effectué.
    """

    def __init__(
        self,
        config: SessionMemoryConfig,
        *,
        clock: type[float] | None = None,
    ) -> None:
        self._enabled = config.enabled
        self._expiry_s = config.expiry_hours * 3600.0
        self._directory = Path(config.directory)
        self._clock = clock or time.time
        self._session_path = self._directory / _SESSION_FILENAME

    @property
    def enabled(self) -> bool:
        """Indique si la persistance est activée."""
        return self._enabled

    @property
    def session_path(self) -> Path:
        """Chemin absolu du fichier de session courant."""
        return self._session_path

    def save(self, session: TaskSession) -> bool:
        """Sauvegarde la session sur disque. Retourne True si écrit.

        N'écrit rien si :
        - `enabled` est False ;
        - la session est terminée ou annulée (pas utile à restaurer).
        """
        if not self._enabled:
            return False

        if session.status in {"cancelled", "completed"}:
            log.debug(
                "session_store_skip_terminal",
                status=session.status,
                session_id=session.session_id,
            )
            self.clear()
            return False

        snapshot = SessionSnapshot(
            saved_at=self._clock(),
            session=session,
        )

        self._ensure_directory()
        tmp_path = self._session_path.with_suffix(_TEMP_SUFFIX)
        try:
            tmp_path.write_text(
                snapshot.model_dump_json(indent=2),
                encoding="utf-8",
            )
            # Remplacement atomique (Windows : remplace si existant)
            os.replace(str(tmp_path), str(self._session_path))
        except OSError as exc:
            log.warning(
                "session_store_save_failed",
                error=str(exc),
                path=str(self._session_path),
            )
            # Nettoyage du fichier temporaire si le rename a échoué
            tmp_path.unlink(missing_ok=True)
            return False

        log.info(
            "session_store_saved",
            session_id=session.session_id,
            kind=session.kind,
            status=session.status,
            path=str(self._session_path),
        )
        return True

    def load(self) -> TaskSession | None:
        """Charge la session depuis le disque, si elle existe et n'est pas expirée.

        Retourne None si :
        - `enabled` est False ;
        - le fichier n'existe pas ;
        - le JSON est corrompu ;
        - la session est expirée (TTL dépassé) ;
        - la session est dans un état terminal.
        """
        if not self._enabled:
            return None

        if not self._session_path.is_file():
            return None

        try:
            raw = self._session_path.read_text(encoding="utf-8")
            snapshot = SessionSnapshot.model_validate_json(raw)
        except Exception as exc:
            log.warning(
                "session_store_load_corrupt",
                error=str(exc),
                path=str(self._session_path),
            )
            self.clear()
            return None

        # Vérification TTL
        age_s = self._clock() - snapshot.saved_at
        if age_s > self._expiry_s:
            log.info(
                "session_store_expired",
                age_hours=round(age_s / 3600.0, 2),
                expiry_hours=round(self._expiry_s / 3600.0, 2),
                session_id=snapshot.session.session_id,
            )
            self.clear()
            return None

        # Ignorer les sessions terminales
        if snapshot.session.status in {"cancelled", "completed"}:
            log.debug(
                "session_store_skip_terminal_on_load",
                status=snapshot.session.status,
            )
            self.clear()
            return None

        log.info(
            "session_store_restored",
            session_id=snapshot.session.session_id,
            kind=snapshot.session.kind,
            status=snapshot.session.status,
            age_minutes=round(age_s / 60.0, 1),
        )
        return snapshot.session

    def clear(self) -> None:
        """Supprime le fichier de session s'il existe."""
        try:
            self._session_path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning(
                "session_store_clear_failed",
                error=str(exc),
                path=str(self._session_path),
            )

    def _ensure_directory(self) -> None:
        """Crée le dossier de destination s'il n'existe pas."""
        self._directory.mkdir(parents=True, exist_ok=True)
