"""Client vision local via Ollama.

Ce module ne capture pas l'écran et n'exécute aucune action. Il prend une image
déjà encodée en base64 (Itération 5 : `hands/screenshot.py`) et demande au
modèle vision local de produire une `VisionDecision` validée.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Protocol

import ollama
from pydantic import ValidationError

from brain.vision_contracts import VisionDecision, human_required_decision
from config.schema import VisionConfig
from observability.logger import get_logger

log = get_logger(__name__)

DEFAULT_VISION_PROMPT_PATH = Path("brain/prompts/vision_system.txt")
DEFAULT_VISION_TIMEOUT_S = 20.0


class OllamaVisionClient(Protocol):
    """Contrat minimal du client Ollama async pour un modèle vision."""

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        format: str,
        options: dict[str, Any],
        keep_alive: str,
    ) -> Any:
        """Retourne une réponse compatible avec `ollama.ChatResponse`."""


class LocalVisionClient:
    """Client Qwen2.5-VL local via Ollama."""

    def __init__(
        self,
        config: VisionConfig,
        *,
        client: OllamaVisionClient | None = None,
        prompt_path: Path = DEFAULT_VISION_PROMPT_PATH,
        timeout_s: float = DEFAULT_VISION_TIMEOUT_S,
    ) -> None:
        self._config = config
        self._client = client or ollama.AsyncClient(host=_ollama_host())
        self._system_prompt = prompt_path.read_text(encoding="utf-8")
        self._timeout_s = timeout_s

    async def analyze(
        self,
        *,
        user_request: str,
        image_base64: str,
    ) -> VisionDecision:
        """Analyse une capture d'écran encodée en base64.

        Args:
            user_request: Demande utilisateur déjà transcrite.
            image_base64: Image PNG/JPEG en base64, sans préfixe data URL.

        Returns:
            Décision vision validée et sûre.
        """
        request = user_request.strip()
        if not request:
            return human_required_decision(
                thought="Demande utilisateur vide.",
                speech="Je n'ai pas compris la demande, peux-tu répéter ?",
            )
        if not image_base64.strip():
            return human_required_decision(
                thought="Capture d'écran vide.",
                speech="Je n'ai pas d'image de l'écran à analyser.",
            )

        try:
            response = await asyncio.wait_for(
                self._client.chat(
                    model=self._config.backend_local,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {
                            "role": "user",
                            "content": _user_prompt(request),
                            "images": [image_base64],
                        },
                    ],
                    format="json",
                    options={"temperature": 0.0, "num_predict": 512},
                    keep_alive="10m",
                ),
                timeout=self._timeout_s,
            )
            payload = _extract_message_content(response)
            return _parse_vision_decision(payload)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            log.warning(
                "vision_local_invalid_response",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return human_required_decision(
                thought="Réponse vision locale invalide.",
                speech="Je n'ai pas compris l'écran, peux-tu confirmer ?",
            )
        except ollama.ResponseError as exc:
            if _is_ollama_memory_error(exc):
                log.warning(
                    "vision_local_insufficient_memory",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    model=self._config.backend_local,
                    status_code=exc.status_code,
                )
                return human_required_decision(
                    thought="Memoire systeme insuffisante pour le modele vision local.",
                    speech="La vision locale manque de memoire. Essaie un modele vision plus leger.",
                )

            log.warning(
                "vision_local_unavailable",
                error=str(exc) or type(exc).__name__,
                error_type=type(exc).__name__,
                model=self._config.backend_local,
                status_code=exc.status_code,
            )
            return human_required_decision(
                thought="Vision locale indisponible.",
                speech="Je ne peux pas analyser l'ecran pour le moment.",
            )
        except Exception as exc:
            log.warning(
                "vision_local_unavailable",
                error=str(exc) or type(exc).__name__,
                error_type=type(exc).__name__,
            )
            return human_required_decision(
                thought="Vision locale indisponible.",
                speech="Je ne peux pas analyser l'écran pour le moment.",
            )


def _user_prompt(user_request: str) -> str:
    return (
        "Demande utilisateur :\n"
        f"{user_request}\n\n"
        "Analyse l'image et réponds uniquement avec le JSON du contrat."
    )


def _extract_message_content(response: Any) -> str:
    """Extrait `message.content` depuis un ChatResponse ou un dict de test."""
    if isinstance(response, dict):
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

    message_obj = getattr(response, "message", None)
    content = getattr(message_obj, "content", None)
    if isinstance(content, str):
        return content

    if hasattr(response, "__getitem__"):
        try:
            message = response["message"]
            content = message["content"]
            if isinstance(content, str):
                return content
        except (KeyError, TypeError):
            pass

    raise ValueError("Réponse Ollama sans message.content")


def _parse_vision_decision(payload: str) -> VisionDecision:
    data = json.loads(payload)
    return VisionDecision.model_validate(data)


def _is_ollama_memory_error(exc: ollama.ResponseError) -> bool:
    message = f"{exc.error} {exc}".casefold()
    return (
        "requires more system memory" in message
        or "not enough memory" in message
        or "out of memory" in message
    )


def _ollama_host() -> str:
    """Host Ollama depuis le process Python natif Windows."""
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
