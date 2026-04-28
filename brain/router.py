"""Routeur d'intention local via Ollama.

Le routeur transforme une transcription STT en `IntentRouted` :

- normalisation légère des noms propres mal transcrits ;
- appel Ollama `qwen3:latest` en JSON strict ;
- validation Pydantic ;
- fallback `unknown` en cas de JSON invalide ou d'erreur réseau.

Les tests injectent un faux client : aucun appel réseau n'est fait en unit.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol

import ollama
from pydantic import BaseModel, ConfigDict, Field

from brain.events import IntentDomain, IntentRouted, IntentType
from config.schema import JarvisConfig
from observability.logger import get_logger

log = get_logger(__name__)

DEFAULT_ROUTER_MODEL = "qwen3:latest"
HEURISTIC_ROUTER_MODEL = "heuristic"
DEFAULT_PROMPT_PATH = Path("brain/prompts/router_system.txt")
_MIN_CONFIDENCE = 0.60
DEFAULT_ROUTER_TIMEOUT_S = 2.0


class OllamaChatClient(Protocol):
    """Contrat minimal du client Ollama async."""

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str,
        options: dict[str, Any],
        keep_alive: str,
    ) -> Any:
        """Retourne une réponse compatible avec `ollama.ChatResponse`."""


class RouterModelResponse(BaseModel):
    """JSON attendu depuis le LLM routeur."""

    model_config = ConfigDict(extra="forbid")

    intent: IntentType
    domain: IntentDomain = "general"
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., min_length=1)


class IntentRouter:
    """Route les transcriptions utilisateur en intentions typées."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_ROUTER_MODEL,
        client: OllamaChatClient | None = None,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
        timeout_s: float = DEFAULT_ROUTER_TIMEOUT_S,
    ) -> None:
        self._model = model
        self._client = client or ollama.AsyncClient(host=_ollama_host())
        self._system_prompt = prompt_path.read_text(encoding="utf-8")
        self._timeout_s = timeout_s

    @classmethod
    def from_config(
        cls,
        config: JarvisConfig,
        *,
        client: OllamaChatClient | None = None,
    ) -> IntentRouter:
        """Construit le routeur depuis la config racine."""
        return cls(model=_router_model_from_env_or_default(config), client=client)

    async def route(self, text: str) -> IntentRouted:
        """Retourne une décision de routage pour `text`."""
        original_text = text.strip()
        normalized_text = normalize_transcription(original_text)

        if not normalized_text:
            return self._fallback(
                original_text=original_text or "(vide)",
                normalized_text="(vide)",
                reason="Transcription vide",
            )

        heuristic = _heuristic_route(normalized_text)
        if heuristic is not None:
            return IntentRouted(
                timestamp=time.time(),
                original_text=original_text,
                normalized_text=normalized_text,
                intent=heuristic.intent,
                domain=_route_domain(heuristic.intent, normalized_text, heuristic.domain),
                confidence=heuristic.confidence,
                reason=heuristic.reason,
                model=HEURISTIC_ROUTER_MODEL,
            )

        try:
            response = await asyncio.wait_for(
                self._client.chat(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": normalized_text},
                    ],
                    format="json",
                    options={
                        "temperature": 0.0,
                        "num_predict": 120,
                    },
                    keep_alive="24h",
                ),
                timeout=self._timeout_s,
            )
            payload = _extract_message_content(response)
            model_response = _parse_model_response(payload)
            model_response = _enforce_confidence_policy(model_response)
            return IntentRouted(
                timestamp=time.time(),
                original_text=original_text,
                normalized_text=normalized_text,
                intent=model_response.intent,
                domain=_route_domain(
                    model_response.intent,
                    normalized_text,
                    model_response.domain,
                ),
                confidence=model_response.confidence,
                reason=model_response.reason,
                model=self._model,
            )
        except Exception as exc:
            log_fn = log.debug if isinstance(exc, TimeoutError) else log.warning
            log_fn(
                "intent_router_fallback",
                error=str(exc) or type(exc).__name__,
                error_type=type(exc).__name__,
                model=self._model,
            )
            return self._fallback(
                original_text=original_text,
                normalized_text=normalized_text,
                reason=f"Routeur indisponible ou JSON invalide: {type(exc).__name__}",
            )

    def _fallback(
        self,
        *,
        original_text: str,
        normalized_text: str,
        reason: str,
    ) -> IntentRouted:
        return IntentRouted(
            timestamp=time.time(),
            original_text=original_text,
            normalized_text=normalized_text,
            intent="unknown",
            domain="unknown",
            confidence=0.0,
            reason=reason,
            model=self._model,
        )


def normalize_transcription(text: str) -> str:
    """Normalise quelques erreurs STT fréquentes sans changer le sens."""
    normalized = text.strip().replace("\u2019", "'")
    replacements = {
        r"\bjavis\b": "jarvis",
        r"\bjarviss\b": "jarvis",
        r"\bopéra\b": "opera",
        r"\bl'opéra\b": "l'opera",
        r"\bvis code\b": "vs code",
        r"\bvisual studio codes\b": "visual studio code",
        r"\bdocker desk top\b": "docker desktop",
        r"\bène huit ène\b": "n8n",
        r"\bau\s+vodiscord\b": "ouvre Discord",
        r"\bvodiscord\b": "Discord",
        r"\bdiscode\b": "Discord",
        r"\bdiscorde\b": "Discord",
        r"\bouvre\s+au\s+discord\b": "ouvre Discord",
        r"\bau\s+discord\b": "ouvre Discord",
        r"\bvôme\s+ou\s+discord\b": "ouvre Discord",
        r"\bvome\s+ou\s+discord\b": "ouvre Discord",
        r"\bcapture\s+décro\b": "capture d'écran",
        r"\bcapture\s+decro\b": "capture d'écran",
        r"(?<!capture\s)\bd'écran\b": "capture d'écran",
        r"\bqui\s+passe\b": "KeePass",
        r"\bkey\s+pass\b": "KeePass",
        r"\bkeep\s+pass\b": "KeePass",
        r"\bspoti\s+fi\b": "Spotify",
        r"\bspoti\s+faille\b": "Spotify",
        r"\bvolumes?\s+montent\b": "volume monte",
        r"\bvolume\s+mode\b": "volume monte",
        r"\bc'est\s+un\s+couvre[-\s]+chrome\b": "ouvre Chrome",
        r"\bcouvre[-\s]+chrome\b": "ouvre Chrome",
        r"\bdocker\s+desque\s+top\b": "Docker Desktop",
        r"\bet\s+un\s+docker\s+desktop\b": "éteins Docker Desktop",
        r"\bgestionnaire\s+des\s+tâches\b": "gestionnaire de tâches",
        r"\b(ouvre-moi\s+)?les\s+listes\s+en\s+l'air\s+de\s+tâche\b": (
            r"\1le gestionnaire de tâches"
        ),
    }
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def _heuristic_route(text: str) -> RouterModelResponse | None:
    """Route instantané des cas évidents, sans appel LLM.

    Le but n'est pas de remplacer qwen3, seulement de garder la boucle vocale
    fluide quand l'intention est évidente ou quand Ollama est froid/lent.
    Les règles CHAT passent avant GUI pour éviter de classer
    "explique-moi comment ouvrir Discord" comme une action.
    """
    lowered = text.casefold().strip(" .!?")

    chat_patterns = (
        r"^(salut|bonjour|bonsoir|coucou|merci)\b",
        r"^(comment (ça|ca) va|ça va|ca va)\b",
        r"^(explique|explique-moi|explique moi|pourquoi|c'est quoi|qu'est-ce)\b",
        r"^(comment faire|comment je fais|tu peux m'expliquer)\b",
        r"^(m.morise|memorise|souviens-toi|souviens toi|oublie)\b",
        r"\b(qu'est-ce que tu sais sur moi|qu'est ce que tu sais sur moi)\b",
        r"\b(qu'est-ce|c'est quoi|pourquoi|ça marche|ca marche)\b",
    )
    if _matches_any(lowered, chat_patterns):
        return RouterModelResponse(
            intent="chat",
            domain=_infer_domain(lowered, "chat"),
            confidence=0.88,
            reason="Règle heuristique: salutation, question ou explication",
        )

    action_patterns = (
        r"\b(ouvre|ouvres|ouvrir|ouvre-moi|m'ouvres)\b",
        r"\b(lance|lances|lancer|démarre|demarre|démarres|demarres)\b",
        r"\b(ferme|fermes|quitte|quittes|éteins|eteins|éteint|eteint|éteindre|eteindre)\b",
        r"\b(arrête|arrete|arrêtes|arretes|coupe|coupes)\b",
        r"\b(allume|allumes|active|actives|d.sactive|desactive|r.gle|regle)\b",
    )
    targetless_action_patterns = (
        r"^(ouvre|ouvres|ouvrir|ouvre-moi|m'ouvres)$",
        r"^(lance|lances|lancer|d.marre|demarre|d.marres|demarres)$",
        r"^(ferme|fermes|quitte|quittes|.teins|eteins|.teint|eteint|.teindre|eteindre)$",
        r"^(arr.te|arrete|arr.tes|arretes|coupe|coupes)$",
        r"^(allume|allumes|active|actives|d.sactive|desactive|r.gle|regle)$",
    )
    if _matches_any(lowered, targetless_action_patterns):
        return RouterModelResponse(
            intent="unknown",
            domain="unknown",
            confidence=0.0,
            reason="Règle heuristique: commande incomplète",
        )

    gui_patterns = (
        *action_patterns,
        r"\b(clique|clic|double clique|appuie|tape|écris|ecris)\b",
        r"\b(capture d'écran|capture ecran|screenshot)\b",
        r"\b(analyse|regarde)\b.*\b(.cran|ecran|fen.tre|fenetre)\b",
        r"\b(va sur|cherche sur|recherche sur)\b",
        r"\bvolume\b.*\b(monte|augmente|baisse|diminue|coupe|muet|mute)\b",
        r"\b(mets|met|joue|pause|reprends|suivante|pr.c.dente|precedente)\b",
        r"\b(mode travail|au boulot|mode iron man)\b",
        r"^(de|du)\s+(slide|diapo)\b",
        r"\b(change|changes|suivant|suivante|précédent|precedent)\b.*\b(slide|diapo)\b",
        r"\b(slide|diapo)\b.*\b(suivant|suivante|précédent|precedent)\b",
    )
    if _matches_any(lowered, gui_patterns):
        return RouterModelResponse(
            intent="gui",
            domain=_infer_domain(lowered, "gui"),
            confidence=0.9,
            reason="Règle heuristique: commande GUI explicite",
        )

    app_patterns = (
        r"\b(discord|chrome|spotify|docker desktop|vs code|visual studio code)\b",
        r"\b(kee?pass|opera|n8n|gestionnaire de tâches|gestionnaire de taches)\b",
        r"\b(prgx|powerpoint|excel|word|notion|gmail|github)\b",
    )
    if _matches_any(lowered, action_patterns) and _matches_any(lowered, app_patterns):
        return RouterModelResponse(
            intent="gui",
            domain=_infer_domain(lowered, "gui"),
            confidence=0.9,
            reason="Règle heuristique: verbe d'action + application",
        )

    return None


def _route_domain(
    intent: IntentType,
    normalized_text: str,
    requested_domain: IntentDomain,
) -> IntentDomain:
    """Domaine final conservateur pour une decision routeur."""
    if intent == "unknown":
        return "unknown"
    if requested_domain != "general":
        return requested_domain
    return _infer_domain(normalized_text, intent)


def _infer_domain(text: str, intent: IntentType) -> IntentDomain:
    """Classe le domaine fonctionnel sans declencher d'action."""
    lowered = text.casefold()
    if intent == "unknown":
        return "unknown"

    if _matches_any(
        lowered,
        (
            r"\b(m.morise|memorise|souviens-toi|souviens toi|oublie|m.moire|memoire)\b",
            r"\b(qu'est-ce que tu sais sur moi|qu'est ce que tu sais sur moi)\b",
        ),
    ):
        return "memory"

    if _matches_any(
        lowered,
        (
            r"\b(home assistant|lumi.re|lumiere|prise|thermostat|sc.ne|scene)\b",
            r"\b(alarme|aspirateur|volet|salon|cuisine|bureau)\b",
        ),
    ):
        return "home_assistant"

    if _matches_any(lowered, (r"\b(mode travail|au boulot|mode iron man|routine)\b",)):
        return "routine"

    if _matches_any(
        lowered,
        (
            r"\b(capture d'.cran|capture ecran|screenshot)\b",
            r"\b(analyse|regarde)\b.*\b(.cran|ecran|fen.tre|fenetre)\b",
            r"\b(clique|clic|double clique|appuie|tape|.cris|ecris)\b",
        ),
    ):
        return "vision"

    if _matches_any(
        lowered,
        (
            r"\b(spotify|youtube|musique|playlist|artiste|titre)\b",
            r"\b(play|pause|reprends|suivante|pr.c.dente|precedente)\b",
        ),
    ):
        return "media"

    if _matches_any(
        lowered,
        (
            r"\b(t.l.chargements|telechargements|documents|images|vid.os|videos)\b",
            r"\b(dossier|dossiers|explorateur|bureau)\b",
        ),
    ):
        return "folders"

    if _matches_any(
        lowered,
        (
            r"\b(cpu|ram|batterie|volume|heure|date|jour|mois|ann.e|annee)\b",
            r"\bvolume\b.*\b(monte|augmente|baisse|diminue|coupe|muet|mute)\b",
            r"\b(pc|windows|shutdown|gestionnaire de t.ches|gestionnaire de taches)\b",
        ),
    ):
        return "system"

    if _matches_any(
        lowered,
        (
            r"\b(google doc|google sheet|google docs|google sheets)\b",
            r"\b(gmail|agenda|outlook|email|mail)\b",
        ),
    ):
        return "google_workspace"

    if _matches_any(lowered, (r"\b(web|internet|serpapi|cherche|recherche|google)\b",)):
        return "web_search"

    if intent == "gui" and _matches_any(
        lowered,
        (
            r"\b(discord|chrome|edge|firefox|opera|spotify|steam|notion|github)\b",
            r"\b(vs code|visual studio code|docker desktop|n8n|ollama|powershell)\b",
            r"\b(keepass|powerpoint|excel|word|slide|diapo)\b",
        ),
    ):
        return "apps"

    return "general"


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


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


def _parse_model_response(payload: str) -> RouterModelResponse:
    data = json.loads(payload)
    return RouterModelResponse.model_validate(data)


def _enforce_confidence_policy(response: RouterModelResponse) -> RouterModelResponse:
    """Applique la règle confidence du prompt côté code aussi."""
    if response.intent == "unknown" and response.confidence >= _MIN_CONFIDENCE:
        return response.model_copy(update={"confidence": 0.59, "domain": "unknown"})
    if response.intent in {"chat", "gui"} and response.confidence < _MIN_CONFIDENCE:
        return RouterModelResponse(
            intent="unknown",
            domain="unknown",
            confidence=response.confidence,
            reason=f"Confidence trop faible pour {response.intent}: {response.reason}",
        )
    return response


def _router_model_from_env_or_default(config: JarvisConfig) -> str:
    """Modèle routeur. Placeholder : config env dédiée arrivera plus tard."""
    _ = config
    return DEFAULT_ROUTER_MODEL


def _ollama_host() -> str:
    """Host Ollama pour le process Python natif Windows.

    `host.docker.internal` sert surtout depuis un conteneur vers l'hôte. Ici
    `main.py` tourne directement sur Windows, donc `localhost` est le chemin
    le plus fiable vers le port publié par Docker Desktop / Ollama.
    """
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
