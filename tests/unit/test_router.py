"""Tests unitaires du routeur d'intention."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from brain.events import IntentDomain
from brain.router import (
    IntentRouter,
    _enforce_confidence_policy,
    _extract_message_content,
    _ollama_host,
    normalize_transcription,
)

pytestmark = pytest.mark.unit


class _FakeOllamaClient:
    def __init__(self, payload: str | Exception) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str,
        options: dict[str, Any],
        keep_alive: str,
    ) -> dict[str, dict[str, str]]:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "format": format,
                "options": options,
                "keep_alive": keep_alive,
            }
        )
        if isinstance(self.payload, Exception):
            raise self.payload
        return {"message": {"content": self.payload}}


class _SlowOllamaClient:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str,
        options: dict[str, Any],
        keep_alive: str,
    ) -> dict[str, dict[str, str]]:
        _ = (model, messages, format, options, keep_alive)
        await asyncio.sleep(self.delay_s)
        return {"message": {"content": '{"intent":"chat","confidence":0.99,"reason":"trop tard"}'}}


def _prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "router_system.txt"
    path.write_text("Tu réponds en JSON.", encoding="utf-8")
    return path


class TestIntentRouter:
    @pytest.mark.asyncio
    async def test_routes_valid_gui_response(self, tmp_path: Path) -> None:
        client = _FakeOllamaClient(
            json.dumps(
                {
                    "intent": "gui",
                    "domain": "routine",
                    "confidence": 0.96,
                    "reason": "Demande d'ouverture d'application",
                }
            )
        )
        router = IntentRouter(
            model="qwen3:latest",
            client=client,
            prompt_path=_prompt_file(tmp_path),
        )

        result = await router.route("organise mon espace de travail")

        assert result.intent == "gui"
        assert result.domain == "routine"
        assert result.confidence == pytest.approx(0.96)
        assert result.original_text == "organise mon espace de travail"
        assert result.normalized_text == "organise mon espace de travail"
        assert result.model == "qwen3:latest"
        assert client.calls[0]["format"] == "json"
        assert client.calls[0]["options"]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_routes_valid_chat_response(self, tmp_path: Path) -> None:
        client = _FakeOllamaClient(
            '{"intent":"chat","confidence":0.91,"reason":"Demande d explication"}'
        )
        router = IntentRouter(client=client, prompt_path=_prompt_file(tmp_path))

        result = await router.route("Docker compose et WSL2")

        assert result.intent == "chat"
        assert result.confidence == pytest.approx(0.91)

    @pytest.mark.asyncio
    async def test_heuristic_routes_gui_without_calling_client(self, tmp_path: Path) -> None:
        client = _FakeOllamaClient(
            '{"intent":"chat","confidence":0.99,"reason":"ne doit pas être appelé"}'
        )
        router = IntentRouter(client=client, prompt_path=_prompt_file(tmp_path))

        result = await router.route("Javis ouvre Discord")

        assert result.intent == "gui"
        assert result.domain == "apps"
        assert result.confidence == pytest.approx(0.9)
        assert result.normalized_text == "jarvis ouvre Discord"
        assert result.model == "heuristic"
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_heuristic_routes_noisy_discord_transcription_as_gui(
        self,
        tmp_path: Path,
    ) -> None:
        client = _FakeOllamaClient(
            '{"intent":"chat","confidence":0.99,"reason":"ne doit pas être appelé"}'
        )
        router = IntentRouter(client=client, prompt_path=_prompt_file(tmp_path))

        result = await router.route("vôme ou discode")

        assert result.intent == "gui"
        assert result.domain == "apps"
        assert result.normalized_text == "ouvre Discord"
        assert result.model == "heuristic"
        assert client.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        [
            "tu m'ouvres Spotify",
            "tu éteins Docker Desktop",
            "éteint Docker Desktop",
            "ouvre KeePass",
            "Ouvre-moi le gestionnaire de tâches",
            "change de slide",
            "slide suivante",
            "de slide",
            "et un docker desktop",
            "Volumes montent.",
            "volume mode.",
            "C'est un couvre-chrome.",
        ],
    )
    async def test_heuristic_routes_common_system_commands_as_gui(
        self,
        tmp_path: Path,
        text: str,
    ) -> None:
        client = _FakeOllamaClient(
            '{"intent":"chat","confidence":0.99,"reason":"ne doit pas être appelé"}'
        )
        router = IntentRouter(client=client, prompt_path=_prompt_file(tmp_path))

        result = await router.route(text)

        assert result.intent == "gui"
        assert result.model == "heuristic"
        assert client.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("text", "intent", "domain"),
        [
            ("ouvre Chrome", "gui", "apps"),
            ("C'est un couvre-chrome.", "gui", "apps"),
            ("clique sur enregistrer", "gui", "vision"),
            ("analyse mon ecran", "gui", "vision"),
            ("ouvre mes telechargements", "gui", "folders"),
            ("mets Spotify en pause", "gui", "media"),
            ("Volumes montent.", "gui", "system"),
            ("volume mode.", "gui", "system"),
            ("allume la lumiere du salon", "gui", "home_assistant"),
            ("au boulot", "gui", "routine"),
            ("memorise que je travaille de nuit", "chat", "memory"),
            ("cherche sur google la meteo", "gui", "web_search"),
            ("ouvre Gmail", "gui", "google_workspace"),
        ],
    )
    async def test_heuristic_assigns_intent_domains(
        self,
        tmp_path: Path,
        text: str,
        intent: str,
        domain: IntentDomain,
    ) -> None:
        client = _FakeOllamaClient(
            '{"intent":"chat","confidence":0.99,"reason":"ne doit pas etre appele"}'
        )
        router = IntentRouter(client=client, prompt_path=_prompt_file(tmp_path))

        result = await router.route(text)

        assert result.intent == intent
        assert result.domain == domain
        assert result.model == "heuristic"
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_heuristic_routes_chat_without_calling_client(self, tmp_path: Path) -> None:
        client = _FakeOllamaClient(
            '{"intent":"gui","confidence":0.99,"reason":"ne doit pas être appelé"}'
        )
        router = IntentRouter(client=client, prompt_path=_prompt_file(tmp_path))

        result = await router.route("salut Jarvis")

        assert result.intent == "chat"
        assert result.domain == "general"
        assert result.confidence == pytest.approx(0.88)
        assert result.model == "heuristic"
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_chat_heuristic_wins_over_gui_words(self, tmp_path: Path) -> None:
        client = _FakeOllamaClient(
            '{"intent":"gui","confidence":0.99,"reason":"ne doit pas être appelé"}'
        )
        router = IntentRouter(client=client, prompt_path=_prompt_file(tmp_path))

        result = await router.route("explique-moi comment ouvrir Discord")

        assert result.intent == "chat"
        assert result.domain == "general"
        assert result.model == "heuristic"
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_targetless_action_is_unknown_without_calling_client(
        self,
        tmp_path: Path,
    ) -> None:
        client = _FakeOllamaClient(
            '{"intent":"gui","confidence":0.99,"reason":"ne doit pas être appelé"}'
        )
        router = IntentRouter(client=client, prompt_path=_prompt_file(tmp_path))

        result = await router.route("ouvre.")

        assert result.intent == "unknown"
        assert result.domain == "unknown"
        assert result.confidence == pytest.approx(0.0)
        assert result.model == "heuristic"
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_empty_text_falls_back_to_unknown(self, tmp_path: Path) -> None:
        client = _FakeOllamaClient('{"intent":"chat","confidence":1,"reason":"x"}')
        router = IntentRouter(client=client, prompt_path=_prompt_file(tmp_path))

        result = await router.route("   ")

        assert result.intent == "unknown"
        assert result.domain == "unknown"
        assert result.confidence == 0.0
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back_to_unknown(self, tmp_path: Path) -> None:
        router = IntentRouter(
            client=_FakeOllamaClient("pas du json"),
            prompt_path=_prompt_file(tmp_path),
        )

        result = await router.route("phrase ambigue sans verbe clair")

        assert result.intent == "unknown"
        assert result.domain == "unknown"
        assert result.confidence == 0.0
        assert "JSON" in result.reason

    @pytest.mark.asyncio
    async def test_client_exception_falls_back_to_unknown(self, tmp_path: Path) -> None:
        router = IntentRouter(
            client=_FakeOllamaClient(RuntimeError("ollama down")),
            prompt_path=_prompt_file(tmp_path),
        )

        result = await router.route("phrase ambigue sans verbe clair")

        assert result.intent == "unknown"
        assert result.domain == "unknown"
        assert result.confidence == 0.0
        assert "RuntimeError" in result.reason

    @pytest.mark.asyncio
    async def test_client_timeout_falls_back_to_unknown(self, tmp_path: Path) -> None:
        router = IntentRouter(
            client=_SlowOllamaClient(delay_s=0.2),
            prompt_path=_prompt_file(tmp_path),
            timeout_s=0.01,
        )

        result = await router.route("phrase ambigue sans verbe clair")

        assert result.intent == "unknown"
        assert result.confidence == 0.0
        assert "TimeoutError" in result.reason


class TestNormalizeTranscription:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Javis ouvre Discord", "jarvis ouvre Discord"),
            ("ouvre Opéra", "ouvre opera"),
            ("ouvre l'Opéra", "ouvre l'opera"),
            ("ouvre vis code", "ouvre vs code"),
            ("docker desk top", "docker desktop"),
            ("au Vodiscord", "ouvre Discord"),
            ("vôme ou discode", "ouvre Discord"),
            ("capture décro", "capture d'écran"),
            ("d'écran", "capture d'écran"),
            ("capture d'écran", "capture d'écran"),
            ("qui passe", "KeePass"),
            ("key pass", "KeePass"),
            ("spoti fi", "Spotify"),
            ("Volumes montent.", "volume monte."),
            ("volume mode.", "volume monte."),
            ("C'est un couvre-chrome.", "ouvre Chrome."),
            ("docker desque top", "Docker Desktop"),
            ("et un docker desktop", "éteins Docker Desktop"),
            ("gestionnaire des tâches", "gestionnaire de tâches"),
            ("ouvre-moi les listes en l'air de tâche", "ouvre-moi le gestionnaire de tâches"),
            ("  ouvre   Chrome  ", "ouvre Chrome"),
        ],
    )
    def test_normalizes_known_stt_errors(self, raw: str, expected: str) -> None:
        assert normalize_transcription(raw) == expected


class TestHelpers:
    def test_ollama_host_defaults_to_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        assert _ollama_host() == "http://localhost:11434"

    def test_ollama_host_uses_env_without_trailing_slash(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/")
        assert _ollama_host() == "http://127.0.0.1:11434"

    def test_extract_message_content_from_dict(self) -> None:
        assert _extract_message_content({"message": {"content": "{}"}}) == "{}"

    def test_extract_message_content_rejects_missing_content(self) -> None:
        with pytest.raises(ValueError, match="message.content"):
            _extract_message_content({"message": {}})

    def test_unknown_confidence_is_capped(self) -> None:
        from brain.router import RouterModelResponse

        response = RouterModelResponse(
            intent="unknown",
            confidence=0.99,
            reason="Ambigu",
        )
        result = _enforce_confidence_policy(response)
        assert result.intent == "unknown"
        assert result.domain == "unknown"
        assert result.confidence == pytest.approx(0.59)

    def test_low_confidence_chat_becomes_unknown(self) -> None:
        from brain.router import RouterModelResponse

        response = RouterModelResponse(
            intent="chat",
            confidence=0.42,
            reason="Peu clair",
        )
        result = _enforce_confidence_policy(response)
        assert result.intent == "unknown"
        assert result.domain == "unknown"
        assert result.confidence == pytest.approx(0.42)
