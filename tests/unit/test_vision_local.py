"""Tests unitaires du client vision local."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from ollama import ResponseError

from brain.vision_local import LocalVisionClient, _extract_message_content
from config.schema import VisionConfig

pytestmark = pytest.mark.unit


class _FakeVisionClient:
    def __init__(self, payload: str | Exception, *, delay_s: float = 0.0) -> None:
        self.payload = payload
        self.delay_s = delay_s
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
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
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if isinstance(self.payload, Exception):
            raise self.payload
        return {"message": {"content": self.payload}}


def _prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "vision_system.txt"
    path.write_text("Réponds en JSON valide.", encoding="utf-8")
    return path


def _valid_payload(**overrides: object) -> str:
    data: dict[str, object] = {
        "thought": "Je vois Chrome.",
        "confidence": 0.91,
        "speech": "Je clique.",
        "actions": [{"type": "left_click", "x": 100, "y": 200}],
        "external_tools": [],
        "requires_human": False,
        "task_complete": False,
    }
    data.update(overrides)
    return json.dumps(data)


class TestLocalVisionClient:
    @pytest.mark.asyncio
    async def test_analyze_valid_response(self, tmp_path: Path) -> None:
        fake = _FakeVisionClient(_valid_payload())
        client = LocalVisionClient(
            VisionConfig(backend_local="qwen2.5vl:7b"),
            client=fake,
            prompt_path=_prompt_file(tmp_path),
        )

        decision = await client.analyze(
            user_request="clique sur enregistrer",
            image_base64="abc123",
        )

        assert decision.confidence == pytest.approx(0.91)
        assert decision.actions[0].type == "left_click"
        assert fake.calls[0]["model"] == "qwen2.5vl:7b"
        assert fake.calls[0]["format"] == "json"
        assert fake.calls[0]["messages"][1]["images"] == ["abc123"]

    @pytest.mark.asyncio
    async def test_empty_user_request_returns_human_required(self, tmp_path: Path) -> None:
        fake = _FakeVisionClient(_valid_payload())
        client = LocalVisionClient(
            VisionConfig(),
            client=fake,
            prompt_path=_prompt_file(tmp_path),
        )

        decision = await client.analyze(user_request=" ", image_base64="abc123")

        assert decision.requires_human is True
        assert decision.actions == []
        assert fake.calls == []

    @pytest.mark.asyncio
    async def test_empty_image_returns_human_required(self, tmp_path: Path) -> None:
        fake = _FakeVisionClient(_valid_payload())
        client = LocalVisionClient(
            VisionConfig(),
            client=fake,
            prompt_path=_prompt_file(tmp_path),
        )

        decision = await client.analyze(user_request="ouvre Discord", image_base64=" ")

        assert decision.requires_human is True
        assert decision.actions == []
        assert fake.calls == []

    @pytest.mark.asyncio
    async def test_invalid_json_returns_human_required(self, tmp_path: Path) -> None:
        client = LocalVisionClient(
            VisionConfig(),
            client=_FakeVisionClient("pas du json"),
            prompt_path=_prompt_file(tmp_path),
        )

        decision = await client.analyze(user_request="ouvre Discord", image_base64="abc123")

        assert decision.requires_human is True
        assert decision.actions == []

    @pytest.mark.asyncio
    async def test_low_confidence_with_actions_returns_human_required(self, tmp_path: Path) -> None:
        client = LocalVisionClient(
            VisionConfig(),
            client=_FakeVisionClient(_valid_payload(confidence=0.2)),
            prompt_path=_prompt_file(tmp_path),
        )

        decision = await client.analyze(user_request="ouvre Discord", image_base64="abc123")

        assert decision.requires_human is True
        assert decision.actions == []

    @pytest.mark.asyncio
    async def test_client_exception_returns_human_required(self, tmp_path: Path) -> None:
        client = LocalVisionClient(
            VisionConfig(),
            client=_FakeVisionClient(RuntimeError("ollama down")),
            prompt_path=_prompt_file(tmp_path),
        )

        decision = await client.analyze(user_request="ouvre Discord", image_base64="abc123")

        assert decision.requires_human is True
        assert decision.actions == []

    @pytest.mark.asyncio
    async def test_ollama_memory_error_returns_specific_human_required(
        self,
        tmp_path: Path,
    ) -> None:
        error = ResponseError(
            "model requires more system memory (12.7 GiB) than is available (7.3 GiB)",
            500,
        )
        client = LocalVisionClient(
            VisionConfig(backend_local="qwen2.5vl:7b"),
            client=_FakeVisionClient(error),
            prompt_path=_prompt_file(tmp_path),
        )

        decision = await client.analyze(user_request="analyse mon ecran", image_base64="abc123")

        assert decision.requires_human is True
        assert decision.actions == []
        assert "memoire" in decision.speech.casefold()

    @pytest.mark.asyncio
    async def test_timeout_returns_human_required(self, tmp_path: Path) -> None:
        client = LocalVisionClient(
            VisionConfig(),
            client=_FakeVisionClient(_valid_payload(), delay_s=0.2),
            prompt_path=_prompt_file(tmp_path),
            timeout_s=0.01,
        )

        decision = await client.analyze(user_request="ouvre Discord", image_base64="abc123")

        assert decision.requires_human is True
        assert decision.actions == []


class TestHelpers:
    def test_extract_message_content_from_dict(self) -> None:
        assert _extract_message_content({"message": {"content": "{}"}}) == "{}"

    def test_extract_message_content_rejects_missing_content(self) -> None:
        with pytest.raises(ValueError, match="message.content"):
            _extract_message_content({"message": {}})
