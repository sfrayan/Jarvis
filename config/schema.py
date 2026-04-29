"""Schémas Pydantic v2 pour la configuration Jarvis.

Ce module définit **uniquement les classes de validation**. La logique de
chargement (YAML + merge local.yaml + env vars) est reportée à l'Itération 2
(`config/loader.py`).

En Itération 1, on doit pouvoir faire :

    import yaml
    from config.schema import JarvisConfig
    data = yaml.safe_load(open("config/default.yaml"))
    cfg = JarvisConfig.model_validate(data)

Et la validation doit passer sans erreur sur `config/default.yaml` livré.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Sécurité (voir CLAUDE.md — priorité absolue)
# -----------------------------------------------------------------------------
class KillSwitchConfig(BaseModel):
    """Paramètres du kill switch (pynput listener global)."""

    model_config = ConfigDict(extra="forbid")

    hotkey: str = Field(default="f12", description="Touche unique déclenchant l'arrêt")
    escape_long_ms: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maintien d'Échap (ms) déclenchant l'arrêt",
    )
    corner_trigger: bool = Field(
        default=True,
        description="Souris poussée en (0,0) déclenche pyautogui.FAILSAFE",
    )


SafetyMode = Literal["observe", "dry_run", "assisted", "autonomous"]


class SafetyConfig(BaseModel):
    """Politique de sécurité globale de l'agent."""

    model_config = ConfigDict(extra="forbid")

    mode: SafetyMode = Field(
        default="dry_run",
        description="Mode d'exécution — défaut dry_run au premier lancement",
    )
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    allowlist_destructive_patterns: list[str] = Field(
        default_factory=list,
        description="Patterns forçant requires_human=true quelle que soit la confidence",
    )


# -----------------------------------------------------------------------------
# Audio (capture micro)
# -----------------------------------------------------------------------------
class AudioConfig(BaseModel):
    """Paramètres de capture audio d'entrée."""

    model_config = ConfigDict(extra="forbid")

    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    silence_timeout_ms: int = Field(default=800, ge=100, le=5000)
    device_index: int | None = Field(
        default=None,
        description="Index sounddevice ; null = micro par défaut Windows",
    )


# -----------------------------------------------------------------------------
# STT (speech-to-text)
# -----------------------------------------------------------------------------
STTBackend = Literal["faster-whisper"]
STTModel = Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"]
ComputeDevice = Literal["auto", "cuda", "cpu"]


class STTConfig(BaseModel):
    """Configuration du moteur de transcription."""

    model_config = ConfigDict(extra="forbid")

    backend: STTBackend = Field(default="faster-whisper")
    model: STTModel = Field(default="medium")
    language: str = Field(default="fr", min_length=2, max_length=5)
    device: ComputeDevice = Field(default="auto")
    beam_size: int = Field(default=5, ge=1, le=10)


# -----------------------------------------------------------------------------
# Vision (GUI grounding)
# -----------------------------------------------------------------------------
VisionCloudBackend = Literal["anthropic", "openrouter"]


class VisionConfig(BaseModel):
    """Configuration du module de vision pour le grounding GUI."""

    model_config = ConfigDict(extra="forbid")

    backend_local: str = Field(
        default="qwen2.5vl:7b",
        description="Nom de modèle Ollama pour la vision locale",
    )
    backend_cloud: VisionCloudBackend = Field(default="anthropic")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_iterations: int = Field(default=5, ge=1, le=20)
    screenshot_max_long_edge: int = Field(default=1568, ge=512, le=4096)
    screenshot_max_pixels: int = Field(default=1_150_000, ge=100_000, le=10_000_000)


# -----------------------------------------------------------------------------
# TTS (synthèse vocale via Piper)
# -----------------------------------------------------------------------------
TTSBackend = Literal["log", "piper"]


class TTSConfig(BaseModel):
    """Configuration du TTS (client Wyoming Piper)."""

    model_config = ConfigDict(extra="forbid")

    backend: TTSBackend = Field(default="log")
    voice: str = Field(default="fr_FR-siwis-medium")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=10200, ge=1, le=65535)
    timeout_s: float = Field(default=2.0, gt=0.0, le=30.0)
    fallback_to_log: bool = Field(default=True)


# -----------------------------------------------------------------------------
# Mémoire de session (persistance inter-lancements)
# -----------------------------------------------------------------------------
class SessionMemoryConfig(BaseModel):
    """Persistance légère de la session de dialogue entre deux lancements."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Active la sauvegarde de la session courante sur disque",
    )
    directory: str = Field(
        default="data/sessions",
        min_length=1,
        description="Dossier relatif autorisé pour le fichier de session JSON",
    )
    expiry_hours: float = Field(
        default=4.0,
        gt=0.0,
        le=168.0,
        description="Durée de vie max d'une session sauvegardée (heures)",
    )


# -----------------------------------------------------------------------------
# Brouillons locaux
# -----------------------------------------------------------------------------
class DraftStorageConfig(BaseModel):
    """Configuration de sauvegarde locale des brouillons assistant."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Active la sauvegarde locale des AssistantDraft",
    )
    directory: str = Field(
        default="data/drafts",
        min_length=1,
        description="Dossier relatif autorise pour les brouillons Markdown",
    )


# -----------------------------------------------------------------------------
# Cloud fallback (optionnel)
# -----------------------------------------------------------------------------
class CloudConfig(BaseModel):
    """Politique de fallback vers APIs cloud (désactivé par défaut)."""

    model_config = ConfigDict(extra="forbid")

    fallback: bool = Field(default=False)
    retry_after_local_failures: int = Field(default=2, ge=1, le=10)


# -----------------------------------------------------------------------------
# Observabilité
# -----------------------------------------------------------------------------
LogFormat = Literal["console", "json"]


class ObservabilityConfig(BaseModel):
    """Configuration logging et métriques."""

    model_config = ConfigDict(extra="forbid")

    metrics_enabled: bool = Field(default=False)
    metrics_port: int = Field(default=9090, ge=1024, le=65535)
    log_format: LogFormat = Field(default="console")


# -----------------------------------------------------------------------------
# Configuration racine
# -----------------------------------------------------------------------------
class JarvisConfig(BaseModel):
    """Configuration racine du projet Jarvis.

    Note : en Itération 2, cette classe deviendra `BaseSettings` pour lire
    automatiquement les variables d'env (`JARVIS_*`). Pour l'instant, elle
    reste un `BaseModel` pur — le loader gèrera la fusion YAML + env
    explicitement.
    """

    model_config = ConfigDict(extra="forbid")

    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    session_memory: SessionMemoryConfig = Field(default_factory=SessionMemoryConfig)
    drafts: DraftStorageConfig = Field(default_factory=DraftStorageConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
