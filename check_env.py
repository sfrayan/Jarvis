"""Vérification de l'environnement Jarvis.

Script standalone, exécutable avec `python check_env.py`.

Aucun effet de bord : aucune création de fichier, aucune installation, aucun
démarrage de service. Lecture seule.

Chaque vérification produit un `CheckResult` avec un statut parmi :
    OK   : critère satisfait
    WARN : critère non bloquant — affiche une recommandation
    FAIL : critère bloquant — Jarvis ne peut pas fonctionner correctement

Code de sortie : 0 si aucun FAIL, 1 sinon.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, NamedTuple


# -----------------------------------------------------------------------------
# Couleurs ANSI (Windows Terminal / PowerShell 5+ les supportent nativement)
# -----------------------------------------------------------------------------
class _C:
    OK = "\033[32m"  # vert
    WARN = "\033[33m"  # jaune
    FAIL = "\033[31m"  # rouge
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


Status = Literal["OK", "WARN", "FAIL"]


class CheckResult(NamedTuple):
    name: str
    status: Status
    detail: str
    hint: str | None = None


# -----------------------------------------------------------------------------
# Constantes — ajuster si la stack évolue
# -----------------------------------------------------------------------------
REQUIRED_MODELS: list[str] = [
    "qwen3:latest",
    "qwen2.5-coder:7b",
    "deepseek-coder:6.7b",
    "qwen2.5vl:7b",
    "nomic-embed-text:latest",
]

EXPECTED_COMPOSE_SERVICES: list[str] = ["ollama", "chromadb", "piper-tts", "n8n"]

MIN_PYTHON = (3, 11)


# -----------------------------------------------------------------------------
# Vérifications individuelles
# -----------------------------------------------------------------------------
def check_python() -> CheckResult:
    v = sys.version_info
    current = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= MIN_PYTHON:
        return CheckResult("Python", "OK", current)
    return CheckResult(
        "Python",
        "FAIL",
        f"{current} < {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
        hint="Installer Python 3.11+ : https://www.python.org/downloads/",
    )


def check_os() -> CheckResult:
    name = platform.system()
    if name == "Windows":
        return CheckResult("OS", "OK", f"Windows {platform.release()}")
    return CheckResult(
        "OS",
        "WARN",
        f"{name} {platform.release()} — Jarvis cible Windows 11",
        hint="Les modules hands/* utilisent des APIs Windows natives",
    )


def check_gpu() -> CheckResult:
    try:
        import torch  # type: ignore[import-not-found, import-untyped]
    except ImportError:
        return CheckResult(
            "GPU CUDA",
            "WARN",
            "torch non installé (skip)",
            hint="pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124",
        )
    if not torch.cuda.is_available():
        return CheckResult(
            "GPU CUDA",
            "WARN",
            "torch présent mais aucun GPU CUDA détecté — fallback CPU",
            hint="Vérifier pilote NVIDIA et/ou réinstaller torch en variante CUDA",
        )
    name = torch.cuda.get_device_name(0)
    count = torch.cuda.device_count()
    return CheckResult("GPU CUDA", "OK", f"{name} ({count} dispositif(s))")


def check_screen() -> CheckResult:
    try:
        import mss
    except ImportError as exc:
        return CheckResult("Écran", "FAIL", f"mss non installé ({exc})")
    try:
        with mss.mss() as sct:
            monitors = sct.monitors
            # monitors[0] est la bounding box de tous les moniteurs,
            # monitors[1:] les moniteurs individuels.
            if len(monitors) < 2:
                return CheckResult("Écran", "FAIL", "aucun moniteur détecté")
            primary = monitors[1]
            return CheckResult(
                "Écran",
                "OK",
                f"{len(monitors) - 1} moniteur(s) · primaire {primary['width']}x{primary['height']}",
            )
    except Exception as exc:  # bibliothèque externe, cause inconnue a priori
        return CheckResult("Écran", "FAIL", f"erreur mss : {exc}")


def check_audio() -> CheckResult:
    try:
        import sounddevice as sd
    except ImportError as exc:
        return CheckResult("Audio I/O", "FAIL", f"sounddevice non installé ({exc})")
    try:
        default_in = sd.query_devices(kind="input")
        default_out = sd.query_devices(kind="output")
    except Exception as exc:  # PortAudio peut lever des erreurs variées
        return CheckResult("Audio I/O", "FAIL", f"erreur sounddevice : {exc}")
    if not default_in:
        return CheckResult(
            "Audio I/O",
            "FAIL",
            "aucun micro par défaut",
            hint="Vérifier Paramètres Windows → Confidentialité → Microphone",
        )
    mic_name = default_in["name"] if isinstance(default_in, dict) else str(default_in)
    spk_name = default_out["name"] if isinstance(default_out, dict) else "?"
    return CheckResult("Audio I/O", "OK", f"in={mic_name} | out={spk_name}")


def check_docker() -> CheckResult:
    if shutil.which("docker") is None:
        return CheckResult(
            "Docker",
            "FAIL",
            "binaire docker introuvable dans PATH",
            hint="Installer Docker Desktop : https://www.docker.com/products/docker-desktop/",
        )
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return CheckResult("Docker", "FAIL", "`docker ps` timeout (daemon bloqué ?)")
    if result.returncode != 0:
        stderr_excerpt = result.stderr.strip().splitlines()[0][:180] if result.stderr else ""
        return CheckResult(
            "Docker",
            "FAIL",
            "daemon injoignable",
            hint=f"Lancer Docker Desktop. stderr : {stderr_excerpt}",
        )
    return CheckResult("Docker", "OK", "daemon répond")


def _fetch_ollama_tags() -> dict[str, Any] | None:
    """Interroge `/api/tags` ; retourne le JSON ou None si injoignable."""
    try:
        import httpx
    except ImportError:
        return None
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def check_ollama_reachable(tags: dict[str, Any] | None) -> CheckResult:
    if tags is None:
        return CheckResult(
            "Ollama",
            "FAIL",
            "injoignable sur :11434",
            hint="docker compose up -d  (ou `ollama serve` en natif)",
        )
    n_models = len(tags.get("models", []))
    return CheckResult("Ollama", "OK", f"{n_models} modèle(s) chargé(s)")


def _normalize_model(name: str) -> str:
    """Ajoute `:latest` si aucune tag explicite (convention Ollama)."""
    return name if ":" in name else f"{name}:latest"


def check_ollama_models(tags: dict[str, Any] | None) -> CheckResult:
    if tags is None:
        return CheckResult(
            "Modèles Ollama",
            "FAIL",
            "skip : Ollama injoignable (check précédent)",
        )
    present = {_normalize_model(str(m.get("name", ""))) for m in tags.get("models", [])}
    required = {_normalize_model(n) for n in REQUIRED_MODELS}
    missing = sorted(required - present)
    if not missing:
        return CheckResult("Modèles Ollama", "OK", f"tous présents ({len(required)})")
    pulls = " ; ".join(f"ollama pull {m}" for m in missing)
    return CheckResult(
        "Modèles Ollama",
        "WARN",
        f"manquant(s) : {', '.join(missing)}",
        hint=pulls,
    )


def check_containers() -> CheckResult:
    project_root = Path(__file__).parent
    if not (project_root / "docker-compose.yml").exists():
        return CheckResult("Conteneurs Jarvis", "WARN", "pas de docker-compose.yml ici")
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--services", "--status", "running"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project_root,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return CheckResult("Conteneurs Jarvis", "WARN", "`docker compose ps` indisponible")
    if result.returncode != 0:
        return CheckResult(
            "Conteneurs Jarvis",
            "WARN",
            "commande en échec — normal avant premier `up -d`",
            hint="docker compose up -d",
        )
    running = set(result.stdout.split())
    expected = set(EXPECTED_COMPOSE_SERVICES)
    missing = sorted(expected - running)
    if not missing and running >= expected:
        return CheckResult("Conteneurs Jarvis", "OK", f"4/4 up : {', '.join(sorted(running))}")
    if not running:
        return CheckResult(
            "Conteneurs Jarvis",
            "WARN",
            "aucun conteneur Jarvis up",
            hint="docker compose up -d",
        )
    return CheckResult(
        "Conteneurs Jarvis",
        "WARN",
        f"{len(running & expected)}/4 up · manquant(s) : {', '.join(missing)}",
        hint="docker compose up -d",
    )


def check_env_file() -> CheckResult:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        return CheckResult("Fichier .env", "OK", str(env_path))
    return CheckResult(
        "Fichier .env",
        "WARN",
        "absent",
        hint="copy .env.example .env  (PowerShell)",
    )


# -----------------------------------------------------------------------------
# Rendu
# -----------------------------------------------------------------------------
def _color_for(status: Status) -> str:
    return {"OK": _C.OK, "WARN": _C.WARN, "FAIL": _C.FAIL}[status]


def _print_result(result: CheckResult) -> None:
    color = _color_for(result.status)
    status_tag = f"[{result.status}]"
    name = f"{result.name:<20}"
    print(f"  {color}{status_tag:<7}{_C.RESET} {name} {_C.DIM}{result.detail}{_C.RESET}")
    if result.hint and result.status != "OK":
        print(f"  {_C.DIM}           → {result.hint}{_C.RESET}")


def _enable_windows_ansi() -> None:
    """Active le support ANSI sur les consoles Windows anciennes (conhost)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        # ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        kernel32.SetConsoleMode(handle, 0x0007)
    except Exception:
        # Pas critique : si l'activation échoue, les séquences seront brutes.
        pass


def main() -> int:
    _enable_windows_ansi()

    print(f"\n{_C.BOLD}{_C.CYAN}Jarvis — vérification de l'environnement{_C.RESET}\n")

    tags = _fetch_ollama_tags()
    results: list[CheckResult] = [
        check_python(),
        check_os(),
        check_gpu(),
        check_screen(),
        check_audio(),
        check_docker(),
        check_ollama_reachable(tags),
        check_ollama_models(tags),
        check_containers(),
        check_env_file(),
    ]

    for result in results:
        _print_result(result)

    n_ok = sum(1 for r in results if r.status == "OK")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    total = len(results)

    print()
    summary = f"{n_ok}/{total} OK · {n_warn} WARN · {n_fail} FAIL"
    summary_color = _C.OK if n_fail == 0 else _C.FAIL
    print(f"  {summary_color}{_C.BOLD}{summary}{_C.RESET}\n")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
