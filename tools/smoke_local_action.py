"""Smoke test runtime pour une action locale Jarvis.

Usage:
    python tools/smoke_local_action.py "ouvre Obsidian"
    python tools/smoke_local_action.py "ouvre Obsidian" --assisted
    python tools/smoke_local_action.py --list-apps

Par defaut le script force `dry_run`. L'option `--assisted` execute l'action si
la capacite locale est sure selon le registry.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SmokeMode = Literal["dry_run", "assisted"]
_BUILT_IN_APP_NAMES = {
    "Calculator",
    "Chrome",
    "Discord",
    "Docker Desktop",
    "EA App",
    "Edge",
    "Firefox",
    "KeePass",
    "Notepad",
    "Opera",
    "Opera GX",
    "Origin",
    "Paint",
    "Settings",
    "Spotify",
    "Steam",
    "Task Manager",
    "VS Code",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(description="Teste une action locale Jarvis.")
    parser.add_argument(
        "text",
        nargs="?",
        help="Texte utilisateur a router, ex: 'ouvre Obsidian'.",
    )
    parser.add_argument(
        "--assisted",
        action="store_true",
        help="Execute l'action si elle est sure. Par defaut: dry_run.",
    )
    parser.add_argument(
        "--list-apps",
        action="store_true",
        help="Liste des apps detectees par l'inventaire readonly puis quitte.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Nombre maximal d'apps affichees avec --list-apps.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Niveau de logs.",
    )
    args = parser.parse_args(argv)
    if args.text is None and not args.list_apps:
        parser.error("text est requis sauf avec --list-apps")
    return args


def list_detected_apps(*, limit: int) -> int:
    """Journalise les apps dynamiques detectees par l'inventaire local."""
    from hands.inventory import LocalInventoryScanner
    from observability.logger import get_logger

    log = get_logger(__name__)
    inventory = LocalInventoryScanner().scan()
    names = sorted(
        {
            app.name
            for app in inventory.apps
            if app.source == "start_menu" and app.name not in _BUILT_IN_APP_NAMES
        },
        key=str.casefold,
    )
    log.info(
        "smoke_detected_dynamic_apps",
        total_apps=len(inventory.apps),
        dynamic_candidates=len(names),
        shown=min(max(limit, 0), len(names)),
        apps=names[: max(limit, 0)],
        warnings=list(inventory.warnings),
    )
    return 0


async def run_smoke(text: str, *, mode: SmokeMode) -> int:
    """Route puis planifie/executer une action locale."""
    from brain.router import IntentRouter
    from config.loader import load_config
    from hands.local_actions import LocalActionPlanner
    from observability.logger import get_logger

    log = get_logger(__name__)
    config = load_config()
    safety = config.safety.model_copy(update={"mode": mode})
    router = IntentRouter.from_config(config)
    planner = LocalActionPlanner(safety)

    routed = await router.route(text)
    log.info(
        "smoke_intent_routed",
        intent=routed.intent,
        domain=routed.domain,
        confidence=round(routed.confidence, 3),
        model=routed.model,
        normalized_text=routed.normalized_text,
        reason=routed.reason,
    )

    report = planner.plan(routed)
    if report is None:
        log.warning(
            "smoke_no_local_action",
            intent=routed.intent,
            domain=routed.domain,
            normalized_text=routed.normalized_text,
        )
        return 2

    log.info(
        "smoke_local_action_report",
        status=report.status,
        mode=report.mode,
        executed=report.executed,
        requires_human=report.requires_human,
        actions=[{"type": action.type, "text": action.text} for action in report.actions],
        reason=report.reason,
    )
    return 0 if report.status in {"completed", "dry_run", "observe"} else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entree CLI."""
    from observability.logger import configure_logging, get_logger

    args = parse_args(argv)
    configure_logging(level=args.log_level, log_format="console", cache_logger=False)
    log = get_logger(__name__)
    mode: SmokeMode = "assisted" if args.assisted else "dry_run"
    if mode == "assisted":
        log.warning(
            "smoke_assisted_mode_enabled",
            message="Une action locale sure peut etre executee reellement.",
        )
    if args.list_apps:
        return list_detected_apps(limit=args.limit)
    return asyncio.run(run_smoke(args.text, mode=mode))


if __name__ == "__main__":
    raise SystemExit(main())
