"""Point d'entrée de Jarvis.

Séquence de bootstrap (ordre important, cf. CLAUDE.md §Sécurité) :

1. Charge la config (default.yaml + local.yaml).
2. Configure structlog.
3. **Démarre le kill switch AVANT tout autre thread applicatif**.
4. Monte event_bus, state_machine, OODA loop.
5. `asyncio.run(loop.run())` — bloque jusqu'au shutdown.
6. Cleanup kill switch (arrête le listener pynput).

En Itération 2, la loop reste passive (pas de VAD/STT/vision) : elle démarre
en IDLE et attend le kill switch. Les itérations 3+ viendront abonner les
producteurs d'événements à l'event bus dans `_run()`.

Arrêt manuel :
- **F12** : kill switch → state = EMERGENCY_STOP → sortie.
- **Échap maintenu > 1 s** : idem.
- **Ctrl+C dans le terminal** : KeyboardInterrupt → déclenche le kill switch.
"""

from __future__ import annotations

import asyncio
import os
import sys

from config.loader import load_config
from config.schema import JarvisConfig
from core.event_bus import EventBus
from core.loop import OODALoop
from core.state_machine import StateMachine
from observability.logger import configure_logging, get_logger
from safety.kill_switch import KillSwitch


def _bootstrap() -> tuple[JarvisConfig, KillSwitch]:
    """Configure logging puis démarre le kill switch.

    Les deux doivent être les TOUT PREMIERS objets initialisés :
    - Sans logging configuré, les logs partent en sortie brute.
    - Démarrer le kill switch APRÈS d'autres threads laisserait une fenêtre
      où F12 ne stopperait pas les opérations déjà en cours.
    """
    config_path = os.environ.get("JARVIS_CONFIG_PATH", "config/default.yaml")
    config = load_config(default_path=config_path)

    log_level = os.environ.get("JARVIS_LOG_LEVEL", "INFO")
    configure_logging(
        level=log_level,
        log_format=config.observability.log_format,
    )

    log = get_logger(__name__)
    log.info(
        "jarvis_bootstrap",
        config_path=config_path,
        safety_mode=config.safety.mode,
        log_level=log_level.upper(),
        log_format=config.observability.log_format,
    )

    kill_switch = KillSwitch(config.safety.kill_switch)
    kill_switch.start()

    return config, kill_switch


async def _run(config: JarvisConfig, kill_switch: KillSwitch) -> None:
    """Monte l'orchestrateur et tourne jusqu'au shutdown."""
    _ = config  # Itération 2 : config non consommée au-delà du bootstrap
    bus = EventBus()
    sm = StateMachine(bus)
    loop = OODALoop(bus, sm, kill_switch)
    await loop.run()


def main() -> int:
    """Entrypoint synchrone. Retourne un exit code Unix (0 = succès)."""
    config, kill_switch = _bootstrap()
    log = get_logger(__name__)
    log.info("jarvis_starting", iteration=2)

    exit_code = 0
    try:
        asyncio.run(_run(config, kill_switch))
    except KeyboardInterrupt:
        log.info("keyboard_interrupt_received")
        if not kill_switch.triggered:
            kill_switch.trigger("keyboard_interrupt")
    except Exception as exc:
        log.error("jarvis_fatal", error=str(exc), exc_info=True)
        exit_code = 1
    finally:
        kill_switch.stop()
        log.info("jarvis_stopped", exit_code=exit_code)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
