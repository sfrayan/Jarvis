"""Point d'entrée de Jarvis.

Séquence de bootstrap (ordre important, cf. CLAUDE.md §Sécurité) :

1. Charge la config (default.yaml + local.yaml).
2. Configure structlog.
3. **Démarre le kill switch AVANT tout autre thread applicatif**.
4. Monte event_bus, state_machine, OODA loop et EarsService.
5. `asyncio.run(_run(...))` — bloque jusqu'au shutdown.
6. Cleanup kill switch (arrête le listener pynput).

En Itération 3, `EarsService` écoute le micro : VAD Silero -> STT
faster-whisper -> événement `Transcription`. Le routeur n'existe pas encore,
donc les transcriptions sont simplement logguées et la FSM revient en IDLE.

Arrêt manuel :
- **F12** : kill switch → state = EMERGENCY_STOP → sortie.
- **Échap maintenu > 1 s** : idem.
- **Ctrl+C dans le terminal** : KeyboardInterrupt → déclenche le kill switch.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import suppress

from config.loader import load_config
from config.schema import JarvisConfig
from core.event_bus import EventBus
from core.loop import OODALoop
from core.state_machine import StateMachine
from ears.events import Transcription
from ears.service import EarsService
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
    bus = EventBus()
    sm = StateMachine(bus)
    loop = OODALoop(bus, sm, kill_switch)
    ears = EarsService(
        audio_config=config.audio,
        stt_config=config.stt,
        event_bus=bus,
        state_machine=sm,
    )

    bus.subscribe(Transcription, _log_transcription)

    loop_task = asyncio.create_task(loop.run(), name="jarvis_ooda_loop")
    ears_task = asyncio.create_task(ears.run(), name="jarvis_ears_service")

    done, pending = await asyncio.wait(
        {loop_task, ears_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    if loop_task in done:
        ears.stop()
    if ears_task in done:
        loop.stop()

    await _drain_or_cancel(pending)

    for task in (loop_task, ears_task):
        if task.done():
            task.result()


async def _log_transcription(event: Transcription) -> None:
    """Handler temporaire tant que le routeur d'intention n'existe pas."""
    log = get_logger(__name__)
    log.info(
        "user_transcription",
        text=event.text,
        language=event.language,
        language_probability=round(event.language_probability, 3),
        inference_duration_ms=round(event.inference_duration_ms, 1),
        audio_duration_ms=round(event.audio_duration_ms, 1),
    )


async def _drain_or_cancel(tasks: set[asyncio.Task[None]], *, timeout_s: float = 5.0) -> None:
    """Attend les tâches après signal d'arrêt, puis annule si besoin."""
    if not tasks:
        return

    done, pending = await asyncio.wait(tasks, timeout=timeout_s)
    for task in pending:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    for task in done:
        task.result()


def main() -> int:
    """Entrypoint synchrone. Retourne un exit code Unix (0 = succès)."""
    config, kill_switch = _bootstrap()
    log = get_logger(__name__)
    log.info("jarvis_starting", iteration=3)

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
