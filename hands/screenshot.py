"""Capture d'ecran locale pour le module vision.

Ce module ne clique pas, ne tape rien, et n'ouvre aucune application. Il capture
uniquement le bureau via `mss`, redimensionne l'image pour le modele vision, puis
retourne un PNG encode en base64.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from types import TracebackType
from typing import Literal, Protocol, cast

import mss
from PIL import Image

from config.schema import VisionConfig

ImageFormat = Literal["png"]


class ScreenShotLike(Protocol):
    """Surface minimale retournee par `mss.grab`."""

    size: tuple[int, int]
    rgb: bytes


class MSSClientLike(Protocol):
    """Surface minimale du contexte `mss.mss()`."""

    monitors: Sequence[Mapping[str, int]]

    def __enter__(self) -> MSSClientLike:
        """Entre dans le contexte de capture."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Libere les ressources de capture."""

    def grab(self, monitor: Mapping[str, int]) -> ScreenShotLike:
        """Capture le moniteur donne."""


MSSFactory = Callable[[], MSSClientLike]


def _default_mss_factory() -> MSSClientLike:
    return cast(MSSClientLike, mss.mss())


@dataclass(frozen=True, slots=True)
class ScreenshotFrame:
    """Image capturee et prete pour un modele vision."""

    image_base64: str
    width: int
    height: int
    original_width: int
    original_height: int
    left: int
    top: int
    format: ImageFormat = "png"

    @property
    def resized(self) -> bool:
        """Indique si l'image a ete reduite avant encodage."""
        return (self.width, self.height) != (self.original_width, self.original_height)

    @property
    def scale_x(self) -> float:
        """Facteur image -> coordonnee native horizontale."""
        return self.original_width / self.width

    @property
    def scale_y(self) -> float:
        """Facteur image -> coordonnee native verticale."""
        return self.original_height / self.height


class ScreenshotCapture:
    """Capture le bureau Windows pour alimenter la vision locale."""

    def __init__(
        self,
        config: VisionConfig,
        *,
        mss_factory: MSSFactory = _default_mss_factory,
    ) -> None:
        self._config = config
        self._mss_factory = mss_factory

    def capture(self, *, monitor_index: int = 0) -> ScreenshotFrame:
        """Capture un moniteur `mss`.

        `monitor_index=0` correspond au bureau virtuel complet dans `mss`.
        Les index 1..n correspondent aux moniteurs physiques.
        """
        with self._mss_factory() as capture:
            if not capture.monitors:
                raise RuntimeError("Aucun moniteur detecte par mss")
            if monitor_index < 0 or monitor_index >= len(capture.monitors):
                raise ValueError(f"Index moniteur invalide: {monitor_index}")
            monitor = capture.monitors[monitor_index]
            screenshot = capture.grab(monitor)

        image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        original_width, original_height = image.size
        resized = _resize_for_vision(
            image,
            max_long_edge=self._config.screenshot_max_long_edge,
            max_pixels=self._config.screenshot_max_pixels,
        )
        width, height = resized.size

        buffer = BytesIO()
        resized.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        return ScreenshotFrame(
            image_base64=encoded,
            width=width,
            height=height,
            original_width=original_width,
            original_height=original_height,
            left=int(monitor.get("left", 0)),
            top=int(monitor.get("top", 0)),
        )


def _resize_for_vision(
    image: Image.Image,
    *,
    max_long_edge: int,
    max_pixels: int,
) -> Image.Image:
    width, height = image.size
    target_width, target_height = _target_size(
        width=width,
        height=height,
        max_long_edge=max_long_edge,
        max_pixels=max_pixels,
    )
    if (target_width, target_height) == (width, height):
        return image
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _target_size(
    *,
    width: int,
    height: int,
    max_long_edge: int,
    max_pixels: int,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Dimensions de capture invalides")

    long_edge_scale = max_long_edge / max(width, height)
    pixel_scale = math.sqrt(max_pixels / (width * height))
    scale = min(1.0, long_edge_scale, pixel_scale)
    if scale >= 1.0:
        return width, height
    return max(1, int(width * scale)), max(1, int(height * scale))
