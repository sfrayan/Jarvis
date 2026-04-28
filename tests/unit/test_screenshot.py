"""Tests unitaires de la capture ecran locale."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from types import TracebackType

import pytest
from PIL import Image

from config.schema import VisionConfig
from hands.screenshot import MSSClientLike, MSSFactory, ScreenshotCapture

pytestmark = pytest.mark.unit


class _FakeShot:
    def __init__(self, image: Image.Image) -> None:
        self.size = image.size
        self.rgb = image.convert("RGB").tobytes()


class _FakeMSS:
    def __init__(
        self,
        image: Image.Image,
        *,
        monitors: list[dict[str, int]] | None = None,
    ) -> None:
        self.image = image
        self.monitors: Sequence[Mapping[str, int]] = (
            monitors if monitors is not None else [_monitor_for(image)]
        )
        self.grabbed_monitor: Mapping[str, int] | None = None
        self.entered = False
        self.exited = False

    def __enter__(self) -> MSSClientLike:
        self.entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _ = (exc_type, exc, traceback)
        self.exited = True

    def grab(self, monitor: Mapping[str, int]) -> _FakeShot:
        self.grabbed_monitor = monitor
        return _FakeShot(self.image)


def _monitor_for(image: Image.Image, *, left: int = 0, top: int = 0) -> dict[str, int]:
    width, height = image.size
    return {"left": left, "top": top, "width": width, "height": height}


def _solid_image(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), color=(12, 34, 56))


def _factory(fake: MSSClientLike) -> MSSFactory:
    return lambda: fake


class TestScreenshotCapture:
    def test_capture_returns_png_base64(self) -> None:
        image = _solid_image(320, 180)
        fake = _FakeMSS(image)
        capture = ScreenshotCapture(VisionConfig(), mss_factory=_factory(fake))

        frame = capture.capture()

        assert frame.format == "png"
        assert frame.width == 320
        assert frame.height == 180
        assert frame.original_width == 320
        assert frame.original_height == 180
        assert frame.resized is False
        assert fake.entered is True
        assert fake.exited is True
        assert fake.grabbed_monitor == fake.monitors[0]
        assert base64.b64decode(frame.image_base64).startswith(b"\x89PNG\r\n\x1a\n")

    def test_capture_uses_requested_monitor(self) -> None:
        image = _solid_image(640, 360)
        monitors = [
            _monitor_for(image),
            _monitor_for(image, left=1920, top=0),
        ]
        fake = _FakeMSS(image, monitors=monitors)
        capture = ScreenshotCapture(VisionConfig(), mss_factory=_factory(fake))

        frame = capture.capture(monitor_index=1)

        assert fake.grabbed_monitor == monitors[1]
        assert frame.left == 1920
        assert frame.top == 0

    def test_capture_resizes_by_long_edge(self) -> None:
        image = _solid_image(2000, 1000)
        fake = _FakeMSS(image)
        capture = ScreenshotCapture(
            VisionConfig(screenshot_max_long_edge=1000, screenshot_max_pixels=10_000_000),
            mss_factory=_factory(fake),
        )

        frame = capture.capture()

        assert frame.width == 1000
        assert frame.height == 500
        assert frame.resized is True
        assert frame.scale_x == pytest.approx(2.0)
        assert frame.scale_y == pytest.approx(2.0)

    def test_capture_resizes_by_max_pixels(self) -> None:
        image = _solid_image(2000, 1000)
        fake = _FakeMSS(image)
        capture = ScreenshotCapture(
            VisionConfig(screenshot_max_long_edge=4096, screenshot_max_pixels=500_000),
            mss_factory=_factory(fake),
        )

        frame = capture.capture()

        assert frame.width == 1000
        assert frame.height == 500

    def test_capture_preserves_virtual_desktop_offset(self) -> None:
        image = _solid_image(2000, 1000)
        fake = _FakeMSS(image, monitors=[_monitor_for(image, left=-1920, top=40)])
        capture = ScreenshotCapture(
            VisionConfig(screenshot_max_long_edge=1000, screenshot_max_pixels=10_000_000),
            mss_factory=_factory(fake),
        )

        frame = capture.capture()

        assert frame.left == -1920
        assert frame.top == 40
        assert frame.scale_x == pytest.approx(2.0)
        assert frame.scale_y == pytest.approx(2.0)

    def test_capture_rejects_invalid_monitor_index(self) -> None:
        fake = _FakeMSS(_solid_image(100, 100))
        capture = ScreenshotCapture(
            VisionConfig(),
            mss_factory=_factory(fake),
        )

        with pytest.raises(ValueError, match="Index moniteur invalide"):
            capture.capture(monitor_index=99)

    def test_capture_rejects_missing_monitors(self) -> None:
        fake = _FakeMSS(_solid_image(100, 100), monitors=[])
        capture = ScreenshotCapture(
            VisionConfig(),
            mss_factory=_factory(fake),
        )

        with pytest.raises(RuntimeError, match="Aucun moniteur"):
            capture.capture()
