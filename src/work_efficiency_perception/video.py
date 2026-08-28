from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, RLock, Thread, current_thread
from time import monotonic, time
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FrameSourceMetadata:
    source: str
    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_sec: float | None
    is_live: bool


@dataclass(frozen=True, slots=True)
class VideoFrame:
    sequence: int
    timestamp_sec: float
    image: NDArray[np.uint8]
    captured_at_sec: float | None = None


class FrameSource(Protocol):
    @property
    def metadata(self) -> FrameSourceMetadata: ...

    def __enter__(self) -> FrameSource: ...  # noqa: PYI034

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...

    def __iter__(self) -> Iterator[VideoFrame]: ...

    def iter_frames(self, stop_event: Event | None = None) -> Iterator[VideoFrame]: ...

    def close(self) -> None: ...


class VideoFileSource:
    """OpenCV-backed video reader with timestamps based on video time."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._capture: cv2.VideoCapture | None = None
        self._metadata: FrameSourceMetadata | None = None

    @property
    def metadata(self) -> FrameSourceMetadata:
        if self._metadata is None:
            raise RuntimeError("video source is not open")
        return self._metadata

    def __enter__(self) -> VideoFileSource:  # noqa: PYI034 - Python 3.10 compatibility
        if not self.path.is_file():
            raise FileNotFoundError(f"video not found: {self.path}")
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"video cannot be opened: {self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or width <= 0 or height <= 0 or frame_count < 0:
            capture.release()
            raise ValueError(f"video metadata is invalid: {self.path}")
        self._capture = capture
        self._metadata = FrameSourceMetadata(
            str(self.path),
            width,
            height,
            fps,
            frame_count,
            frame_count / fps if frame_count else 0.0,
            False,
        )
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __iter__(self):
        return self.iter_frames()

    def iter_frames(self, stop_event: Event | None = None) -> Iterator[VideoFrame]:
        if self._capture is None:
            raise RuntimeError("video source is not open")
        sequence = 0
        while stop_event is None or not stop_event.is_set():
            success, image = self._capture.read()
            if not success:
                break
            reported_msec = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            fallback_sec = sequence / self.metadata.fps
            timestamp_sec = reported_msec / 1000.0 if reported_msec > 0 else fallback_sec
            yield VideoFrame(sequence, timestamp_sec, image)
            sequence += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None


class CameraStreamSource:
    """Live stream reader that drops stale frames and reconnects after failures."""

    def __init__(
        self,
        stream_url: str,
        *,
        reconnect_delay_sec: float = 1.0,
        open_timeout_sec: float = 10.0,
        capture_factory: Callable[[str], Any] = cv2.VideoCapture,
        monotonic_clock: Callable[[], float] = monotonic,
        wall_clock: Callable[[], float] = time,
    ) -> None:
        if not stream_url:
            raise ValueError("stream_url must not be blank")
        if reconnect_delay_sec < 0 or open_timeout_sec <= 0:
            raise ValueError("stream timing settings are invalid")
        self.stream_url = stream_url
        self.reconnect_delay_sec = reconnect_delay_sec
        self.open_timeout_sec = open_timeout_sec
        self._capture_factory = capture_factory
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._stop_event = Event()
        self._condition = Condition(RLock())
        self._metadata: FrameSourceMetadata | None = None
        self._latest_frame: VideoFrame | None = None
        self._latest_version = 0
        self._capture: Any | None = None
        self._thread: Thread | None = None

    @property
    def metadata(self) -> FrameSourceMetadata:
        if self._metadata is None:
            raise RuntimeError("camera stream is not connected")
        return self._metadata

    def __enter__(self) -> CameraStreamSource:  # noqa: PYI034 - Python 3.10 compatibility
        self._stop_event.clear()
        self._thread = Thread(target=self._capture_loop, name="camera-stream", daemon=True)
        self._thread.start()
        deadline = self._monotonic_clock() + self.open_timeout_sec
        with self._condition:
            while self._metadata is None and not self._stop_event.is_set():
                remaining = deadline - self._monotonic_clock()
                if remaining <= 0:
                    self.close()
                    raise ConnectionError(f"camera stream cannot be opened: {self.stream_url}")
                self._condition.wait(timeout=min(remaining, 0.1))
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[VideoFrame]:
        return self.iter_frames()

    def iter_frames(self, stop_event: Event | None = None) -> Iterator[VideoFrame]:
        version = 0
        while not self._stop_event.is_set() and (stop_event is None or not stop_event.is_set()):
            with self._condition:
                self._condition.wait_for(
                    lambda version=version: (
                        self._latest_version > version
                        or self._stop_event.is_set()
                        or (stop_event is not None and stop_event.is_set())
                    ),
                    timeout=0.5,
                )
                if self._stop_event.is_set() or (stop_event is not None and stop_event.is_set()):
                    break
                frame = self._latest_frame
                version = self._latest_version
            if frame is not None:
                yield frame

    def close(self) -> None:
        self._stop_event.set()
        capture = self._capture
        if capture is not None:
            capture.release()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=2)
        self._thread = None
        self._capture = None

    def _capture_loop(self) -> None:
        sequence = 0
        while not self._stop_event.is_set():
            capture = self._capture_factory(self.stream_url)
            self._capture = capture
            if not capture.isOpened():
                capture.release()
                self._stop_event.wait(self.reconnect_delay_sec)
                continue
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            with self._condition:
                self._metadata = FrameSourceMetadata(
                    self.stream_url, width, height, max(fps, 0.0), None, None, True
                )
                self._condition.notify_all()
            while not self._stop_event.is_set():
                success, image = capture.read()
                if not success:
                    break
                frame = VideoFrame(
                    sequence,
                    self._monotonic_clock(),
                    image,
                    self._wall_clock(),
                )
                sequence += 1
                with self._condition:
                    self._latest_frame = frame
                    self._latest_version += 1
                    self._condition.notify_all()
            capture.release()
            self._capture = None
            self._stop_event.wait(self.reconnect_delay_sec)


# Backward-compatible name used by existing callers.
VideoSource = VideoFileSource
VideoMetadata = FrameSourceMetadata
