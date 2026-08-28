from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numpy.typing import ArrayLike

from .models import Keypoint, PoseObservation

COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


@dataclass(frozen=True, slots=True)
class PoseTrackerConfig:
    model_path: Path
    confidence_threshold: float = 0.5
    tracker_config: str = "bytetrack.yaml"
    device: str | int | None = None
    image_size: int = 1280

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero")


class YoloPoseByteTrackTracker:
    """Adapter for an Ultralytics YOLO pose model with ByteTrack IDs."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.5,
        tracker_config: str = "bytetrack.yaml",
        device: str | int | None = None,
        image_size: int = 1280,
        model: Any | None = None,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker_config
        self.device = device
        self.image_size = image_size
        self._model = model
        if image_size <= 0:
            raise ValueError("image_size must be greater than zero")

    @classmethod
    def from_config(
        cls, config: PoseTrackerConfig, *, model: Any | None = None
    ) -> YoloPoseByteTrackTracker:
        return cls(
            config.model_path,
            confidence_threshold=config.confidence_threshold,
            tracker_config=config.tracker_config,
            device=config.device,
            image_size=config.image_size,
            model=model,
        )

    @property
    def model(self) -> Any:
        if self._model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"YOLO pose model not found: {self.model_path}")
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
            if self._model.task != "pose":
                raise ValueError(f"YOLO model is not a pose model: {self.model_path}")
        return self._model

    def track_frame(
        self, image: ArrayLike, *, frame_seq: int, timestamp_sec: float
    ) -> tuple[PoseObservation, ...]:
        kwargs: dict[str, Any] = {
            "source": image,
            "stream": False,
            "show": False,
            "tracker": self.tracker_config,
            "persist": True,
            "classes": [0],
            "conf": self.confidence_threshold,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.track(**kwargs)
        result = results[0] if results else None
        if result is None:
            return ()
        return self.convert_result(result, frame_seq=frame_seq, timestamp_sec=timestamp_sec)

    def iter_video(self, video_path: str | Path, *, fps: float) -> Iterator[tuple[PoseObservation, ...]]:
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        source = Path(video_path)
        if not source.is_file():
            raise FileNotFoundError(f"video not found: {source}")
        kwargs: dict[str, Any] = {
            "source": str(source), "stream": True, "show": False,
            "tracker": self.tracker_config, "persist": True,
            "classes": [0], "conf": self.confidence_threshold, "verbose": False,
            "imgsz": self.image_size,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        for frame_seq, result in enumerate(self.model.track(**kwargs)):
            yield self.convert_result(result, frame_seq=frame_seq, timestamp_sec=frame_seq / fps)

    def convert_result(self, result: Any, *, frame_seq: int, timestamp_sec: float) -> tuple[PoseObservation, ...]:
        boxes, poses = result.boxes, result.keypoints
        if boxes is None or boxes.id is None or poses is None or len(boxes) == 0:
            return ()
        xyxy = boxes.xyxy.cpu().tolist()
        ids = boxes.id.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        coordinates = poses.xy.cpu().tolist()
        scores = poses.conf.cpu().tolist() if poses.conf is not None else [[1.0] * len(COCO_KEYPOINT_NAMES) for _ in coordinates]
        height, width = result.orig_shape
        observations = []
        for bbox, track_id, confidence, points, point_scores in zip(xyxy, ids, confidences, coordinates, scores):
            keypoints = tuple(
                Keypoint(name, float(xy[0]), float(xy[1]), float(score))
                for name, xy, score in zip(COCO_KEYPOINT_NAMES, points, point_scores)
            )
            observations.append(PoseObservation(
                frame_seq, timestamp_sec, int(track_id), tuple(map(float, bbox)),
                float(confidence), keypoints, int(width), int(height),
            ))
        return tuple(observations)
