from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ZoneState(str, Enum):
    OUTSIDE = "OUTSIDE"
    ENTERING = "ENTERING"
    INSIDE = "INSIDE"
    LEAVING = "LEAVING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class Keypoint:
    name: str
    x: float
    y: float
    score: float


@dataclass(frozen=True, slots=True)
class PoseObservation:
    frame_seq: int
    timestamp_sec: float
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: tuple[Keypoint, ...]
    frame_width: int
    frame_height: int


@dataclass(frozen=True, slots=True)
class WorkArea:
    area_id: str
    polygon: tuple[tuple[float, float], ...]
    expected_worker_count: int | None = None
    required_team_work_duration_sec: float | None = None

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError("work-area polygon must have at least three points")
        if any(not isfinite(value) for point in self.polygon for value in point):
            raise ValueError("work-area polygon coordinates must be finite")
        if any(not 0 <= value <= 1 for point in self.polygon for value in point):
            raise ValueError("work-area polygon coordinates must be normalized")
        if self.expected_worker_count is not None and self.expected_worker_count < 0:
            raise ValueError("expected_worker_count must be non-negative")
        if (
            self.required_team_work_duration_sec is not None
            and self.required_team_work_duration_sec < 0
        ):
            raise ValueError("required_team_work_duration_sec must be non-negative")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    entry_detection_duration_sec: float = 3.0
    exit_detection_duration_sec: float = 5.0
    lost_track_tolerance_sec: float = 2.0
    keypoint_confidence: float = 0.25
    max_frame_gap_sec: float = 2.0
    team_exit_grace_sec: float = 20 * 60.0

    def __post_init__(self) -> None:
        values = (
            self.entry_detection_duration_sec,
            self.exit_detection_duration_sec,
            self.lost_track_tolerance_sec,
            self.max_frame_gap_sec,
            self.team_exit_grace_sec,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("duration settings must be finite and non-negative")
        if not 0 <= self.keypoint_confidence <= 1:
            raise ValueError("keypoint_confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class CalibrationPointPair:
    camera_point: tuple[float, float]
    scene_point_px: tuple[float, float]

    def __post_init__(self) -> None:
        values = (*self.camera_point, *self.scene_point_px)
        if any(not isfinite(value) for value in values):
            raise ValueError("calibration point coordinates must be finite")


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    homography: tuple[tuple[float, float, float], ...]
    mean_error_px: float
    max_error_px: float


@dataclass(frozen=True, slots=True)
class ComplianceInterval:
    start_sec: float
    end_sec: float | None
    headcount_compliant: bool
