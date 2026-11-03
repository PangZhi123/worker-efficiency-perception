from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Point(ApiModel):
    x: float
    y: float


class FeaturePairRequest(ApiModel):
    camera_point: Point = Field(alias="cameraPoint")
    scene_point: Point = Field(alias="scenePoint")


class WorkAreaRequest(ApiModel):
    area_id: Annotated[str, Field(min_length=1, max_length=64)] = Field(alias="areaId")
    coordinate_type: Literal["NORMALIZED"] = Field(default="NORMALIZED", alias="coordinateType")
    polygon: Annotated[list[Point], Field(min_length=3)]
    expected_worker_count: Annotated[int | None, Field(ge=0)] = Field(
        default=None, alias="expectedWorkerCount"
    )
    required_team_work_duration_sec: Annotated[float | None, Field(ge=0)] = Field(
        default=None, alias="requiredTeamWorkDurationSec"
    )


class AlgorithmRequest(ApiModel):
    person_confidence: Annotated[float, Field(ge=0, le=1)] = Field(
        default=0.5, alias="personConfidence"
    )
    entry_detection_duration_sec: Annotated[float, Field(ge=0)] = Field(
        default=3, alias="entryDetectionDurationSec"
    )
    exit_detection_duration_sec: Annotated[float, Field(ge=0)] = Field(
        default=5, alias="exitDetectionDurationSec"
    )


class StorageRequest(ApiModel):
    pose_sampling_ms: Annotated[int, Field(ge=0)] = Field(default=1000, alias="poseSamplingMs")


class ConfigurationRequest(ApiModel):
    camera_id: Annotated[str, Field(min_length=1, max_length=64)] = Field(alias="cameraId")
    video_id: Annotated[str | None, Field(min_length=1, max_length=64)] = Field(
        default=None, alias="videoId"
    )
    scene_id: Annotated[str, Field(min_length=1, max_length=64)] = Field(alias="sceneId")
    feature_pairs: Annotated[list[FeaturePairRequest], Field(min_length=4)] = Field(
        alias="featurePairs"
    )
    work_areas: Annotated[list[WorkAreaRequest], Field(min_length=1)] = Field(alias="workAreas")
    algorithm: AlgorithmRequest = Field(default_factory=AlgorithmRequest)
    storage: StorageRequest = Field(default_factory=StorageRequest)


class TaskOptionsRequest(ApiModel):
    persist_pose_and_trajectory: bool = Field(default=True, alias="persistPoseAndTrajectory")


class RecordingStartRequest(ApiModel):
    recording_started_at: datetime | None = Field(default=None, alias="recordingStartedAt")

    @field_validator("recording_started_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("recordingStartedAt requires a timezone")
        return value


class TaskCreateRequest(RecordingStartRequest):
    input_type: Literal["CAMERA_STREAM", "VIDEO_FILE"] = Field(alias="inputType")
    camera_id: Annotated[str | None, Field(min_length=1, max_length=64)] = Field(
        default=None, alias="cameraId"
    )
    video_id: Annotated[str | None, Field(min_length=1, max_length=64)] = Field(
        default=None, alias="videoId"
    )
    configuration_id: Annotated[str, Field(min_length=1, max_length=64)] = Field(
        alias="configurationId"
    )
    options: TaskOptionsRequest = Field(default_factory=TaskOptionsRequest)


class TaskActionRequest(ApiModel):
    action: Literal["STOP"]


class HistoricalFeaturePairRequest(ApiModel):
    video_point: Point = Field(alias="videoPoint")
    scene_point: Point = Field(alias="scenePoint")


class HistoricalConfigurationRequest(ApiModel):
    calibration_frame_timestamp_sec: Annotated[float, Field(ge=0)] = Field(
        alias="calibrationFrameTimestampSec"
    )
    feature_pairs: Annotated[list[HistoricalFeaturePairRequest], Field(min_length=4)] = Field(
        alias="featurePairs"
    )
    work_areas: Annotated[list[WorkAreaRequest], Field(min_length=1)] = Field(alias="workAreas")
    algorithm: AlgorithmRequest = Field(default_factory=AlgorithmRequest)
    storage: StorageRequest = Field(default_factory=StorageRequest)


class HistoricalTaskCreateRequest(RecordingStartRequest):
    options: TaskOptionsRequest = Field(default_factory=TaskOptionsRequest)


class CameraRegistrationRequest(ApiModel):
    camera_id: Annotated[str, Field(min_length=1, max_length=64)] = Field(alias="cameraId")
    stream_url: Annotated[str, Field(min_length=1, max_length=2048)] = Field(alias="streamUrl")


class ErrorData(ApiModel):
    details: list[dict] = Field(default_factory=list)
