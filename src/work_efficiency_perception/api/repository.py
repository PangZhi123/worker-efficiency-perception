from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock


@dataclass(frozen=True, slots=True)
class SceneRecord:
    scene_id: str
    scene_name: str
    site_id: str
    image_url: str
    width: int
    height: int
    media_type: str
    content: bytes
    analysis_id: str | None = None


@dataclass(frozen=True, slots=True)
class VideoRecord:
    video_id: str
    path: str
    camera_id: str | None
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    analysis_id: str | None = None


@dataclass(frozen=True, slots=True)
class CameraRecord:
    camera_id: str
    stream_url: str


@dataclass(frozen=True, slots=True)
class ConfigurationRecord:
    configuration_id: str
    calibration_version: str
    area_version: str
    payload: dict
    configuration_type: str = "REALTIME"
    analysis_id: str | None = None


class HistoricalPreparationStatus(str, Enum):
    EMPTY = "EMPTY"
    VIDEO_UPLOADED = "VIDEO_UPLOADED"
    ASSETS_READY = "ASSETS_READY"
    READY = "READY"
    LOCKED = "LOCKED"


@dataclass(frozen=True, slots=True)
class HistoricalAnalysisRecord:
    analysis_id: str
    preparation_status: HistoricalPreparationStatus
    created_at: str
    updated_at: str
    video_id: str | None = None
    scene_id: str | None = None
    configuration_id: str | None = None
    latest_task_id: str | None = None


class TaskStatus(str, Enum):
    CREATED = "CREATED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


TERMINAL_TASK_STATUSES = {TaskStatus.STOPPED, TaskStatus.COMPLETED, TaskStatus.FAILED}


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    input_type: str
    camera_id: str | None
    video_id: str | None
    configuration_id: str
    persist_pose_and_trajectory: bool
    status: TaskStatus
    created_at: str
    updated_at: str
    failure_code: str | None = None
    failure_message: str | None = None
    analysis_id: str | None = None


class InMemoryRepository:
    """Thread-safe development repository; replace with a database adapter later."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._scenes: dict[str, SceneRecord] = {}
        self._videos: dict[str, VideoRecord] = {}
        self._cameras: dict[str, CameraRecord] = {}
        self._configurations: dict[str, ConfigurationRecord] = {}
        self._historical_configurations: dict[str, ConfigurationRecord] = {}
        self._tasks: dict[str, TaskRecord] = {}
        self._historical_analyses: dict[str, HistoricalAnalysisRecord] = {}

    def add_scene(self, scene: SceneRecord) -> None:
        with self._lock:
            self._scenes[scene.scene_id] = scene

    def get_scene(self, scene_id: str) -> SceneRecord | None:
        with self._lock:
            return self._scenes.get(scene_id)

    def remove_scene(self, scene_id: str) -> None:
        with self._lock:
            self._scenes.pop(scene_id, None)

    def add_video(self, video: VideoRecord) -> None:
        with self._lock:
            self._videos[video.video_id] = video

    def get_video(self, video_id: str) -> VideoRecord | None:
        with self._lock:
            return self._videos.get(video_id)

    def remove_video(self, video_id: str) -> None:
        with self._lock:
            self._videos.pop(video_id, None)

    def add_camera(self, camera: CameraRecord) -> None:
        with self._lock:
            self._cameras[camera.camera_id] = camera

    def get_camera(self, camera_id: str) -> CameraRecord | None:
        with self._lock:
            return self._cameras.get(camera_id)

    def add_configuration(self, configuration: ConfigurationRecord) -> None:
        if configuration.configuration_type != "REALTIME":
            raise ValueError("realtime configuration repository only accepts REALTIME")
        with self._lock:
            self._configurations[configuration.configuration_id] = configuration

    def get_configuration(self, configuration_id: str) -> ConfigurationRecord | None:
        """Resolve either configuration type for the shared task runtime only."""
        with self._lock:
            return self._configurations.get(configuration_id) or self._historical_configurations.get(
                configuration_id
            )

    def list_configurations(self) -> tuple[ConfigurationRecord, ...]:
        with self._lock:
            return tuple(self._configurations.values())

    def remove_configuration(self, configuration_id: str) -> None:
        with self._lock:
            self._configurations.pop(configuration_id, None)

    def add_historical_configuration(self, configuration: ConfigurationRecord) -> None:
        if configuration.configuration_type != "HISTORICAL":
            raise ValueError("historical configuration repository only accepts HISTORICAL")
        with self._lock:
            self._historical_configurations[configuration.configuration_id] = configuration

    def get_historical_configuration(
        self, configuration_id: str
    ) -> ConfigurationRecord | None:
        with self._lock:
            return self._historical_configurations.get(configuration_id)

    def remove_historical_configuration(self, configuration_id: str) -> None:
        with self._lock:
            self._historical_configurations.pop(configuration_id, None)

    def add_historical_analysis(self, analysis: HistoricalAnalysisRecord) -> None:
        with self._lock:
            if analysis.analysis_id in self._historical_analyses:
                raise ValueError(f"historical analysis already exists: {analysis.analysis_id}")
            self._historical_analyses[analysis.analysis_id] = analysis

    def get_historical_analysis(
        self, analysis_id: str
    ) -> HistoricalAnalysisRecord | None:
        with self._lock:
            return self._historical_analyses.get(analysis_id)

    def update_historical_analysis(
        self,
        analysis_id: str,
        *,
        expected: set[HistoricalPreparationStatus],
        updated_at: str,
        **changes: object,
    ) -> HistoricalAnalysisRecord | None:
        with self._lock:
            analysis = self._historical_analyses.get(analysis_id)
            if analysis is None:
                raise KeyError(analysis_id)
            if analysis.preparation_status not in expected:
                return None
            updated = replace(analysis, updated_at=updated_at, **changes)
            self._historical_analyses[analysis_id] = updated
            return updated

    def add_historical_task(
        self, analysis_id: str, task: TaskRecord, *, updated_at: str
    ) -> HistoricalAnalysisRecord:
        """Atomically lock a ready analysis and attach a new execution attempt."""
        with self._lock:
            analysis = self._historical_analyses.get(analysis_id)
            if analysis is None:
                raise KeyError(analysis_id)
            if analysis.preparation_status == HistoricalPreparationStatus.READY:
                pass
            elif analysis.preparation_status == HistoricalPreparationStatus.LOCKED:
                latest = (
                    self._tasks.get(analysis.latest_task_id)
                    if analysis.latest_task_id is not None
                    else None
                )
                if latest is None or latest.status not in {
                    TaskStatus.STOPPED,
                    TaskStatus.FAILED,
                }:
                    raise ValueError("historical analysis cannot be started again")
            else:
                raise ValueError("historical analysis is not ready")
            if task.task_id in self._tasks:
                raise ValueError(f"task already exists: {task.task_id}")
            self._tasks[task.task_id] = task
            updated = replace(
                analysis,
                preparation_status=HistoricalPreparationStatus.LOCKED,
                latest_task_id=task.task_id,
                updated_at=updated_at,
            )
            self._historical_analyses[analysis_id] = updated
            return updated

    def add_task(self, task: TaskRecord) -> None:
        with self._lock:
            if task.task_id in self._tasks:
                raise ValueError(f"task already exists: {task.task_id}")
            self._tasks[task.task_id] = task

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> tuple[TaskRecord, ...]:
        with self._lock:
            return tuple(self._tasks.values())

    def update_task(self, task: TaskRecord) -> None:
        with self._lock:
            if task.task_id not in self._tasks:
                raise KeyError(task.task_id)
            self._tasks[task.task_id] = task

    def transition_task(
        self,
        task_id: str,
        *,
        expected: set[TaskStatus],
        status: TaskStatus,
        updated_at: str,
        failure_code: str | None = None,
        failure_message: str | None = None,
    ) -> TaskRecord | None:
        """Atomically update a task only when its current state is expected."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            if task.status not in expected:
                return None
            updated = replace(
                task,
                status=status,
                updated_at=updated_at,
                failure_code=failure_code,
                failure_message=failure_message,
            )
            self._tasks[task_id] = updated
            return updated
