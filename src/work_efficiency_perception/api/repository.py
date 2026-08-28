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


@dataclass(frozen=True, slots=True)
class VideoRecord:
    video_id: str
    path: str
    camera_id: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


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


class InMemoryRepository:
    """Thread-safe development repository; replace with a database adapter later."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._scenes: dict[str, SceneRecord] = {}
        self._videos: dict[str, VideoRecord] = {}
        self._cameras: dict[str, CameraRecord] = {}
        self._configurations: dict[str, ConfigurationRecord] = {}
        self._tasks: dict[str, TaskRecord] = {}

    def add_scene(self, scene: SceneRecord) -> None:
        with self._lock:
            self._scenes[scene.scene_id] = scene

    def get_scene(self, scene_id: str) -> SceneRecord | None:
        with self._lock:
            return self._scenes.get(scene_id)

    def add_video(self, video: VideoRecord) -> None:
        with self._lock:
            self._videos[video.video_id] = video

    def get_video(self, video_id: str) -> VideoRecord | None:
        with self._lock:
            return self._videos.get(video_id)

    def add_camera(self, camera: CameraRecord) -> None:
        with self._lock:
            self._cameras[camera.camera_id] = camera

    def get_camera(self, camera_id: str) -> CameraRecord | None:
        with self._lock:
            return self._cameras.get(camera_id)

    def add_configuration(self, configuration: ConfigurationRecord) -> None:
        with self._lock:
            self._configurations[configuration.configuration_id] = configuration

    def get_configuration(self, configuration_id: str) -> ConfigurationRecord | None:
        with self._lock:
            return self._configurations.get(configuration_id)

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
