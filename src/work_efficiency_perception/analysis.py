from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, RLock
from typing import Protocol

from .api.repository import ConfigurationRecord, InMemoryRepository, TaskRecord
from .engine import WorkEfficiencyEngine
from .models import ComplianceInterval, EngineConfig, PoseObservation, WorkArea
from .tracker import PoseTrackerConfig, YoloPoseByteTrackTracker
from .video import CameraStreamSource, FrameSource, VideoFileSource

HISTORICAL_VIDEO_SAMPLE_INTERVAL_SEC = 0.2


class FrameTracker(Protocol):
    def track_frame(
        self, image: object, *, frame_seq: int, timestamp_sec: float
    ) -> tuple[PoseObservation, ...]: ...


@dataclass(frozen=True, slots=True)
class FrameAnalysisResult:
    frame_seq: int
    timestamp_sec: float
    progress: float | None
    persons: tuple[PoseObservation, ...]
    captured_at_sec: float | None = None
    work_efficiency: dict | None = None
    frame_width: int | None = None
    frame_height: int | None = None


@dataclass(frozen=True, slots=True)
class TaskAnalysisSnapshot:
    processed_frames: int = 0
    analyzed_frames: int = 0
    progress: float | None = None
    latest_frame: FrameAnalysisResult | None = None
    compliance_intervals: dict[str, tuple[ComplianceInterval, ...]] | None = None


class AnalysisResultStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._snapshots: dict[str, TaskAnalysisSnapshot] = {}
        self._subscribers: dict[str, set[Queue[FrameAnalysisResult]]] = {}

    def update(
        self,
        task_id: str,
        frame: FrameAnalysisResult,
        compliance_intervals: dict[str, tuple[ComplianceInterval, ...]] | None = None,
    ) -> None:
        with self._lock:
            previous = self._snapshots.get(task_id, TaskAnalysisSnapshot())
            self._snapshots[task_id] = TaskAnalysisSnapshot(
                processed_frames=frame.frame_seq + 1,
                analyzed_frames=previous.analyzed_frames + 1,
                progress=frame.progress,
                latest_frame=frame,
                compliance_intervals=compliance_intervals,
            )
            for subscriber in self._subscribers.get(task_id, set()):
                try:
                    subscriber.put_nowait(frame)
                except Full:
                    try:
                        subscriber.get_nowait()
                    except Empty:
                        pass
                    subscriber.put_nowait(frame)

    def complete(self, task_id: str, processed_frames: int) -> None:
        """Normalize progress after a finite source has been fully consumed."""
        with self._lock:
            snapshot = self._snapshots.get(task_id)
            if snapshot is None:
                self._snapshots[task_id] = TaskAnalysisSnapshot(
                    processed_frames=processed_frames,
                    progress=1.0,
                )
                return
            latest = (
                None
                if snapshot.latest_frame is None
                else replace(snapshot.latest_frame, progress=1.0)
            )
            self._snapshots[task_id] = replace(
                snapshot,
                processed_frames=processed_frames,
                progress=1.0,
                latest_frame=latest,
            )

    def get(self, task_id: str) -> TaskAnalysisSnapshot:
        with self._lock:
            return self._snapshots.get(task_id, TaskAnalysisSnapshot())

    def subscribe(self, task_id: str) -> ResultSubscription:
        queue: Queue[FrameAnalysisResult] = Queue(maxsize=1)
        with self._lock:
            self._subscribers.setdefault(task_id, set()).add(queue)
        return ResultSubscription(self, task_id, queue)

    def _unsubscribe(self, task_id: str, queue: Queue[FrameAnalysisResult]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(task_id)
            if subscribers is None:
                return
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(task_id, None)


class ResultSubscription:
    def __init__(
        self,
        store: AnalysisResultStore,
        task_id: str,
        queue: Queue[FrameAnalysisResult],
    ) -> None:
        self._store = store
        self._task_id = task_id
        self._queue = queue
        self._closed = False

    def receive(self, timeout: float = 1.0) -> FrameAnalysisResult | None:
        if self._closed:
            return None
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def close(self) -> None:
        if not self._closed:
            self._store._unsubscribe(self._task_id, self._queue)
            self._closed = True


class MediaAnalysisWorker:
    """Run the same pose pipeline for historical files and live camera streams."""

    def __init__(
        self,
        repository: InMemoryRepository,
        tracker_factory: Callable[[ConfigurationRecord], FrameTracker],
        result_store: AnalysisResultStore,
        *,
        video_source_factory: Callable[[str | Path], FrameSource] = VideoFileSource,
        camera_source_factory: Callable[[str], FrameSource] = CameraStreamSource,
    ) -> None:
        self._repository = repository
        self._tracker_factory = tracker_factory
        self._result_store = result_store
        self._video_source_factory = video_source_factory
        self._camera_source_factory = camera_source_factory

    def run(
        self,
        task: TaskRecord,
        configuration: ConfigurationRecord,
        stop_event: Event,
    ) -> None:
        source = self._resolve_source(task)
        tracker = self._tracker_factory(configuration)
        engine = _build_efficiency_engine(configuration)
        with source:
            total_frames = source.metadata.frame_count
            is_historical = not source.metadata.is_live
            next_sample_timestamp = 0.0
            decoded_frames = 0
            last_decoded_frame = None
            last_analyzed_sequence = -1
            for frame in source.iter_frames(stop_event):
                decoded_frames = frame.sequence + 1
                last_decoded_frame = frame
                if is_historical and frame.timestamp_sec + 1e-9 < next_sample_timestamp:
                    continue
                self._analyze_frame(task, source, tracker, engine, frame, total_frames)
                last_analyzed_sequence = frame.sequence
                if is_historical:
                    next_sample_timestamp = (
                        frame.timestamp_sec + HISTORICAL_VIDEO_SAMPLE_INTERVAL_SEC
                    )

            if (
                is_historical
                and not stop_event.is_set()
                and last_decoded_frame is not None
                and last_decoded_frame.sequence != last_analyzed_sequence
            ):
                self._analyze_frame(
                    task, source, tracker, engine, last_decoded_frame, total_frames
                )
            if is_historical and not stop_event.is_set():
                self._result_store.complete(task.task_id, decoded_frames)

    def _analyze_frame(
        self,
        task: TaskRecord,
        source: FrameSource,
        tracker: FrameTracker,
        engine: WorkEfficiencyEngine,
        frame: object,
        total_frames: int | None,
    ) -> None:
        captured_at_sec = frame.captured_at_sec
        if (task.input_type in {"VIDEO_FILE", "HISTORICAL_VIDEO"}
                and task.recording_started_at is not None):
            captured_at_sec = datetime.fromisoformat(task.recording_started_at).timestamp() + frame.timestamp_sec
        elif captured_at_sec is None and task.video_id is not None:
            video = self._repository.get_video(task.video_id)
            if video is not None and video.captured_at is not None:
                captured_at = datetime.fromisoformat(video.captured_at)
                if captured_at.tzinfo is None:
                    captured_at = captured_at.replace(tzinfo=timezone.utc)
                captured_at_sec = captured_at.timestamp() + frame.timestamp_sec
        persons = tracker.track_frame(
            frame.image,
            frame_seq=frame.sequence,
            timestamp_sec=frame.timestamp_sec,
        )
        work_efficiency = engine.process(persons, frame.timestamp_sec)
        for person in work_efficiency["persons"]:
            point = person["scenePointNorm"]
            person["trajectoryDelta"] = (
                []
                if point is None
                else [{"seq": frame.sequence, "x": point["x"], "y": point["y"]}]
            )
        progress = (
            min((frame.sequence + 1) / total_frames, 1.0) if total_frames else None
        )
        self._result_store.update(
            task.task_id,
            FrameAnalysisResult(
                frame.sequence,
                frame.timestamp_sec,
                progress,
                persons,
                captured_at_sec,
                work_efficiency,
                source.metadata.width,
                source.metadata.height,
            ),
            {
                area.area_id: engine.compliance_intervals(area.area_id)
                for area in engine.areas
            },
        )

    def _resolve_source(self, task: TaskRecord) -> FrameSource:
        if task.input_type in {"VIDEO_FILE", "HISTORICAL_VIDEO"} and task.video_id is not None:
            video = self._repository.get_video(task.video_id)
            if video is None:
                raise FileNotFoundError(f"video is not registered: {task.video_id}")
            return self._video_source_factory(video.path)
        if task.input_type == "CAMERA_STREAM" and task.camera_id is not None:
            camera = self._repository.get_camera(task.camera_id)
            if camera is None:
                raise FileNotFoundError(f"camera is not registered: {task.camera_id}")
            return self._camera_source_factory(camera.stream_url)
        raise ValueError("task input source is invalid")


class VideoAnalysisWorker(MediaAnalysisWorker):
    """Backward-compatible name; now supports both configured input types."""


def _build_efficiency_engine(configuration: ConfigurationRecord) -> WorkEfficiencyEngine:
    try:
        request = configuration.payload["request"]
        homography = configuration.payload["homography"]
        areas = tuple(
            WorkArea(
                item["areaId"],
                tuple((point["x"], point["y"]) for point in item["polygon"]),
                item.get("expectedWorkerCount"),
                item.get("requiredTeamWorkDurationSec"),
            )
            for item in request["workAreas"]
        )
        algorithm = request.get("algorithm", {})
        config = EngineConfig(
            entry_detection_duration_sec=algorithm.get("entryDetectionDurationSec", 3),
            exit_detection_duration_sec=algorithm.get("exitDetectionDurationSec", 5),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("configuration payload cannot build efficiency engine") from error
    return WorkEfficiencyEngine(areas, homography, config)


def build_yolo_tracker_factory(
    config: PoseTrackerConfig,
) -> Callable[[ConfigurationRecord], YoloPoseByteTrackTracker]:
    def create_tracker(configuration: ConfigurationRecord) -> YoloPoseByteTrackTracker:
        request = configuration.payload.get("request", {})
        person_confidence = request.get("algorithm", {}).get(
            "personConfidence", config.confidence_threshold
        )
        return YoloPoseByteTrackTracker.from_config(
            replace(config, confidence_threshold=float(person_confidence))
        )

    return create_tracker
