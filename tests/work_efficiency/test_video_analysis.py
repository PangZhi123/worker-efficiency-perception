from pathlib import Path
from queue import Queue
from threading import Event
from time import monotonic, sleep

import cv2
import numpy as np
import pytest

from work_efficiency_perception.analysis import (
    AnalysisResultStore,
    FrameAnalysisResult,
    VideoAnalysisWorker,
    build_yolo_tracker_factory,
)
from work_efficiency_perception.api.repository import (
    CameraRecord,
    ConfigurationRecord,
    InMemoryRepository,
    TaskRecord,
    TaskStatus,
    VideoRecord,
)
from work_efficiency_perception.models import Keypoint, PoseObservation
from work_efficiency_perception.tracker import PoseTrackerConfig, YoloPoseByteTrackTracker
from work_efficiency_perception.video import (
    CameraStreamSource,
    FrameSourceMetadata,
    VideoFrame,
    VideoSource,
)


def make_video(path: Path, *, frame_count: int = 4, fps: float = 2) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 24)
    )
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()


def configuration_record() -> ConfigurationRecord:
    return ConfigurationRecord(
        "cfg_1",
        "cal_1",
        "area_1",
        {
            "homography": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            "request": {
                "workAreas": [
                    {
                        "areaId": "zone",
                        "polygon": [
                            {"x": 0, "y": 0},
                            {"x": 1, "y": 0},
                            {"x": 1, "y": 1},
                            {"x": 0, "y": 1},
                        ],
                        "expectedWorkerCount": 1,
                        "requiredTeamWorkDurationSec": 10,
                    }
                ],
                "algorithm": {
                    "entryDetectionDurationSec": 0,
                    "exitDetectionDurationSec": 0,
                },
            },
        },
    )


def test_video_source_reads_metadata_frames_and_video_timestamps(tmp_path):
    path = tmp_path / "sample.avi"
    make_video(path)

    with VideoSource(path) as source:
        frames = list(source)
        metadata = source.metadata

    assert (metadata.width, metadata.height, metadata.frame_count) == (32, 24, 4)
    assert metadata.fps == pytest.approx(2)
    assert metadata.duration_sec == pytest.approx(2)
    assert [frame.sequence for frame in frames] == [0, 1, 2, 3]
    assert [frame.timestamp_sec for frame in frames] == pytest.approx([0, 0.5, 1, 1.5])


class EmptyModel:
    def __init__(self) -> None:
        self.kwargs = None

    def track(self, **kwargs):
        self.kwargs = kwargs
        return []


def test_tracker_tracks_single_frames_with_persistent_bytetrack():
    model = EmptyModel()
    tracker = YoloPoseByteTrackTracker(
        "unused.pt", model=model, device=0, image_size=960, confidence_threshold=0.4
    )

    result = tracker.track_frame(np.zeros((24, 32, 3)), frame_seq=7, timestamp_sec=1.5)

    assert result == ()
    assert model.kwargs["persist"] is True
    assert model.kwargs["tracker"] == "bytetrack.yaml"
    assert model.kwargs["imgsz"] == 960
    assert model.kwargs["device"] == 0


def test_tracker_factory_applies_configuration_person_confidence():
    configuration = configuration_record()
    configuration.payload["request"]["algorithm"]["personConfidence"] = 0.73
    factory = build_yolo_tracker_factory(PoseTrackerConfig(Path("unused.pt")))

    tracker = factory(configuration)

    assert tracker.confidence_threshold == pytest.approx(0.73)


class EmptyTracker:
    def track_frame(self, image, *, frame_seq, timestamp_sec):
        return ()


class RecordingTracker:
    def __init__(self):
        self.sequences = []

    def track_frame(self, image, *, frame_seq, timestamp_sec):
        self.sequences.append(frame_seq)
        return ()


class FakeHistoricalSource:
    metadata = FrameSourceMetadata("video.mp4", 32, 24, 10, 10, 1, False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def iter_frames(self, stop_event=None):
        for sequence in range(10):
            yield VideoFrame(
                sequence,
                sequence / 10,
                np.zeros((24, 32, 3), dtype=np.uint8),
            )

    def close(self):
        pass


class InsideWorkerTracker:
    def track_frame(self, image, *, frame_seq, timestamp_sec):
        points = (
            Keypoint("left_ankle", 0.5, 0.5, 0.9),
            Keypoint("right_ankle", 0.5, 0.5, 0.9),
        )
        return (
            PoseObservation(
                frame_seq,
                timestamp_sec,
                1,
                (0.4, 0.2, 0.6, 0.5),
                0.9,
                points,
                32,
                24,
            ),
        )


def test_historical_video_uses_default_sampling_and_always_analyzes_last_frame():
    repository = InMemoryRepository()
    repository.add_video(VideoRecord("video_1", "unused.mp4", "cam_1", 32, 24, 10, 10, 1))
    configuration = configuration_record()
    repository.add_configuration(configuration)
    task = TaskRecord(
        "task_sampled",
        "HISTORICAL_VIDEO",
        None,
        "video_1",
        "cfg_1",
        True,
        TaskStatus.RUNNING,
        "created",
        "updated",
    )
    repository.add_task(task)
    tracker = RecordingTracker()
    store = AnalysisResultStore()
    worker = VideoAnalysisWorker(
        repository,
        lambda configuration: tracker,
        store,
        video_source_factory=lambda path: FakeHistoricalSource(),
    )

    worker.run(task, configuration, Event())

    assert tracker.sequences == [0, 2, 4, 6, 8, 9]
    snapshot = store.get(task.task_id)
    assert snapshot.processed_frames == 10
    assert snapshot.analyzed_frames == 6
    assert snapshot.progress == 1
    assert snapshot.latest_frame.frame_seq == 9


def test_video_analysis_worker_updates_progress_and_latest_frame(tmp_path):
    path = tmp_path / "sample.avi"
    make_video(path)
    repository = InMemoryRepository()
    repository.add_video(VideoRecord("video_1", str(path), "cam_1", 32, 24, 2, 4, 2))
    configuration = configuration_record()
    repository.add_configuration(configuration)
    task = TaskRecord(
        "task_1",
        "HISTORICAL_VIDEO",
        None,
        "video_1",
        "cfg_1",
        True,
        TaskStatus.RUNNING,
        "created",
        "updated",
    )
    repository.add_task(task)
    store = AnalysisResultStore()
    worker = VideoAnalysisWorker(repository, lambda configuration: EmptyTracker(), store)

    worker.run(task, configuration, Event())

    snapshot = store.get(task.task_id)
    assert snapshot.processed_frames == 4
    assert snapshot.analyzed_frames == 4
    assert snapshot.progress == 1
    assert snapshot.latest_frame is not None
    assert snapshot.latest_frame.timestamp_sec == pytest.approx(1.5)


def test_video_analysis_worker_marks_inside_person_as_working(tmp_path):
    path = tmp_path / "sample.avi"
    make_video(path, frame_count=2, fps=1)
    repository = InMemoryRepository()
    repository.add_video(VideoRecord("video_1", str(path), "cam_1", 32, 24, 1, 2, 2))
    configuration = configuration_record()
    repository.add_configuration(configuration)
    task = TaskRecord(
        "task_working",
        "VIDEO_FILE",
        None,
        "video_1",
        "cfg_1",
        True,
        TaskStatus.RUNNING,
        "created",
        "updated",
    )
    repository.add_task(task)
    store = AnalysisResultStore()

    VideoAnalysisWorker(repository, lambda configuration: InsideWorkerTracker(), store).run(
        task, configuration, Event()
    )

    result = store.get(task.task_id).latest_frame.work_efficiency
    assert result["persons"][0]["workState"] == "WORKING"
    assert result["areas"][0]["actualWorkingCount"] == 1
    assert result["areas"][0]["qualifiedTeamWorkDurationSec"] == 1


class ControlledCapture:
    def __init__(self) -> None:
        self.frames = Queue()
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 64,
            cv2.CAP_PROP_FRAME_HEIGHT: 48,
            cv2.CAP_PROP_FPS: 25,
        }
        return values.get(property_id, 0)

    def read(self):
        value = self.frames.get(timeout=2)
        if value is None:
            return False, None
        return True, value

    def release(self):
        if not self.released:
            self.released = True
            self.frames.put(None)


class FailedCapture:
    def isOpened(self):
        return False

    def release(self):
        pass


def test_camera_stream_reconnects_and_keeps_only_latest_frame():
    capture = ControlledCapture()
    attempts = [FailedCapture(), capture]

    def capture_factory(url):
        return attempts.pop(0) if attempts else capture

    with CameraStreamSource(
        "rtsp://camera",
        reconnect_delay_sec=0,
        capture_factory=capture_factory,
    ) as source:
        for index in range(3):
            capture.frames.put(np.full((48, 64, 3), index, dtype=np.uint8))
        deadline = monotonic() + 1
        while source._latest_version < 3 and monotonic() < deadline:
            sleep(0.01)

        frame = next(iter(source))

        assert source.metadata.is_live is True
        assert source.metadata.frame_count is None
        assert frame.sequence == 2
        assert int(frame.image[0, 0, 0]) == 2


class FakeLiveSource:
    metadata = FrameSourceMetadata("rtsp://camera", 64, 48, 25, None, None, True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def iter_frames(self, stop_event=None):
        yield VideoFrame(0, 10.0, np.zeros((48, 64, 3)), 1000.0)

    def close(self):
        pass


def test_analysis_worker_uses_same_pipeline_for_camera_stream():
    repository = InMemoryRepository()
    repository.add_camera(CameraRecord("cam_1", "rtsp://camera"))
    configuration = configuration_record()
    repository.add_configuration(configuration)
    task = TaskRecord(
        "task_live",
        "CAMERA_STREAM",
        "cam_1",
        None,
        "cfg_1",
        True,
        TaskStatus.RUNNING,
        "created",
        "updated",
    )
    repository.add_task(task)
    store = AnalysisResultStore()
    worker = VideoAnalysisWorker(
        repository,
        lambda configuration: EmptyTracker(),
        store,
        camera_source_factory=lambda url: FakeLiveSource(),
    )

    worker.run(task, configuration, Event())

    snapshot = store.get(task.task_id)
    assert snapshot.processed_frames == 1
    assert snapshot.progress is None
    assert snapshot.latest_frame.timestamp_sec == 10


def test_result_subscription_drops_stale_frames_for_slow_clients():
    store = AnalysisResultStore()
    subscription = store.subscribe("task_1")
    for sequence in range(3):
        store.update("task_1", FrameAnalysisResult(sequence, float(sequence), None, ()))

    latest = subscription.receive(timeout=0)
    subscription.close()

    assert latest.frame_seq == 2
