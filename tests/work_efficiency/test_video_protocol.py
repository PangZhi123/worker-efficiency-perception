"""Material-style video protocol, including execution and durable resource binding."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from time import monotonic, sleep

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from work_efficiency_perception.analysis import VideoAnalysisWorker
from work_efficiency_perception.api import create_app
from work_efficiency_perception.api.repository import InMemoryRepository, TaskStatus
from work_efficiency_perception.persistence import JsonArchive, JsonRepository, JsonResultStore

API = "/api/work-efficiency/v1"


@pytest.fixture
def mp4(tmp_path):
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    assert writer.isOpened()
    for value in (0, 50, 100):
        writer.write(np.full((24, 32, 3), value, np.uint8))
    writer.release()
    return path.read_bytes()


def upload(client, content, **extra):
    return client.post(API + "/videos", files={"file": ("sample.mp4", content, "video/mp4")},
                       data={"capturedAt": "2026-11-03T01:00:00Z", **extra})


def configure(client, video, camera="cam_01"):
    _, png = cv2.imencode(".png", np.zeros((24, 32, 3), np.uint8))
    response = client.post(API + "/scenes", data={
        "videoId": video, "sceneName": "history", "siteId": "site_01", "description": "test",
    }, files={"file": ("scene.png", png.tobytes(), "image/png")})
    assert response.status_code == 201
    scene = response.json()["data"]["sceneId"]
    payload = {
        "videoId": video, "cameraId": camera, "sceneId": scene,
        "featurePairs": [{"cameraPoint": {"x": x, "y": y}, "scenePoint": {"x": x, "y": y}}
                         for x, y in ((0, 0), (32, 0), (32, 24), (0, 24))],
        "workAreas": [{"areaId": name, "polygon": [
            {"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}],
            "expectedWorkerCount": 2, "requiredTeamWorkDurationSec": 300}
            for name in ("area_1", "area_2")],
    }
    response = client.post(API + "/configurations", json=payload)
    assert response.status_code == 200, response.text
    return response.json()["data"]["configurationId"], payload


def start(client, video, config, **extra):
    return client.post(API + "/tasks", json={"inputType": "VIDEO_FILE",
        "videoId": video, "configurationId": config, **extra})


def wait_status(client, task, status):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        response = client.get(API + f"/tasks/{task}").json()["data"]
        if response["status"] == status:
            return response
        sleep(.01)
    raise AssertionError(response)


class StoppableWorker:
    def run(self, task, configuration, stop_event):
        assert stop_event.wait(5)


class EmptyTracker:
    def track_frame(self, image, **kwargs):
        return ()


def test_upload_deduplication_and_resource_isolation(tmp_path, mp4):
    client = TestClient(create_app(video_directory=tmp_path / "uploads"))
    first = upload(client, mp4, capturedAt="2026-11-03T01:00:00Z")
    assert first.status_code == 201
    video = first.json()["data"]
    assert video["status"] == "READY"
    assert video["cameraId"] is None
    schema = client.get("/openapi.json").json()
    content = schema["paths"][API + "/videos"]["post"]["requestBody"]["content"]
    body = content["multipart/form-data"]["schema"]
    properties = schema["components"]["schemas"][body["$ref"].split("/")[-1]]["properties"]
    assert "cameraId" not in properties
    assert video["sizeBytes"] == len(mp4)
    assert len(video["sha256"]) == 64
    duplicate = upload(client, mp4, capturedAt="2026-11-04T01:00:00Z").json()["data"]
    assert duplicate["videoId"] == video["videoId"]
    assert duplicate["deduplicated"] is True
    assert duplicate["capturedAt"] == video["capturedAt"]
    other = upload(client, mp4 + b"extra").json()["data"]
    assert other["videoId"] != video["videoId"]
    assert other["deduplicated"] is False
    assert len(list((tmp_path / "uploads").glob("*.mp4"))) == 2
    assert not list((tmp_path / "uploads").glob(".upload_*"))
    assert len(client.get(API + "/videos").json()["data"]["items"]) == 2
    assert client.get(video["previewUrl"]).headers["content-type"] == "image/jpeg"
    assert client.get(video["previewUrl"] + "?timestampSec=5").status_code == 422
    config, payload = configure(client, video["videoId"])
    assert client.get(API + f"/configurations/{config}").json()["data"]["request"] == {
        **payload, "algorithm": {"personConfidence": .5, "entryDetectionDurationSec": 3,
                                  "exitDetectionDurationSec": 5}, "storage": {"poseSamplingMs": 1000},
        "workAreas": [{**area, "coordinateType": "NORMALIZED"} for area in payload["workAreas"]],
    }
    reused = start(client, other["videoId"], config)
    assert reused.status_code == 200
    assert reused.json()["data"]["cameraId"] == "cam_01"
    client.patch(API + "/tasks/" + reused.json()["data"]["taskId"], json={"action": "STOP"})
    assert start(client, video["videoId"], config, cameraId="cam_01").status_code == 422
    payload["videoId"] = other["videoId"]
    assert client.post(API + "/configurations", json=payload).status_code == 409
    payload["videoId"] = video["videoId"]
    payload["cameraId"] = "cam_02"
    assert client.post(API + "/configurations", json=payload).status_code == 200
    payload.pop("videoId")
    assert client.post(API + "/configurations", json=payload).status_code == 409
    assert client.post(API + "/tasks", json={"inputType": "CAMERA_STREAM",
        "cameraId": "cam_01", "configurationId": config}).status_code == 400


def test_history_single_slot_and_same_camera_realtime_coexist(tmp_path, mp4):
    with TestClient(create_app(video_directory=tmp_path / "uploads",
                               analysis_worker=StoppableWorker())) as client:
        video = upload(client, mp4).json()["data"]["videoId"]
        config, _ = configure(client, video)
        first = start(client, video, config)
        assert first.status_code == 200
        task = first.json()["data"]["taskId"]
        wait_status(client, task, "RUNNING")
        assert start(client, video, config).status_code == 409
        other = upload(client, mp4 + b"extra").json()["data"]["videoId"]
        other_config, _ = configure(client, other, camera="cam_02")
        assert start(client, other, other_config).status_code == 409
        _, png = cv2.imencode(".png", np.zeros((24, 32, 3), np.uint8))
        scene = client.post(API + "/scenes", data={"sceneName": "live", "siteId": "site"},
            files={"file": ("scene.png", png.tobytes(), "image/png")}).json()["data"]
        realtime = client.get(API + f"/configurations/{config}").json()["data"]["request"]
        realtime.pop("videoId")
        realtime["sceneId"] = scene["sceneId"]
        rt_config = client.post(API + "/configurations", json=realtime).json()["data"]["configurationId"]
        client.post(API + "/cameras", json={"cameraId": "cam_01", "streamUrl": "rtsp://unused"})
        live = client.post(API + "/tasks", json={"inputType": "CAMERA_STREAM",
            "cameraId": "cam_01", "configurationId": rt_config})
        assert live.status_code == 201
        live_task = live.json()["data"]["taskId"]
        wait_status(client, live_task, "RUNNING")
        assert client.post(API + "/tasks", json={"inputType": "CAMERA_STREAM",
            "cameraId": "cam_01", "configurationId": rt_config}).status_code == 409
        response = client.patch(API + f"/tasks/{task}", json={"action": "STOP"})
        assert response.json()["data"]["action"] == "STOP"
        stopped = wait_status(client, task, "STOPPED")
        assert client.patch(API + f"/tasks/{task}", json={"action": "STOP"}).json()["data"]["updatedAt"] == stopped["updatedAt"]
        again = start(client, video, config)
        assert again.status_code == 200
        next_task = again.json()["data"]["taskId"]
        assert next_task != task
        client.patch(API + f"/tasks/{next_task}", json={"action": "STOP"})
        client.patch(API + f"/tasks/{live_task}", json={"action": "STOP"})


def test_real_video_execution_and_json_restart(tmp_path, mp4):
    repo = JsonRepository(JsonArchive(tmp_path / "results"))
    store = JsonResultStore(repo.database, repo)
    worker = VideoAnalysisWorker(repo, tracker_factory=lambda config: EmptyTracker(), result_store=store)
    with TestClient(create_app(repository=repo, analysis_result_store=store,
            analysis_worker=worker, video_directory=tmp_path / "uploads")) as client:
        video = upload(client, mp4, capturedAt="2026-11-03T01:00:00Z").json()["data"]["videoId"]
        config, _ = configure(client, video)
        task = start(client, video, config).json()["data"]["taskId"]
        wait_status(client, task, "COMPLETED")
        progress = client.get(API + f"/tasks/{task}/progress").json()["data"]
        assert progress["progress"] == 1
        assert progress["processedFrames"] == 3
        latest = client.get(API + f"/tasks/{task}/latest-result").json()["data"]
        assert len(latest["areas"]) == 2
        assert latest["frame"]["capturedAt"].startswith("2026-11-03T01:00:00")
        # Completed input may be reanalysed with a fresh task, as in material.
        again = start(client, video, config)
        assert again.status_code == 200
        wait_status(client, again.json()["data"]["taskId"], "COMPLETED")
    restored = JsonRepository(JsonArchive(tmp_path / "results"))
    with TestClient(create_app(repository=restored)) as client:
        assert client.get(API + f"/videos/{video}").status_code == 200
        assert client.get(API + f"/configurations/{config}").status_code == 200
        assert restored.get_scene(restored.get_configuration(config).payload["request"]["sceneId"]).video_id == video
        assert restored.get_task(task).status == TaskStatus.COMPLETED
    archive = repo.database.task_directory(task) / "task_configuration.json"
    assert '"imageBase64"' in archive.read_text(encoding="utf-8")


@pytest.mark.parametrize("content,name,mime,status", [
    (b"", "x.mp4", "video/mp4", 422),
    (b"invalid", "x.mp4", "video/mp4", 422),
    (b"invalid", "x.avi", "video/mp4", 415),
    (b"invalid", "x.mp4", "text/plain", 415),
])
def test_invalid_upload_leaves_no_file(tmp_path, content, name, mime, status):
    client = TestClient(create_app(video_directory=tmp_path / "uploads"))
    response = client.post(API + "/videos",
                           files={"file": (name, content, mime)})
    assert response.status_code == status
    assert not list((tmp_path / "uploads").glob("*"))


def test_upload_limit_authentication_and_unknown_video(tmp_path, mp4, monkeypatch):
    client = TestClient(create_app(video_directory=tmp_path / "uploads"))
    monkeypatch.setenv("WORK_EFFICIENCY_MAX_VIDEO_BYTES", "10")
    assert upload(client, mp4).status_code == 413
    assert not list((tmp_path / "uploads").glob("*"))
    assert upload(client, mp4, capturedAt="bad").status_code == 422
    assert client.get(API + "/videos/missing").status_code == 404
    secure = TestClient(create_app(bearer_token="secret", video_directory=tmp_path / "secure"))
    assert upload(secure, mp4).status_code == 401


def test_task_claim_is_atomic(tmp_path, mp4):
    repo = InMemoryRepository()
    client = TestClient(create_app(repository=repo, video_directory=tmp_path / "uploads"))
    video = upload(client, mp4).json()["data"]["videoId"]
    config, _ = configure(client, video)
    task_id = start(client, video, config).json()["data"]["taskId"]
    original = repo.get_task(task_id)
    repo.update_task(replace(original, status=TaskStatus.STOPPED))
    barrier = Event()

    def claim(index):
        barrier.wait()
        try:
            repo.claim_task(replace(original, task_id=f"race_{index}"), "cam_01")
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(claim, i) for i in range(2)]
        barrier.set()
        assert sum(future.result() for future in futures) == 1


def test_storage_failure_rolls_back_video_and_file(tmp_path, mp4, monkeypatch):
    repo = JsonRepository(JsonArchive(tmp_path / "results"))
    client = TestClient(create_app(repository=repo, video_directory=tmp_path / "uploads"))

    def fail_save():
        raise OSError("simulated disk failure")

    monkeypatch.setattr(repo, "_save", fail_save)
    response = upload(client, mp4)
    assert response.status_code == 500
    assert response.json()["code"] == "WE0401"
    assert not repo.list_videos()
    assert not list((tmp_path / "uploads").glob("*"))


def test_previously_saved_records_load_with_new_optional_fields(tmp_path):
    import json

    from work_efficiency_perception.api.repository import SceneRecord, VideoRecord

    repo = JsonRepository(JsonArchive(tmp_path / "results"))
    repo.add_video(VideoRecord("old_video", "old.mp4", None, 32, 24, 5, 10, 2,
                               analysis_id="old_analysis"))
    repo.add_scene(SceneRecord("old_scene", "old", "site", "/old", 32, 24,
                               "image/png", b"old", analysis_id="old_analysis"))
    file = repo.database.path / "resources.json"
    saved = json.loads(file.read_text(encoding="utf-8"))
    for name in ("original_file_name", "size_bytes", "sha256", "captured_at", "uploaded_at"):
        saved["_videos"]["old_video"]["fields"].pop(name)
    for name in ("video_id", "description"):
        saved["_scenes"]["old_scene"]["fields"].pop(name)
    file.write_text(json.dumps(saved), encoding="utf-8")
    restored = JsonRepository(JsonArchive(repo.database.path))
    assert restored.get_video("old_video").analysis_id == "old_analysis"
    assert restored.get_video("old_video").sha256 is None
    assert restored.get_scene("old_scene").video_id is None


@pytest.mark.parametrize("first_type,second_type", [
    ("VIDEO_FILE", "VIDEO_FILE"), ("HISTORICAL_VIDEO", "VIDEO_FILE"),
    ("VIDEO_FILE", "HISTORICAL_VIDEO"),
])
@pytest.mark.parametrize("state", [TaskStatus.CREATED, TaskStatus.STARTING,
                                   TaskStatus.RUNNING, TaskStatus.STOPPING])
def test_all_history_routes_hold_slot_until_terminal(first_type, second_type, state):
    from work_efficiency_perception.api.repository import (
        HistoricalAnalysisRecord,
        HistoricalPreparationStatus,
        TaskConflictError,
        TaskRecord,
    )

    repo = InMemoryRepository()
    now = "2026-11-03T00:00:00+00:00"
    first = TaskRecord("first", first_type, None, "v", "cfg", True, state, now, now)
    repo.add_task(first)
    second = replace(first, task_id="second", input_type=second_type, status=TaskStatus.CREATED)
    if second_type == "HISTORICAL_VIDEO":
        repo.add_historical_analysis(HistoricalAnalysisRecord(
            "legacy", HistoricalPreparationStatus.READY, now, now))

    def submit():
        if second_type == "HISTORICAL_VIDEO":
            repo.add_historical_task("legacy", second, updated_at=now)
        else:
            repo.claim_task(second, "different-camera")

    with pytest.raises(TaskConflictError):
        submit()
    assert repo.get_task("second") is None
    repo.update_task(replace(first, status=TaskStatus.STOPPED))
    submit()
    assert repo.get_task("second") is not None


@pytest.mark.parametrize("captured,explicit,status", [
    (None, None, 422),
    ("2026-11-03T09:00:00", None, 422),
    ("invalid", None, 422),
    ("2026-11-03T09:00:00+08:00", None, 200),
    (None, "2026-11-02T09:00:00+08:00", 200),
    ("invalid", "2026-11-02T09:00:00+08:00", 200),
    ("2026-11-03T01:00:00Z", "2026-11-02T09:00:00", 422),
    ("2026-11-03T01:00:00Z", "invalid", 422),
])
def test_recording_time_is_required_and_timezone_aware(tmp_path, mp4, captured, explicit, status):
    repo = InMemoryRepository()
    client = TestClient(create_app(repository=repo, video_directory=tmp_path / "uploads"))
    video = upload(client, mp4).json()["data"]["videoId"]
    config, _ = configure(client, video)
    repo.add_video(replace(repo.get_video(video), captured_at=captured))
    response = start(client, video, config, **(
        {"recordingStartedAt": explicit} if explicit is not None else {}))
    assert response.status_code == status
    if status == 200:
        assert response.json()["data"]["recordingStartedAt"] == (explicit or captured)
    else:
        assert not repo.list_tasks()


def test_history_can_select_realtime_configuration_from_another_camera(tmp_path, mp4):
    client = TestClient(create_app(video_directory=tmp_path / "uploads"))
    video = upload(client, mp4).json()["data"]["videoId"]
    _, payload = configure(client, video)
    _, png = cv2.imencode(".png", np.zeros((24, 32, 3), np.uint8))
    scene = client.post(API + "/scenes", data={"sceneName": "reusable", "siteId": "site"},
        files={"file": ("scene.png", png.tobytes(), "image/png")}).json()["data"]
    payload.pop("videoId")
    payload.update(cameraId="another_camera", sceneId=scene["sceneId"])
    config = client.post(API + "/configurations", json=payload).json()["data"]["configurationId"]
    task = start(client, video, config)
    assert task.status_code == 200
    assert task.json()["data"]["cameraId"] == "another_camera"
    assert task.json()["data"]["configurationSnapshot"]["request"]["sceneId"] == scene["sceneId"]
    assert client.get(API + f"/configurations/{config}").json()["data"]["sceneImageUrl"] == scene["imageUrl"]


def test_snapshot_drives_execution_and_archive_after_registry_changes(tmp_path, mp4):
    import json

    from work_efficiency_perception.api.runtime import TaskManager

    repo = JsonRepository(JsonArchive(tmp_path / "results"))
    client = TestClient(create_app(repository=repo, video_directory=tmp_path / "uploads"))
    video = upload(client, mp4).json()["data"]["videoId"]
    config, _ = configure(client, video)
    response = start(client, video, config)
    task_id = response.json()["data"]["taskId"]
    original = repo.get_configuration(config)
    original.payload["request"]["workAreas"][0]["expectedWorkerCount"] = 999
    repo.remove_historical_configuration(config)
    observed = []

    class SnapshotWorker:
        def run(self, task, configuration, stop_event):
            observed.append(configuration.payload["request"]["workAreas"][0]["expectedWorkerCount"])
            configuration.payload["request"]["workAreas"][0]["expectedWorkerCount"] = 123

    manager = TaskManager(repo, SnapshotWorker(), timestamp=lambda: "2026-11-04T00:00:00+00:00")
    try:
        manager.start(task_id)
        wait_status(client, task_id, "COMPLETED")
    finally:
        manager.shutdown()
    assert observed == [2]
    snapshot = repo.get_task(task_id).configuration_snapshot
    assert snapshot.payload["request"]["workAreas"][0]["expectedWorkerCount"] == 2
    archived = json.loads((repo.database.task_directory(task_id) / "task_configuration.json").read_text(encoding="utf-8"))
    assert archived["configuration"]["payload"]["request"]["workAreas"][0]["expectedWorkerCount"] == 2
    restored = JsonRepository(JsonArchive(repo.database.path))
    assert restored.get_task(task_id).configuration_snapshot.payload == snapshot.payload
    assert restored.get_task(task_id).recording_started_at == "2026-11-03T01:00:00+00:00"


def test_recording_override_sets_frame_and_summary_time(tmp_path, mp4):
    repo = JsonRepository(JsonArchive(tmp_path / "results"))
    store = JsonResultStore(repo.database, repo)
    worker = VideoAnalysisWorker(repo, tracker_factory=lambda config: EmptyTracker(), result_store=store)
    with TestClient(create_app(repository=repo, analysis_result_store=store,
            analysis_worker=worker, video_directory=tmp_path / "uploads")) as client:
        video = upload(client, mp4).json()["data"]["videoId"]
        config, _ = configure(client, video)
        response = start(client, video, config, recordingStartedAt="2025-01-02T09:00:00+08:00")
        task_id = response.json()["data"]["taskId"]
        wait_status(client, task_id, "COMPLETED")
        latest = client.get(API + f"/tasks/{task_id}/latest-result").json()["data"]
        assert latest["frame"]["capturedAt"] == "2025-01-02T01:00:00.400000+00:00"
        frame = store.read_frames(task_id)["items"][-1]
        assert frame["frame"]["capturedAtSec"] == store.get(task_id).latest_frame.captured_at_sec
        summary = client.get(API + "/team-efficiency", params={
            "startTime": "2025-01-02T00:00:00Z", "endTime": "2025-01-03T00:00:00Z",
        }).json()["data"]["items"]
        assert len(summary) == 2
        assert summary[0]["intervals"][0]["start"] == "2025-01-02T01:00:00+00:00"
