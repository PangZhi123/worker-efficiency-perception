from datetime import datetime, timedelta, timezone
from time import monotonic, sleep

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from work_efficiency_perception.analysis import AnalysisResultStore, FrameAnalysisResult
from work_efficiency_perception.api import create_app
from work_efficiency_perception.api.repository import CameraRecord, VideoRecord
from work_efficiency_perception.models import ComplianceInterval


class CompletingWorker:
    def run(self, task, configuration, stop_event):
        assert task.configuration_id == configuration.configuration_id


class FailingWorker:
    def run(self, task, configuration, stop_event):
        raise RuntimeError("model failed")


class StoppableWorker:
    def run(self, task, configuration, stop_event):
        stop_event.wait(timeout=2)


def wait_for_status(client: TestClient, task_id: str, expected: str) -> dict:
    deadline = monotonic() + 2
    while monotonic() < deadline:
        task = client.get(f"/api/work-efficiency/v1/tasks/{task_id}").json()["data"]
        if task["status"] == expected:
            return task
        sleep(0.01)
    raise AssertionError(f"task {task_id} did not reach {expected}")


def scene_png() -> bytes:
    image = np.zeros((500, 1000, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def upload_scene(client: TestClient) -> dict:
    response = client.post(
        "/api/work-efficiency/v1/scenes",
        files={"file": ("scene.png", scene_png(), "image/png")},
        data={"sceneName": "test scene", "siteId": "site_001"},
        headers={"X-Request-ID": "req_scene"},
    )
    assert response.status_code == 201
    assert response.json()["requestId"] == "req_scene"
    return response.json()["data"]


def configuration_payload(scene_id: str) -> dict:
    return {
        "cameraId": "cam_01",
        "sceneId": scene_id,
        "featurePairs": [
            {"cameraPoint": {"x": 100, "y": 50}, "scenePoint": {"x": 0, "y": 0}},
            {"cameraPoint": {"x": 300, "y": 50}, "scenePoint": {"x": 1000, "y": 0}},
            {
                "cameraPoint": {"x": 300, "y": 250},
                "scenePoint": {"x": 1000, "y": 500},
            },
            {"cameraPoint": {"x": 100, "y": 250}, "scenePoint": {"x": 0, "y": 500}},
        ],
        "workAreas": [
            {
                "areaId": "zone_01",
                "coordinateType": "NORMALIZED",
                "polygon": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.9, "y": 0.1},
                    {"x": 0.9, "y": 0.9},
                    {"x": 0.1, "y": 0.9},
                ],
                "expectedWorkerCount": 2,
                "requiredTeamWorkDurationSec": 3600,
            }
        ],
    }


def create_configuration(client: TestClient) -> str:
    scene = upload_scene(client)
    response = client.post(
        "/api/work-efficiency/v1/configurations",
        json=configuration_payload(scene["sceneId"]),
    )
    assert response.status_code == 201
    return response.json()["data"]["configurationId"]


def register_camera(client: TestClient, camera_id: str = "cam_01") -> None:
    client.app.state.repository.add_camera(CameraRecord(camera_id, "rtsp://camera"))


def register_video(client: TestClient, video_id: str = "video_01") -> None:
    client.app.state.repository.add_video(
        VideoRecord(video_id, "unused.mp4", "cam_01", 32, 24, 25, 100, 4)
    )


def historical_configuration_payload() -> dict:
    return {
        "calibrationFrameTimestampSec": 0,
        "featurePairs": [
            {"videoPoint": {"x": 0, "y": 0}, "scenePoint": {"x": 0, "y": 0}},
            {"videoPoint": {"x": 32, "y": 0}, "scenePoint": {"x": 1000, "y": 0}},
            {"videoPoint": {"x": 32, "y": 24}, "scenePoint": {"x": 1000, "y": 500}},
            {"videoPoint": {"x": 0, "y": 24}, "scenePoint": {"x": 0, "y": 500}},
        ],
        "workAreas": [
            {
                "areaId": "zone_01",
                "coordinateType": "NORMALIZED",
                "polygon": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.9, "y": 0.1},
                    {"x": 0.9, "y": 0.9},
                    {"x": 0.1, "y": 0.9},
                ],
                "expectedWorkerCount": 2,
                "requiredTeamWorkDurationSec": 3600,
            }
        ],
    }


def prepare_historical_analysis(client: TestClient, tmp_path) -> dict:
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 2, (32, 24)
    )
    assert writer.isOpened()
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.write(np.full((24, 32, 3), 127, dtype=np.uint8))
    writer.release()
    analysis = client.post(
        "/api/work-efficiency/v1/historical-analyses"
    ).json()["data"]
    analysis_id = analysis["analysisId"]
    with source.open("rb") as video_file:
        video = client.post(
            f"/api/work-efficiency/v1/historical-analyses/{analysis_id}/video",
            files={"file": ("sample.mp4", video_file, "video/mp4")},
        )
    assert video.status_code == 201
    scene = client.post(
        f"/api/work-efficiency/v1/historical-analyses/{analysis_id}/scene",
        files={"file": ("scene.png", scene_png(), "image/png")},
        data={"sceneName": "historical scene"},
    )
    assert scene.status_code == 201
    configuration = client.put(
        f"/api/work-efficiency/v1/historical-analyses/{analysis_id}/configuration",
        json=historical_configuration_payload(),
    )
    assert configuration.status_code == 200
    return {
        "analysisId": analysis_id,
        "video": video.json()["data"],
        "scene": scene.json()["data"],
        "configuration": configuration.json()["data"],
    }


def test_scene_upload_and_configuration_creation():
    app = create_app()
    client = TestClient(app)
    scene = upload_scene(client)

    image_response = client.get(scene["imageUrl"])
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"

    response = client.post(
        "/api/work-efficiency/v1/configurations",
        json=configuration_payload(scene["sceneId"]),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "ACTIVE"
    assert body["data"]["quality"]["meanErrorPx"] < 1e-6
    configuration = app.state.repository.get_configuration(body["data"]["configurationId"])
    assert configuration is not None
    assert configuration.payload["homography"][0][0] == pytest.approx(0.005)


def test_historical_analysis_has_independent_video_scene_and_configuration(tmp_path):
    client = TestClient(create_app(video_directory=tmp_path / "uploads"))
    prepared = prepare_historical_analysis(client, tmp_path)

    detail = client.get(
        f"/api/work-efficiency/v1/historical-analyses/{prepared['analysisId']}"
    )
    assert detail.status_code == 200
    data = detail.json()["data"]
    assert data["preparationStatus"] == "READY"
    assert data["video"]["videoId"] == prepared["video"]["videoId"]
    assert "cameraId" not in data["video"]
    assert data["scene"]["sceneId"] == prepared["scene"]["sceneId"]
    assert data["configuration"]["historicalConfigurationId"].startswith("hist_cfg_")

    preview = client.get(
        f"/api/work-efficiency/v1/historical-analyses/{prepared['analysisId']}/video/preview",
        params={"timestampSec": 0.5},
    )
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(preview.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (24, 32)

    scene_file = client.get(prepared["scene"]["imageUrl"])
    assert scene_file.status_code == 200
    assert scene_file.headers["content-type"] == "image/png"


def test_historical_analysis_enforces_order_and_locks_inputs(tmp_path):
    client = TestClient(create_app(video_directory=tmp_path / "uploads"))
    analysis_id = client.post(
        "/api/work-efficiency/v1/historical-analyses"
    ).json()["data"]["analysisId"]
    scene_before_video = client.post(
        f"/api/work-efficiency/v1/historical-analyses/{analysis_id}/scene",
        files={"file": ("scene.png", scene_png(), "image/png")},
        data={"sceneName": "scene"},
    )
    assert scene_before_video.status_code == 400
    assert scene_before_video.json()["code"] == "WE0202"

    prepared = prepare_historical_analysis(client, tmp_path)
    task = client.post(
        f"/api/work-efficiency/v1/historical-analyses/{prepared['analysisId']}/tasks",
        json={},
    )
    assert task.status_code == 201
    assert task.json()["data"]["inputType"] == "HISTORICAL_VIDEO"
    assert task.json()["data"]["analysisId"] == prepared["analysisId"]

    locked = client.put(
        f"/api/work-efficiency/v1/historical-analyses/{prepared['analysisId']}/configuration",
        json=historical_configuration_payload(),
    )
    assert locked.status_code == 409
    assert locked.json()["data"]["suggestedAction"] == "CREATE_NEW_HISTORICAL_ANALYSIS"


def test_realtime_and_historical_configurations_cannot_be_cross_used(tmp_path):
    client = TestClient(create_app(video_directory=tmp_path / "uploads"))
    prepared = prepare_historical_analysis(client, tmp_path)
    register_camera(client)

    response = client.post(
        "/api/work-efficiency/v1/tasks",
        json={
            "inputType": "CAMERA_STREAM",
            "cameraId": "cam_01",
            "configurationId": prepared["configuration"]["historicalConfigurationId"],
        },
    )

    assert response.status_code == 400
    assert response.json()["code"] == "WE0202"


def test_failed_historical_task_can_retry_but_completed_task_cannot(tmp_path):
    with TestClient(
        create_app(analysis_worker=FailingWorker(), video_directory=tmp_path / "uploads")
    ) as client:
        prepared = prepare_historical_analysis(client, tmp_path)
        url = (
            f"/api/work-efficiency/v1/historical-analyses/"
            f"{prepared['analysisId']}/tasks"
        )
        first = client.post(url)
        wait_for_status(client, first.json()["data"]["taskId"], "FAILED")
        retry = client.post(url)
        assert retry.status_code == 201
        assert retry.json()["data"]["taskId"] != first.json()["data"]["taskId"]

    with TestClient(
        create_app(analysis_worker=CompletingWorker(), video_directory=tmp_path / "uploads2")
    ) as client:
        prepared = prepare_historical_analysis(client, tmp_path)
        url = (
            f"/api/work-efficiency/v1/historical-analyses/"
            f"{prepared['analysisId']}/tasks"
        )
        completed = client.post(url)
        wait_for_status(client, completed.json()["data"]["taskId"], "COMPLETED")
        duplicate = client.post(url)
        assert duplicate.status_code == 409
        assert duplicate.json()["code"] == "WE0202"


def test_legacy_history_returns_slot_conflict_and_releases_after_stop(tmp_path):
    client = TestClient(create_app(video_directory=tmp_path / "uploads"))
    first = prepare_historical_analysis(client, tmp_path)
    second = prepare_historical_analysis(client, tmp_path)
    path = "/api/work-efficiency/v1/historical-analyses/"
    running = client.post(path + first["analysisId"] + "/tasks", json={})
    assert running.status_code == 201
    conflict = client.post(path + second["analysisId"] + "/tasks", json={})
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "WE0203"
    assert "historical" in conflict.json()["message"]
    task_id = running.json()["data"]["taskId"]
    client.patch(f"/api/work-efficiency/v1/tasks/{task_id}", json={"action": "STOP"})
    assert client.post(path + second["analysisId"] + "/tasks", json={}).status_code == 201


def test_new_history_can_select_legacy_configuration_without_camera_id(tmp_path):
    client = TestClient(create_app(video_directory=tmp_path / "uploads"))
    prepared = prepare_historical_analysis(client, tmp_path)
    video = client.post("/api/work-efficiency/v1/videos", data={"cameraId": "cam_source"},
        files={"file": ("source.mp4", (tmp_path / "source.mp4").read_bytes(), "video/mp4")})
    response = client.post("/api/work-efficiency/v1/tasks", json={
        "inputType": "VIDEO_FILE", "videoId": video.json()["data"]["videoId"],
        "configurationId": prepared["configuration"]["historicalConfigurationId"],
        "recordingStartedAt": "2026-11-03T09:00:00+08:00",
    })
    assert response.status_code == 200
    assert response.json()["data"]["configurationSnapshot"]["request"]["sceneId"]


def test_missing_historical_analysis_returns_input_unavailable():
    response = TestClient(create_app()).get(
        "/api/work-efficiency/v1/historical-analyses/hist_missing"
    )
    assert response.status_code == 404
    assert response.json()["code"] == "WE0201"


def test_configuration_rejects_unknown_scene_and_invalid_area():
    app = create_app()
    client = TestClient(app)

    unknown = client.post(
        "/api/work-efficiency/v1/configurations",
        json=configuration_payload("missing"),
    )
    assert unknown.status_code == 400
    assert unknown.json()["code"] == "WE0101"

    scene = upload_scene(client)
    payload = configuration_payload(scene["sceneId"])
    payload["workAreas"][0]["polygon"] = [
        {"x": 0, "y": 0},
        {"x": 1, "y": 1},
        {"x": 0, "y": 1},
        {"x": 1, "y": 0},
    ]
    invalid = client.post("/api/work-efficiency/v1/configurations", json=payload)
    assert invalid.status_code == 400
    assert invalid.json()["code"] == "WE0103"


def test_api_uses_standard_validation_envelope_and_optional_bearer_auth():
    client = TestClient(create_app(bearer_token="secret"))
    unauthorized = client.post("/api/work-efficiency/v1/configurations", json={})
    assert unauthorized.status_code == 401
    assert unauthorized.json()["code"] == "WE0001"

    invalid = client.post(
        "/api/work-efficiency/v1/configurations",
        json={},
        headers={"Authorization": "Bearer secret"},
    )
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "WE0101"

    payload = configuration_payload("scene_ignored")
    payload["workAreas"][0]["expectedWorkerCount"] = -1
    invalid_efficiency = client.post(
        "/api/work-efficiency/v1/configurations",
        json=payload,
        headers={"Authorization": "Bearer secret"},
    )
    assert invalid_efficiency.status_code == 422
    assert invalid_efficiency.json()["code"] == "WE0104"


def test_configuration_rejects_poor_reprojection_quality():
    app = create_app(max_mean_reprojection_error_px=1)
    client = TestClient(app)
    scene = upload_scene(client)
    payload = configuration_payload(scene["sceneId"])
    payload["featurePairs"].append(
        {"cameraPoint": {"x": 200, "y": 150}, "scenePoint": {"x": 900, "y": 450}}
    )

    response = client.post("/api/work-efficiency/v1/configurations", json=payload)

    assert response.status_code == 400
    assert response.json()["code"] == "WE0102"


def test_camera_task_create_query_and_idempotent_stop():
    client = TestClient(create_app())
    configuration_id = create_configuration(client)
    register_camera(client)

    created = client.post(
        "/api/work-efficiency/v1/tasks",
        json={
            "inputType": "CAMERA_STREAM",
            "cameraId": "cam_01",
            "configurationId": configuration_id,
        },
    )
    assert created.status_code == 201
    task = created.json()["data"]
    assert task["status"] == "CREATED"
    assert task["videoId"] is None

    queried = client.get(f"/api/work-efficiency/v1/tasks/{task['taskId']}")
    assert queried.status_code == 200
    assert queried.json()["data"]["configurationId"] == configuration_id

    stopped = client.patch(
        f"/api/work-efficiency/v1/tasks/{task['taskId']}", json={"action": "STOP"}
    )
    assert stopped.status_code == 200
    assert stopped.json()["data"]["status"] == "STOPPED"

    stopped_again = client.patch(
        f"/api/work-efficiency/v1/tasks/{task['taskId']}", json={"action": "STOP"}
    )
    assert stopped_again.status_code == 200
    assert stopped_again.json()["data"]["status"] == "STOPPED"
    assert stopped_again.json()["data"]["updatedAt"] == stopped.json()["data"]["updatedAt"]


def test_task_rejects_invalid_input_combinations_and_missing_configuration():
    client = TestClient(create_app())
    configuration_id = create_configuration(client)

    both_sources = client.post(
        "/api/work-efficiency/v1/tasks",
        json={
            "inputType": "CAMERA_STREAM",
            "cameraId": "cam_01",
            "videoId": "video_01",
            "configurationId": configuration_id,
        },
    )
    assert both_sources.status_code == 400
    assert both_sources.json()["code"] == "WE0101"

    missing_configuration = client.post(
        "/api/work-efficiency/v1/tasks",
        json={
            "inputType": "CAMERA_STREAM",
            "cameraId": "cam_01",
            "configurationId": "cfg_missing",
        },
    )
    assert missing_configuration.status_code == 400
    assert missing_configuration.json()["code"] == "WE0202"

    missing_task = client.patch(
        "/api/work-efficiency/v1/tasks/task_missing", json={"action": "STOP"}
    )
    assert missing_task.status_code == 404
    assert missing_task.json()["code"] == "WE0201"

    legacy_video_task = client.post(
        "/api/work-efficiency/v1/tasks",
        json={
            "inputType": "VIDEO_FILE",
            "videoId": "video_01",
            "configurationId": configuration_id,
        },
    )
    assert legacy_video_task.status_code == 404
    assert legacy_video_task.json()["code"] == "WE0201"


@pytest.mark.parametrize(
    ("worker", "status", "failure_code"),
    [(CompletingWorker(), "COMPLETED", None), (FailingWorker(), "FAILED", "WE0301")],
)
def test_task_manager_records_completion_and_failure(
    worker, status, failure_code, tmp_path
):
    with TestClient(
        create_app(analysis_worker=worker, video_directory=tmp_path / "uploads")
    ) as client:
        prepared = prepare_historical_analysis(client, tmp_path)
        created = client.post(
            f"/api/work-efficiency/v1/historical-analyses/{prepared['analysisId']}/tasks",
            json={},
        ).json()["data"]

        final = wait_for_status(client, created["taskId"], status)

        assert final["failureCode"] == failure_code
        if failure_code:
            assert final["failureMessage"] == "model failed"


def test_task_manager_cooperatively_stops_running_worker():
    with TestClient(create_app(analysis_worker=StoppableWorker())) as client:
        configuration_id = create_configuration(client)
        register_camera(client)
        created = client.post(
            "/api/work-efficiency/v1/tasks",
            json={
                "inputType": "CAMERA_STREAM",
                "cameraId": "cam_01",
                "configurationId": configuration_id,
            },
        ).json()["data"]
        wait_for_status(client, created["taskId"], "RUNNING")

        stopping = client.patch(
            f"/api/work-efficiency/v1/tasks/{created['taskId']}", json={"action": "STOP"}
        )

        assert stopping.status_code == 200
        assert stopping.json()["data"]["status"] in {"STOPPING", "STOPPED"}
        wait_for_status(client, created["taskId"], "STOPPED")


def test_frontend_can_query_progress_latest_result_and_team_efficiency(tmp_path):
    store = AnalysisResultStore()
    app = create_app(
        analysis_result_store=store, video_directory=tmp_path / "uploads"
    )
    client = TestClient(app)
    prepared = prepare_historical_analysis(client, tmp_path)
    task = client.post(
        f"/api/work-efficiency/v1/historical-analyses/{prepared['analysisId']}/tasks",
        json={},
    ).json()["data"]
    work_efficiency = {
        "persons": [
            {
                "trackId": "1",
                "areaId": "zone_01",
                "personZoneState": "INSIDE",
                "workState": "WORKING",
            }
        ],
        "areas": [
            {
                "areaId": "zone_01",
                "actualWorkingCount": 1,
                "expectedWorkerCount": 1,
                "headcountCompliant": True,
                "qualifiedTeamWorkDurationSec": 5,
                "requiredTeamWorkDurationSec": 10,
                "efficiencyStatus": "NOT_MET",
            }
        ],
    }
    store.update(
        task["taskId"],
        FrameAnalysisResult(5, 5.0, 0.5, (), 1000.0, work_efficiency),
        {"zone_01": (ComplianceInterval(0, None, True),)},
    )

    progress = client.get(f"/api/work-efficiency/v1/tasks/{task['taskId']}/progress")
    assert progress.json()["data"]["progress"] == 0.5
    assert progress.json()["data"]["processedFrames"] == 6
    assert progress.json()["data"]["analyzedFrames"] == 1

    latest = client.get(f"/api/work-efficiency/v1/tasks/{task['taskId']}/latest-result")
    assert latest.json()["data"]["persons"][0]["workState"] == "WORKING"

    efficiency = client.get(
        "/api/work-efficiency/v1/team-efficiency",
        params={"taskId": task["taskId"], "areaId": "zone_01"},
    )
    item = efficiency.json()["data"]["items"][0]
    assert item["actualWorkingCount"] == 1
    assert item["headcountCompliantDurationSec"] == 5
    assert "start" in item["intervals"][0]
    assert "startSec" not in item["intervals"][0]

    now = datetime.now(timezone.utc)
    by_time = client.get(
        "/api/work-efficiency/v1/team-efficiency",
        params={
            "startTime": (now - timedelta(minutes=1)).isoformat(),
            "endTime": (now + timedelta(minutes=1)).isoformat(),
        },
    )
    assert by_time.status_code == 200
    assert by_time.json()["data"]["items"][0]["taskId"] == task["taskId"]

    missing_filter = client.get("/api/work-efficiency/v1/team-efficiency")
    assert missing_filter.status_code == 400
    assert missing_filter.json()["code"] == "WE0101"


def test_websocket_pushes_documented_frame_result(tmp_path):
    store = AnalysisResultStore()
    app = create_app(
        analysis_result_store=store, video_directory=tmp_path / "uploads"
    )
    client = TestClient(app)
    prepared = prepare_historical_analysis(client, tmp_path)
    task_id = client.post(
        f"/api/work-efficiency/v1/historical-analyses/{prepared['analysisId']}/tasks",
        json={},
    ).json()["data"]["taskId"]
    result = {
        "persons": [
            {
                "trackId": "1",
                "confidence": 0.9,
                "bboxPx": [1, 2, 3, 4],
                "bboxNorm": [0.1, 0.2, 0.3, 0.4],
                "keypoints": [],
                "anchorPx": {"x": 2, "y": 4},
                "scenePointNorm": {"x": 0.5, "y": 0.5},
                "trajectoryDelta": [{"seq": 1, "x": 0.5, "y": 0.5}],
                "areaId": "zone_01",
                "personZoneState": "INSIDE",
                "workState": "WORKING",
            }
        ],
        "areas": [],
    }

    with client.websocket_connect(f"/ws/work-efficiency/v1/tasks/{task_id}") as websocket:
        store.update(task_id, FrameAnalysisResult(1, 1.0, 0.1, (), 1000.0, result, 32, 24))
        message = websocket.receive_json()

    assert message["type"] == "FRAME_RESULT"
    assert message["taskId"] == task_id
    assert message["frame"]["seq"] == 1
    assert message["frame"]["width"] == 32
    assert datetime.fromisoformat(message["frame"]["capturedAt"]).tzinfo is not None
    assert message["persons"][0]["trajectoryDelta"][0]["seq"] == 1


def test_websocket_requires_configured_bearer_token():
    client = TestClient(create_app(bearer_token="secret"))
    with (
        pytest.raises(WebSocketDisconnect) as error,
        client.websocket_connect("/ws/work-efficiency/v1/tasks/task_1"),
    ):
        pass
    assert error.value.code == 4401


@pytest.mark.parametrize("status", ["COMPLETED", "STOPPED", "FAILED"])
@pytest.mark.parametrize("has_frame", [False, True])
def test_websocket_terminal_status_and_normal_close(status, has_frame):
    from work_efficiency_perception.api.repository import TaskRecord, TaskStatus
    app = create_app()
    task = TaskRecord("terminal", "CAMERA_STREAM", "cam", None, "cfg", True,
                      TaskStatus(status), "2026-11-04T00:00:00+00:00",
                      "2026-11-04T00:01:00+00:00", failure_code="WE0301",
                      failure_message="model failed")
    app.state.repository.add_task(task)
    if has_frame:
        app.state.analysis_result_store.update(task.task_id,
            FrameAnalysisResult(5, 1., 1., (), work_efficiency={"persons": [], "areas": []}))
    with TestClient(app) as client, client.websocket_connect(
        "/ws/work-efficiency/v1/tasks/terminal"
    ) as ws:
        if has_frame:
            assert ws.receive_json()["type"] == "FRAME_RESULT"
        if status == "FAILED":
            assert ws.receive_json()["type"] == "ERROR"
        message = ws.receive_json()
        assert message == {"type": "TASK_STATUS", "taskId": "terminal",
            "taskStatus": status, "endedAt": task.updated_at,
            "error": {"code": "WE0301", "message": "model failed"} if status == "FAILED" else None}
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_json()
        assert error.value.code == 1000


def test_resource_lists_and_deletion_survive_restart(tmp_path):
    from work_efficiency_perception.persistence import JsonArchive, JsonRepository
    repository = JsonRepository(JsonArchive(tmp_path / "archive"))
    with TestClient(create_app(repository=repository)) as client:
        register_camera(client)
        cfg = create_configuration(client)
        assert client.get("/api/work-efficiency/v1/cameras").json()["data"]["items"] == [
            {"cameraId": "cam_01", "status": "REGISTERED"}]
        assert client.get("/api/work-efficiency/v1/configurations").json()["data"]["items"][0]["configurationId"] == cfg
    restored = JsonRepository(JsonArchive(tmp_path / "archive"))
    with TestClient(create_app(repository=restored)) as client:
        assert restored.get_camera("cam_01") is not None
        assert client.get(f"/api/work-efficiency/v1/configurations/{cfg}").status_code == 200
        assert client.delete("/api/work-efficiency/v1/cameras/cam_01").status_code == 200
        assert client.delete(f"/api/work-efficiency/v1/configurations/{cfg}").status_code == 200
    restored = JsonRepository(JsonArchive(tmp_path / "archive"))
    with TestClient(create_app(repository=restored)) as client:
        assert client.get("/api/work-efficiency/v1/cameras").json()["data"]["items"] == []
        assert client.get("/api/work-efficiency/v1/configurations").json()["data"]["items"] == []
        assert client.get(f"/api/work-efficiency/v1/configurations/{cfg}").status_code == 404
        assert restored.get_configuration(cfg).payload["response"]["status"] == "DELETED"
        register_camera(client)
        response = client.post("/api/work-efficiency/v1/tasks", json={
            "inputType": "CAMERA_STREAM", "cameraId": "cam_01", "configurationId": cfg})
        assert response.json()["code"] == "WE0202"


def test_running_camera_delete_conflict_and_configuration_snapshot():
    app = create_app()
    with TestClient(app) as client:
        register_camera(client)
        cfg = create_configuration(client)
        task = client.post("/api/work-efficiency/v1/tasks", json={
            "inputType": "CAMERA_STREAM", "cameraId": "cam_01", "configurationId": cfg}).json()["data"]
        assert client.delete("/api/work-efficiency/v1/cameras/cam_01").status_code == 409
        assert client.delete(f"/api/work-efficiency/v1/configurations/{cfg}").status_code == 200
        snapshot = app.state.repository.get_task(task["taskId"]).configuration_snapshot
        assert snapshot.payload["response"]["status"] != "DELETED"
        client.patch(f"/api/work-efficiency/v1/tasks/{task['taskId']}", json={"action": "STOP"})
        assert client.delete("/api/work-efficiency/v1/cameras/cam_01").status_code == 200


def test_resource_management_requires_auth():
    with TestClient(create_app(bearer_token="secret")) as client:
        for path in ("cameras", "configurations"):
            assert client.get(f"/api/work-efficiency/v1/{path}").status_code == 401
            assert client.delete(f"/api/work-efficiency/v1/{path}/missing").status_code == 401


def test_websocket_running_task_sends_last_frame_before_terminal():
    from dataclasses import replace
    from work_efficiency_perception.api.repository import TaskRecord, TaskStatus
    app = create_app()
    task = TaskRecord("finishing", "CAMERA_STREAM", "cam", None, "cfg", True,
                      TaskStatus.RUNNING, "2026-11-04T00:00:00+00:00",
                      "2026-11-04T00:00:00+00:00")
    app.state.repository.add_task(task)
    with TestClient(app) as client, client.websocket_connect(
        "/ws/work-efficiency/v1/tasks/finishing"
    ) as ws:
        app.state.analysis_result_store.update(task.task_id,
            FrameAnalysisResult(7, 2., 1., (), work_efficiency={"persons": [], "areas": []}))
        app.state.repository.update_task(replace(task, status=TaskStatus.COMPLETED))
        frame = ws.receive_json()
        assert frame["type"] == "FRAME_RESULT"
        assert frame["frame"]["seq"] == 7
        assert ws.receive_json()["taskStatus"] == "COMPLETED"
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_json()
        assert error.value.code == 1000
