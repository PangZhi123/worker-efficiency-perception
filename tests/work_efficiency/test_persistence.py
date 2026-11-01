from dataclasses import replace

from fastapi.testclient import TestClient

from work_efficiency_perception.analysis import FrameAnalysisResult
from work_efficiency_perception.api import create_app
from work_efficiency_perception.api.repository import TaskRecord, TaskStatus, SceneRecord
from work_efficiency_perception.models import ComplianceInterval
from work_efficiency_perception.persistence import JsonArchive, JsonRepository, JsonResultStore


def task(persist=True):
    return TaskRecord("task_saved", "HISTORICAL_VIDEO", None, "v", "c", persist,
                      TaskStatus.RUNNING, "2026-11-01T00:00:00+00:00", "2026-11-01T00:00:00+00:00")


def test_restart_restores_resources_results_and_marks_interrupted_task(tmp_path):
    path = tmp_path / "results"
    db = JsonArchive(path)
    repo = JsonRepository(db)
    store = JsonResultStore(db, repo)
    repo.add_scene(SceneRecord("s", "scene", "site", "/s", 10, 10, "image/png", b"image"))
    repo.add_task(task())
    for seq in (0, 5):
        store.update("task_saved", FrameAnalysisResult(seq, seq / 25, .5, (),
                     work_efficiency={"persons": [{"trackId": "1"}], "areas": []}),
                     {"a": (ComplianceInterval(0, None, True),)})
    db.close()
    db = JsonArchive(path)
    repo = JsonRepository(db)
    store = JsonResultStore(db, repo)
    assert repo.get_scene("s").content == b"image"
    assert repo.get_task("task_saved").status == TaskStatus.FAILED
    snapshot = store.get("task_saved")
    assert snapshot.analyzed_frames == 2
    assert snapshot.latest_frame.frame_seq == 5
    assert snapshot.compliance_intervals["a"][0].headcount_compliant is True
    with TestClient(create_app(repository=repo, analysis_result_store=store)) as client:
        assert client.get("/api/work-efficiency/v1/tasks").json()["data"]["items"][0]["taskId"] == "task_saved"
        page = client.get("/api/work-efficiency/v1/tasks/task_saved/frames?limit=1").json()["data"]
        assert page["nextAfterSeq"] == 0
        page = client.get("/api/work-efficiency/v1/tasks/task_saved/frames?afterSeq=0").json()["data"]
        assert page["items"][0]["frame"]["seq"] == 5
    db.close()


def test_completed_progress_and_pose_opt_out_survive_restart(tmp_path):
    path = tmp_path / "results"
    db = JsonArchive(path)
    repo = JsonRepository(db)
    store = JsonResultStore(db, repo)
    repo.add_task(task(False))
    store.update("task_saved", FrameAnalysisResult(0, 0, .1, (),
                 work_efficiency={"persons": [{"trackId": "1"}], "areas": [{"areaId": "a"}]}))
    store.complete("task_saved", 100)
    repo.update_task(replace(repo.get_task("task_saved"), status=TaskStatus.COMPLETED))
    db.close()
    db = JsonArchive(path)
    repo = JsonRepository(db)
    store = JsonResultStore(db, repo)
    assert repo.get_task("task_saved").status == TaskStatus.COMPLETED
    assert store.get("task_saved").progress == 1
    assert store.get("task_saved").processed_frames == 100
    assert store.get("task_saved").latest_frame.work_efficiency["persons"] == []
    frame = store.read_frames("task_saved")["items"][0]
    assert frame["persons"] == []
    assert frame["areas"] == [{"areaId": "a"}]
    db.close()


def test_matrix_and_30_second_efficiency_archives(tmp_path):
    import json
    from work_efficiency_perception.api.repository import ConfigurationRecord
    db = JsonArchive(tmp_path / "results")
    repo = JsonRepository(db)
    matrix = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    repo.add_configuration(ConfigurationRecord("c", "v1", "v1", {"homography": matrix}))
    repo.add_task(task())
    store = JsonResultStore(db, repo)
    for seq, timestamp in enumerate((0, 29.9, 30.1, 31, 60, 65)):
        store.update("task_saved", FrameAnalysisResult(seq, timestamp, .5, (),
            work_efficiency={"persons": [{"trackId": "1"}], "areas": [{"areaId": "a", "qualifiedTeamWorkDurationSec": timestamp}]}))
    store.complete("task_saved", 6)
    directory = db.task_directory("task_saved")
    config = json.loads((directory / "task_configuration.json").read_text())
    assert config["homography"] == [list(row) for row in matrix]
    paths = sorted((directory / "efficiency_30s").glob("*.json"))
    assert len(paths) == 2
    saved = json.loads(paths[0].read_text())
    assert saved["scheduledTimestampSec"] == 30
    assert saved["timestampSec"] == 30.1
    assert len(list((directory / "persons").glob("*.json"))) == 6
    overall = json.loads((directory / "efficiency_overall.json").read_text())
    assert overall["timestampSec"] == 65
    assert overall["progress"] == 1
    assert not list(db.path.rglob("*.sqlite3"))


def test_json_write_retries_transient_windows_lock(tmp_path, monkeypatch):
    import json
    from work_efficiency_perception import persistence
    destination = tmp_path / "overall.json"
    destination.write_text('{"old": true}', encoding="utf-8")
    original = persistence.os.replace
    calls = []
    def locked_then_available(source, target):
        calls.append(source)
        if len(calls) < 4:
            assert json.loads(destination.read_text()) == {"old": True}
            raise PermissionError("file temporarily locked")
        return original(source, target)
    monkeypatch.setattr(persistence.os, "replace", locked_then_available)
    monkeypatch.setattr(persistence.time, "sleep", lambda delay: None)
    persistence.write_json(destination, {"new": True})
    assert len(calls) == 4
    assert json.loads(destination.read_text()) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_json_write_persistent_lock_preserves_previous_result(tmp_path, monkeypatch):
    import pytest
    from work_efficiency_perception import persistence
    destination = tmp_path / "overall.json"
    destination.write_text('{"old": true}', encoding="utf-8")
    calls = []
    def always_locked(source, target):
        calls.append(source)
        raise PermissionError("locked")
    monkeypatch.setattr(persistence.os, "replace", always_locked)
    monkeypatch.setattr(persistence.time, "sleep", lambda delay: None)
    with pytest.raises(PermissionError):
        persistence.write_json(destination, {"new": True})
    assert len(calls) == 11
    assert destination.read_text() == '{"old": true}'
    assert not list(tmp_path.glob("*.tmp"))
