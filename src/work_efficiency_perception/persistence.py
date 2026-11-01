"""Atomic JSON archives for a single local service process."""
from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import RLock

from . import models
from .analysis import AnalysisResultStore, FrameAnalysisResult, TaskAnalysisSnapshot
from .api import repository as records

_TYPES = {cls.__name__: cls for module in (models, records)
          for cls in vars(module).values() if isinstance(cls, type)
          and (is_dataclass(cls) or issubclass(cls, Enum))}
_TYPES.update(FrameAnalysisResult=FrameAnalysisResult, TaskAnalysisSnapshot=TaskAnalysisSnapshot)


def encode(value):
    if isinstance(value, Enum):
        return {"$enum": type(value).__name__, "value": value.value}
    if is_dataclass(value):
        return {"$type": type(value).__name__, "fields": {
            f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return {"$tuple": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "$type" in value:
        return _TYPES[value["$type"]](**{k: decode(v) for k, v in value["fields"].items()})
    if "$enum" in value:
        return _TYPES[value["$enum"]](value["value"])
    if "$bytes" in value:
        return base64.b64decode(value["$bytes"])
    if "$tuple" in value:
        return tuple(decode(v) for v in value["$tuple"])
    return {k: decode(v) for k, v in value.items()}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Unique sibling files avoid collisions and preserve atomic replacement.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(11):
            try:
                os.replace(temporary, path)
                break
            except OSError as error:
                transient = isinstance(error, PermissionError) or getattr(error, "winerror", None) in {5, 32, 33}
                if not transient or attempt == 10:
                    raise
                time.sleep(min(0.05 * 2 ** attempt, 0.5))
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Cleanup must not hide the original write failure.
                pass



class JsonArchive:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()

    def task_directory(self, task_id):
        if not task_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in task_id):
            raise ValueError("invalid task id for archive")
        return self.path / "tasks" / task_id

    def close(self):
        pass


_COLLECTIONS = ("_scenes", "_videos", "_cameras", "_configurations",
                "_historical_configurations", "_tasks", "_historical_analyses")


class JsonRepository(records.InMemoryRepository):
    def __init__(self, database):
        super().__init__()
        self.database = database
        with database.lock:
            resource_file = database.path / "resources.json"
            if resource_file.exists():
                saved = json.loads(resource_file.read_text(encoding="utf-8"))
                for name in _COLLECTIONS:
                    if name in saved:
                        setattr(self, name, decode(saved[name]))
        now = datetime.now(timezone.utc).isoformat()
        for key, task in self._tasks.items():
            if task.status not in records.TERMINAL_TASK_STATUSES:
                self._tasks[key] = replace(task, status=records.TaskStatus.FAILED,
                    updated_at=now, failure_code="WE0301",
                    failure_message="Service restarted; task interrupted. Retry from the beginning.")
        self._save()

    def _save(self):
        with self.database.lock:
            write_json(self.database.path / "resources.json", {
                name: encode(getattr(self, name)) for name in _COLLECTIONS})
            for task in self._tasks.values():
                directory = self.database.task_directory(task.task_id)
                config = self.get_configuration(task.configuration_id)
                video = self.get_video(task.video_id) if task.video_id else None
                scene_id = None if config is None else config.payload.get("request", {}).get("sceneId")
                if task.analysis_id and task.analysis_id in self._historical_analyses:
                    scene_id = self._historical_analyses[task.analysis_id].scene_id
                scene = self.get_scene(scene_id) if scene_id else None
                scene_info = None if scene is None else {
                    "sceneId": scene.scene_id, "sceneName": scene.scene_name,
                    "width": scene.width, "height": scene.height,
                    "mediaType": scene.media_type,
                    "imageBase64": base64.b64encode(scene.content).decode("ascii")}
                write_json(directory / "task_configuration.json", {
                    "schemaVersion": 1, "task": asdict(task),
                    "configuration": None if config is None else asdict(config),
                    "homography": None if config is None else config.payload.get("homography"),
                    "matrixConvention": "[scene_x_normalized, scene_y_normalized, 1] ~ H @ [video_x_px, video_y_px, 1]",
                    "video": None if video is None else asdict(video), "scene": scene_info})


def durable_mutation(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        with self._lock:
            previous = {name: dict(getattr(self, name)) for name in _COLLECTIONS}
            try:
                result = method(self, *args, **kwargs)
                self._save()
                return result
            except Exception:
                for name, values in previous.items():
                    setattr(self, name, values)
                raise
    return invoke


for _name in ("add_scene", "remove_scene", "add_video", "remove_video", "add_camera",
              "add_configuration", "remove_configuration", "add_historical_configuration",
              "remove_historical_configuration", "add_historical_analysis",
              "update_historical_analysis", "add_historical_task", "add_task", "update_task",
              "transition_task"):
    setattr(JsonRepository, _name, durable_mutation(getattr(records.InMemoryRepository, _name)))


class JsonResultStore(AnalysisResultStore):
    def __init__(self, database, repository):
        super().__init__()
        self.database = database
        self.repository = repository
        for path in (database.path / "tasks").glob("*/snapshot.json"):
            self._snapshots[path.parent.name] = decode(json.loads(path.read_text(encoding="utf-8")))

    def _stored_snapshot(self, task_id, snapshot):
        task = self.repository.get_task(task_id)
        if task is not None and not task.persist_pose_and_trajectory and snapshot.latest_frame:
            frame = snapshot.latest_frame
            result = dict(frame.work_efficiency or {})
            result["persons"] = []
            return replace(snapshot, latest_frame=replace(frame, persons=(), work_efficiency=result))
        return snapshot

    def _efficiency(self, task_id, snapshot):
        frame = snapshot.latest_frame
        return {"taskId": task_id, "timestampSec": None if frame is None else frame.timestamp_sec,
                "processedFrames": snapshot.processed_frames, "analyzedFrames": snapshot.analyzed_frames,
                "progress": snapshot.progress,
                "areas": [] if frame is None else (frame.work_efficiency or {}).get("areas", []),
                "complianceIntervals": {key: [asdict(item) for item in intervals]
                    for key, intervals in (snapshot.compliance_intervals or {}).items()}}

    def _save_snapshot(self, task_id, snapshot):
        directory = self.database.task_directory(task_id)
        write_json(directory / "snapshot.json", encode(self._stored_snapshot(task_id, snapshot)))
        write_json(directory / "efficiency_overall.json", self._efficiency(task_id, snapshot))

    def update(self, task_id, frame, compliance_intervals=None):
        with self._lock, self.database.lock:
            previous = self.get(task_id)
            snapshot = TaskAnalysisSnapshot(frame.frame_seq + 1,
                previous.analyzed_frames + 1, frame.progress, frame, compliance_intervals)
            stored_frame = self._stored_snapshot(task_id, snapshot).latest_frame
            directory = self.database.task_directory(task_id)
            payload = {"taskId": task_id,
                "frame": {"seq": frame.frame_seq, "timestampSec": frame.timestamp_sec,
                    "width": frame.frame_width, "height": frame.frame_height,
                    "capturedAtSec": frame.captured_at_sec}, **(stored_frame.work_efficiency or {})}
            write_json(directory / "persons" / f"frame_{frame.frame_seq:012d}.json", payload)
            # First analyzed frame at/after each 30-second boundary; never invent missing observations.
            prior_time = -1 if previous.latest_frame is None else previous.latest_frame.timestamp_sec
            boundary = int((frame.timestamp_sec + 1e-9) // 30) * 30
            prior_boundary = int((prior_time + 1e-9) // 30) * 30
            if boundary >= 30 and boundary > prior_boundary:
                efficiency = self._efficiency(task_id, snapshot)
                efficiency["scheduledTimestampSec"] = boundary
                write_json(directory / "efficiency_30s" / f"at_{boundary:012d}.json", efficiency)
            self._save_snapshot(task_id, snapshot)
            super().update(task_id, frame, compliance_intervals)

    def complete(self, task_id, processed_frames):
        with self._lock, self.database.lock:
            previous = self.get(task_id)
            super().complete(task_id, processed_frames)
            try:
                self._save_snapshot(task_id, self.get(task_id))
            except Exception:
                self._snapshots[task_id] = previous
                raise

    def read_frames(self, task_id, after_seq=-1, limit=100):
        with self.database.lock:
            directory = self.database.task_directory(task_id) / "persons"
            paths = sorted(p for p in directory.glob("frame_*.json")
                           if int(p.stem.split("_")[1]) > after_seq)[:limit]
            items = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        return {"items": items,
                "nextAfterSeq": items[-1]["frame"]["seq"] if items else after_seq}
