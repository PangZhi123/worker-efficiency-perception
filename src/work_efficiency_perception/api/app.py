from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from ..calibration import calibrate_homography, validate_work_areas
from ..models import CalibrationPointPair, WorkArea
from .repository import (
    TERMINAL_TASK_STATUSES,
    CameraRecord,
    ConfigurationRecord,
    InMemoryRepository,
    SceneRecord,
    TaskRecord,
    TaskStatus,
    VideoRecord,
)
from .runtime import AnalysisWorker, TaskManager
from .schemas import (
    CameraRegistrationRequest,
    ConfigurationRequest,
    TaskActionRequest,
    TaskCreateRequest,
)

MAX_SCENE_FILE_BYTES = 20 * 1024 * 1024
ALLOWED_SCENE_MEDIA_TYPES = {"image/jpeg", "image/png"}


class ApiError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def create_app(
    *,
    repository: InMemoryRepository | None = None,
    bearer_token: str | None = None,
    max_mean_reprojection_error_px: float = 3.0,
    analysis_worker: AnalysisWorker | None = None,
    task_worker_count: int = 2,
    analysis_result_store: Any | None = None,
    video_directory: str | Path = "data/videos",
) -> FastAPI:
    if max_mean_reprojection_error_px < 0:
        raise ValueError("max_mean_reprojection_error_px must be non-negative")
    app = FastAPI(title="Worker Efficiency Perception API", version="0.1.0")
    app.state.repository = repository or InMemoryRepository()
    app.state.video_directory = Path(video_directory).resolve()
    if analysis_result_store is None:
        from ..analysis import AnalysisResultStore

        analysis_result_store = AnalysisResultStore()
    app.state.analysis_result_store = analysis_result_store
    app.state.task_manager = (
        TaskManager(
            app.state.repository,
            analysis_worker,
            timestamp=_timestamp,
            max_workers=task_worker_count,
        )
        if analysis_worker is not None
        else None
    )
    if app.state.task_manager is not None:
        app.router.add_event_handler("shutdown", app.state.task_manager.shutdown)

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if bearer_token is None:
            return
        if authorization != f"Bearer {bearer_token}":
            raise ApiError("WE0001", "unauthorized", status_code=401)

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=_envelope(request, success=False, code=error.code, message=error.message),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        details = [
            {"location": list(item["loc"]), "message": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        locations = {str(part) for item in error.errors() for part in item["loc"]}
        code = "WE0104" if locations & {
            "expectedWorkerCount",
            "requiredTeamWorkDurationSec",
            "personConfidence",
            "entryDetectionDurationSec",
            "exitDetectionDurationSec",
            "poseSamplingMs",
        } else "WE0101"
        return JSONResponse(
            status_code=422,
            content=_envelope(
                request,
                success=False,
                code=code,
                message="invalid request parameters",
                data={"details": details},
            ),
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/work-efficiency/v1/scenes", dependencies=[Depends(authorize)])
    async def create_scene(
        request: Request,
        file: Annotated[UploadFile, File()],
        scene_name: Annotated[str, Form(alias="sceneName", min_length=1, max_length=128)],
        site_id: Annotated[str, Form(alias="siteId", min_length=1, max_length=64)],
    ) -> JSONResponse:
        if file.content_type not in ALLOWED_SCENE_MEDIA_TYPES:
            raise ApiError("WE0101", "scene file must be PNG or JPEG")
        content = await file.read(MAX_SCENE_FILE_BYTES + 1)
        if len(content) > MAX_SCENE_FILE_BYTES:
            raise ApiError("WE0101", "scene file exceeds 20 MB")
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ApiError("WE0101", "scene file is not a valid image")
        height, width = image.shape[:2]
        scene_id = _new_id("scene")
        scene = SceneRecord(
            scene_id=scene_id,
            scene_name=scene_name,
            site_id=site_id,
            image_url=f"/files/scenes/{scene_id}",
            width=int(width),
            height=int(height),
            media_type=file.content_type,
            content=content,
        )
        app.state.repository.add_scene(scene)
        return JSONResponse(
            status_code=201,
            content=_envelope(
                request,
                data={
                    "sceneId": scene.scene_id,
                    "imageUrl": scene.image_url,
                    "width": scene.width,
                    "height": scene.height,
                },
            ),
        )

    @app.get("/files/scenes/{scene_id}", dependencies=[Depends(authorize)])
    async def get_scene_file(scene_id: str) -> Response:
        scene = app.state.repository.get_scene(scene_id)
        if scene is None:
            raise ApiError("WE0101", f"unknown sceneId: {scene_id}", status_code=404)
        return Response(content=scene.content, media_type=scene.media_type)

    @app.post("/api/work-efficiency/v1/configurations", dependencies=[Depends(authorize)])
    async def create_configuration(
        request: Request, body: ConfigurationRequest
    ) -> JSONResponse:
        scene = app.state.repository.get_scene(body.scene_id)
        if scene is None:
            raise ApiError("WE0101", f"unknown sceneId: {body.scene_id}")
        try:
            areas = tuple(
                WorkArea(
                    area.area_id,
                    tuple((point.x, point.y) for point in area.polygon),
                    area.expected_worker_count,
                    area.required_team_work_duration_sec,
                )
                for area in body.work_areas
            )
            validate_work_areas(areas)
            calibration = calibrate_homography(
                tuple(
                    CalibrationPointPair(
                        (pair.camera_point.x, pair.camera_point.y),
                        (pair.scene_point.x, pair.scene_point.y),
                    )
                    for pair in body.feature_pairs
                ),
                scene_width=scene.width,
                scene_height=scene.height,
            )
            if calibration.mean_error_px > max_mean_reprojection_error_px:
                raise ValueError(
                    "calibration mean reprojection error exceeds "
                    f"{max_mean_reprojection_error_px:g} px"
                )
        except ValueError as error:
            message = str(error)
            code = "WE0102" if "calibration" in message or "homography" in message else "WE0103"
            raise ApiError(code, message) from error

        configuration_id = _new_id("cfg")
        calibration_version = _new_id("cal")
        area_version = _new_id("area")
        effective_areas = [
            {
                "areaId": area.area_id,
                "expectedWorkerCount": area.expected_worker_count,
                "expectedWorkerCountSource": "USER"
                if area.expected_worker_count is not None
                else "SYSTEM",
                "requiredTeamWorkDurationSec": area.required_team_work_duration_sec,
                "requiredDurationSource": "USER"
                if area.required_team_work_duration_sec is not None
                else "SYSTEM",
            }
            for area in areas
        ]
        data = {
            "configurationId": configuration_id,
            "calibrationVersion": calibration_version,
            "areaVersion": area_version,
            "quality": {
                "meanErrorPx": calibration.mean_error_px,
                "maxErrorPx": calibration.max_error_px,
            },
            "effectiveAreas": effective_areas,
            "status": "ACTIVE",
        }
        app.state.repository.add_configuration(
            ConfigurationRecord(
                configuration_id=configuration_id,
                calibration_version=calibration_version,
                area_version=area_version,
                payload={
                    "request": body.model_dump(by_alias=True),
                    "homography": calibration.homography,
                    "response": data,
                },
            )
        )
        return JSONResponse(status_code=201, content=_envelope(request, data=data))

    @app.post("/api/work-efficiency/v1/videos", dependencies=[Depends(authorize)])
    async def upload_video(
        request: Request,
        file: Annotated[UploadFile, File()],
        camera_id: Annotated[str, Form(alias="cameraId", min_length=1, max_length=64)],
    ) -> JSONResponse:
        if file.content_type not in {"video/mp4", "application/octet-stream"}:
            raise ApiError("WE0101", "historical video must be an MP4 file")
        video_id = _new_id("video")
        directory: Path = app.state.video_directory
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{video_id}.mp4"
        try:
            with path.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    destination.write(chunk)
            from ..video import VideoFileSource

            with VideoFileSource(path) as source:
                metadata = source.metadata
            record = VideoRecord(
                video_id,
                str(path),
                camera_id,
                metadata.width,
                metadata.height,
                metadata.fps,
                metadata.frame_count or 0,
                metadata.duration_sec or 0.0,
            )
            app.state.repository.add_video(record)
        except (OSError, ValueError) as error:
            path.unlink(missing_ok=True)
            raise ApiError("WE0201", f"video cannot be registered: {error}") from error
        return JSONResponse(
            status_code=201,
            content=_envelope(
                request,
                data={
                    "videoId": record.video_id,
                    "cameraId": record.camera_id,
                    "width": record.width,
                    "height": record.height,
                    "fps": record.fps,
                    "frameCount": record.frame_count,
                    "durationSec": record.duration_sec,
                },
            ),
        )

    @app.post("/api/work-efficiency/v1/cameras", dependencies=[Depends(authorize)])
    async def register_camera(
        request: Request, body: CameraRegistrationRequest
    ) -> JSONResponse:
        if not body.stream_url.lower().startswith(("rtsp://", "rtmp://")):
            raise ApiError("WE0101", "camera streamUrl must use RTSP or RTMP")
        app.state.repository.add_camera(CameraRecord(body.camera_id, body.stream_url))
        return JSONResponse(
            status_code=201,
            content=_envelope(
                request,
                data={"cameraId": body.camera_id, "status": "REGISTERED"},
            ),
        )

    @app.post("/api/work-efficiency/v1/tasks", dependencies=[Depends(authorize)])
    async def create_task(request: Request, body: TaskCreateRequest) -> JSONResponse:
        _validate_task_input(body)
        configuration = app.state.repository.get_configuration(body.configuration_id)
        if configuration is None:
            raise ApiError(
                "WE0202", f"configuration is not active: {body.configuration_id}"
            )
        _validate_task_resource(app.state.repository, body, configuration)
        now = _timestamp()
        task = TaskRecord(
            task_id=_new_id("task"),
            input_type=body.input_type,
            camera_id=body.camera_id,
            video_id=body.video_id,
            configuration_id=body.configuration_id,
            persist_pose_and_trajectory=body.options.persist_pose_and_trajectory,
            status=TaskStatus.CREATED,
            created_at=now,
            updated_at=now,
        )
        app.state.repository.add_task(task)
        if app.state.task_manager is not None:
            task = app.state.task_manager.start(task.task_id)
        return JSONResponse(
            status_code=201,
            content=_envelope(request, data=_task_payload(task)),
        )

    @app.get("/api/work-efficiency/v1/tasks/{task_id}", dependencies=[Depends(authorize)])
    async def get_task(request: Request, task_id: str) -> JSONResponse:
        task = _require_task(app.state.repository, task_id)
        return JSONResponse(content=_envelope(request, data=_task_payload(task)))

    @app.patch("/api/work-efficiency/v1/tasks/{task_id}", dependencies=[Depends(authorize)])
    async def update_task(
        request: Request, task_id: str, body: TaskActionRequest
    ) -> JSONResponse:
        task = _require_task(app.state.repository, task_id)
        if body.action == "STOP" and task.status not in TERMINAL_TASK_STATUSES:
            if app.state.task_manager is not None:
                task = app.state.task_manager.request_stop(task_id)
            else:
                task = replace(task, status=TaskStatus.STOPPED, updated_at=_timestamp())
                app.state.repository.update_task(task)
        return JSONResponse(content=_envelope(request, data=_task_payload(task)))

    @app.get(
        "/api/work-efficiency/v1/tasks/{task_id}/progress",
        dependencies=[Depends(authorize)],
    )
    async def get_task_progress(request: Request, task_id: str) -> JSONResponse:
        task = _require_task(app.state.repository, task_id)
        snapshot = app.state.analysis_result_store.get(task_id)
        latest = snapshot.latest_frame
        data = {
            **_task_payload(task),
            "processedFrames": snapshot.processed_frames,
            "progress": snapshot.progress,
            "currentTimestampSec": None if latest is None else latest.timestamp_sec,
        }
        return JSONResponse(content=_envelope(request, data=data))

    @app.get(
        "/api/work-efficiency/v1/tasks/{task_id}/latest-result",
        dependencies=[Depends(authorize)],
    )
    async def get_latest_result(request: Request, task_id: str) -> JSONResponse:
        _require_task(app.state.repository, task_id)
        snapshot = app.state.analysis_result_store.get(task_id)
        if snapshot.latest_frame is None:
            raise ApiError("WE0201", "task has no analysis result yet", status_code=404)
        return JSONResponse(
            content=_envelope(request, data=_frame_result_payload(snapshot.latest_frame))
        )

    @app.get("/api/work-efficiency/v1/team-efficiency", dependencies=[Depends(authorize)])
    async def get_team_efficiency(
        request: Request,
        task_id: Annotated[str | None, Query(alias="taskId", min_length=1)] = None,
        area_id: Annotated[str | None, Query(alias="areaId")] = None,
        start_time: Annotated[datetime | None, Query(alias="startTime")] = None,
        end_time: Annotated[datetime | None, Query(alias="endTime")] = None,
    ) -> JSONResponse:
        if task_id is None and (start_time is None or end_time is None):
            raise ApiError("WE0101", "taskId or both startTime/endTime are required")
        if (start_time is None) != (end_time is None):
            raise ApiError("WE0101", "startTime and endTime must be provided together")
        if start_time is not None and end_time is not None and end_time <= start_time:
            raise ApiError("WE0101", "endTime must be later than startTime")
        if task_id is not None:
            tasks = (_require_task(app.state.repository, task_id),)
        else:
            tasks = tuple(
                task
                for task in app.state.repository.list_tasks()
                if _datetime_epoch(start_time)
                <= _datetime_epoch(datetime.fromisoformat(task.created_at))
                <= _datetime_epoch(end_time)
            )
        items = []
        for task in tasks:
            snapshot = app.state.analysis_result_store.get(task.task_id)
            latest = snapshot.latest_frame
            if latest is None or latest.work_efficiency is None:
                continue
            intervals_by_area = snapshot.compliance_intervals or {}
            for area in latest.work_efficiency["areas"]:
                if area_id is not None and area["areaId"] != area_id:
                    continue
                intervals = intervals_by_area.get(area["areaId"], ())
                compliant_duration, insufficient_duration = _interval_durations(
                    intervals, latest.timestamp_sec
                )
                items.append(
                    {
                        "taskId": task.task_id,
                        **area,
                        "headcountCompliantDurationSec": compliant_duration,
                        "headcountInsufficientDurationSec": insufficient_duration,
                        "intervals": [
                            {
                                "start": _source_time_iso(task, latest, interval.start_sec),
                                "end": None
                                if interval.end_sec is None
                                else _source_time_iso(task, latest, interval.end_sec),
                                "headcountCompliant": interval.headcount_compliant,
                            }
                            for interval in intervals
                        ],
                    }
                )
        return JSONResponse(content=_envelope(request, data={"items": items}))

    @app.websocket("/ws/work-efficiency/v1/tasks/{task_id}")
    async def stream_task_results(websocket: WebSocket, task_id: str) -> None:
        supplied_token = websocket.query_params.get("access_token")
        authorization = websocket.headers.get("authorization")
        if bearer_token is not None and not (
            authorization == f"Bearer {bearer_token}" or supplied_token == bearer_token
        ):
            await websocket.close(code=4401, reason="unauthorized")
            return
        task = app.state.repository.get_task(task_id)
        if task is None:
            await websocket.close(code=4404, reason="task not found")
            return
        await websocket.accept()
        subscription = app.state.analysis_result_store.subscribe(task_id)
        last_sequence = -1
        try:
            initial = app.state.analysis_result_store.get(task_id).latest_frame
            if initial is not None:
                await websocket.send_json(_websocket_frame_payload(task, initial))
                last_sequence = initial.frame_seq
            while True:
                frame = await asyncio.to_thread(subscription.receive, 0.5)
                if frame is not None and frame.frame_seq > last_sequence:
                    await websocket.send_json(_websocket_frame_payload(task, frame))
                    last_sequence = frame.frame_seq
                task = app.state.repository.get_task(task_id)
                if task is None:
                    await websocket.send_json(
                        {"type": "ERROR", "taskId": task_id, "code": "WE0201", "message": "task not found"}
                    )
                    break
                if task.status == TaskStatus.FAILED:
                    await websocket.send_json(
                        {
                            "type": "ERROR",
                            "taskId": task_id,
                            "code": task.failure_code or "WE0301",
                            "message": task.failure_message or "analysis failed",
                        }
                    )
                    break
                if task.status in TERMINAL_TASK_STATUSES:
                    break
        except WebSocketDisconnect:
            pass
        finally:
            subscription.close()

    return app


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_task_input(body: TaskCreateRequest) -> None:
    if body.input_type == "CAMERA_STREAM":
        valid = body.camera_id is not None and body.video_id is None
    else:
        valid = body.video_id is not None and body.camera_id is None
    if not valid:
        raise ApiError(
            "WE0101",
            "CAMERA_STREAM requires only cameraId; VIDEO_FILE requires only videoId",
        )


def _validate_task_resource(
    repository: InMemoryRepository,
    body: TaskCreateRequest,
    configuration: ConfigurationRecord,
) -> None:
    configured_camera_id = configuration.payload.get("request", {}).get("cameraId")
    if body.input_type == "VIDEO_FILE":
        video = repository.get_video(body.video_id or "")
        if video is None:
            raise ApiError("WE0201", f"video is not registered: {body.video_id}")
        source_camera_id = video.camera_id
    else:
        camera = repository.get_camera(body.camera_id or "")
        if camera is None:
            raise ApiError("WE0201", f"camera is not registered: {body.camera_id}")
        source_camera_id = camera.camera_id
    if configured_camera_id != source_camera_id:
        raise ApiError(
            "WE0202",
            f"configuration is not active for camera: {source_camera_id}",
        )


def _require_task(repository: InMemoryRepository, task_id: str) -> TaskRecord:
    task = repository.get_task(task_id)
    if task is None:
        raise ApiError("WE0201", f"task not found: {task_id}", status_code=404)
    return task


def _task_payload(task: TaskRecord) -> dict:
    return {
        "taskId": task.task_id,
        "inputType": task.input_type,
        "cameraId": task.camera_id,
        "videoId": task.video_id,
        "configurationId": task.configuration_id,
        "status": task.status.value,
        "options": {"persistPoseAndTrajectory": task.persist_pose_and_trajectory},
        "createdAt": task.created_at,
        "updatedAt": task.updated_at,
        "failureCode": task.failure_code,
        "failureMessage": task.failure_message,
    }


def _frame_result_payload(frame: Any) -> dict:
    result = frame.work_efficiency or {"persons": [], "areas": []}
    return {
        "frame": {
            "seq": frame.frame_seq,
            "timestampSec": frame.timestamp_sec,
            "width": frame.frame_width,
            "height": frame.frame_height,
            "capturedAt": None
            if frame.captured_at_sec is None
            else datetime.fromtimestamp(frame.captured_at_sec, timezone.utc).isoformat(),
        },
        "progress": frame.progress,
        "persons": result["persons"],
        "areas": result["areas"],
    }


def _websocket_frame_payload(task: TaskRecord, frame: Any) -> dict:
    payload = _frame_result_payload(frame)
    captured_at = payload["frame"]["capturedAt"]
    if captured_at is None:
        captured_at = _source_time_iso(task, frame, frame.timestamp_sec)
    payload["frame"].pop("timestampSec", None)
    payload["frame"]["capturedAt"] = captured_at
    payload.pop("progress", None)
    return {"type": "FRAME_RESULT", "taskId": task.task_id, **payload}


def _source_time_iso(task: TaskRecord, frame: Any, source_timestamp_sec: float) -> str:
    if frame.captured_at_sec is not None:
        epoch_sec = frame.captured_at_sec + source_timestamp_sec - frame.timestamp_sec
        return datetime.fromtimestamp(epoch_sec, timezone.utc).isoformat()
    task_started = datetime.fromisoformat(task.created_at)
    return (task_started + timedelta(seconds=source_timestamp_sec)).isoformat()


def _datetime_epoch(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _interval_durations(intervals: tuple, current_timestamp_sec: float) -> tuple[float, float]:
    compliant = 0.0
    insufficient = 0.0
    for interval in intervals:
        end = current_timestamp_sec if interval.end_sec is None else interval.end_sec
        duration = max(end - interval.start_sec, 0.0)
        if interval.headcount_compliant:
            compliant += duration
        else:
            insufficient += duration
    return compliant, insufficient


def _envelope(
    request: Request,
    *,
    success: bool = True,
    code: str = "0",
    message: str = "success",
    data: object = None,
) -> dict:
    return {
        "requestId": request.headers.get("X-Request-ID") or _new_id("req"),
        "success": success,
        "code": code,
        "message": message,
        "data": {} if data is None and success else data,
        "timestamp": _timestamp(),
    }
