from __future__ import annotations

import asyncio
import os
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
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
from fastapi.staticfiles import StaticFiles

from ..calibration import calibrate_homography, validate_work_areas
from ..models import CalibrationPointPair, WorkArea
from .repository import (
    TERMINAL_TASK_STATUSES,
    CameraRecord,
    ConfigurationRecord,
    HistoricalAnalysisRecord,
    HistoricalPreparationStatus,
    InMemoryRepository,
    SceneRecord,
    TaskConflictError,
    TaskRecord,
    TaskStatus,
    VideoRecord,
)
from .runtime import AnalysisWorker, TaskManager
from .schemas import (
    CameraRegistrationRequest,
    ConfigurationRequest,
    HistoricalConfigurationRequest,
    HistoricalTaskCreateRequest,
    TaskActionRequest,
    TaskCreateRequest,
)

MAX_SCENE_FILE_BYTES = 20 * 1024 * 1024
ALLOWED_SCENE_MEDIA_TYPES = {"image/jpeg", "image/png"}


class ApiError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        data: object = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.data = data


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
            content=_envelope(
                request,
                success=False,
                code=error.code,
                message=error.message,
                data=error.data,
            ),
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

    @app.get("/api/work-efficiency/v1/videos", dependencies=[Depends(authorize)])
    async def list_videos(request: Request) -> JSONResponse:
        return JSONResponse(content=_envelope(request, data={"items": [
            _video_payload(video) for video in app.state.repository.list_videos()
            if video.analysis_id is None
        ]}))

    @app.post("/api/work-efficiency/v1/videos", dependencies=[Depends(authorize)])
    async def upload_video(
        request: Request,
        file: Annotated[UploadFile, File()],
        camera_id: Annotated[str, Form(alias="cameraId", min_length=1, max_length=64)],
        captured_at: Annotated[str | None, Form(alias="capturedAt")] = None,
    ) -> JSONResponse:
        camera_id = camera_id.strip()
        if not camera_id:
            raise ApiError("WE0101", "cameraId must not be blank", status_code=422)
        if captured_at is not None:
            try:
                captured_at = datetime.fromisoformat(captured_at.replace("Z", "+00:00")).isoformat()
            except ValueError as error:
                raise ApiError("WE0101", "capturedAt must be an ISO 8601 timestamp", status_code=422) from error
        name = Path((file.filename or "upload.mp4").replace("\\", "/")).name
        if Path(name).suffix.lower() != ".mp4" or (file.content_type or "application/octet-stream").lower() not in {
            "video/mp4", "application/mp4", "application/octet-stream",
        }:
            raise ApiError("WE0101", "only MP4 video is supported", status_code=415)
        maximum = int(os.getenv("WORK_EFFICIENCY_MAX_VIDEO_BYTES", str(5 * 1024 ** 3)))
        directory = app.state.video_directory
        temporary = directory / f".upload_{uuid4().hex}.mp4"
        owned_path = temporary
        digest = sha256()
        size = 0
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with temporary.open("xb") as destination:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > maximum:
                        raise ApiError("WE0101", "video exceeds maximum upload size", status_code=413)
                    digest.update(chunk)
                    destination.write(chunk)
            if not size:
                raise ApiError("WE0101", "video file must not be empty", status_code=422)
            matching = [v for v in app.state.repository.list_videos()
                        if v.analysis_id is None and v.sha256 == digest.hexdigest()
                        and Path(v.path).is_file()]
            same_camera = next((v for v in matching if v.camera_id == camera_id), None)
            if same_camera is not None:
                return JSONResponse(status_code=201, content=_envelope(
                    request, data=_video_payload(same_camera, deduplicated=True)))
            video_id = _new_id("video")
            if matching:
                video = replace(matching[0], video_id=video_id, camera_id=camera_id,
                                original_file_name=name, captured_at=captured_at,
                                uploaded_at=_timestamp())
            else:
                from ..video import VideoFileSource

                with VideoFileSource(temporary) as source:
                    metadata = source.metadata
                    if not metadata.frame_count or next(iter(source), None) is None:
                        raise ValueError("video has no decodable frames")
                final_path = directory / f"{video_id}.mp4"
                temporary.replace(final_path)
                owned_path = final_path
                video = VideoRecord(
                    video_id, str(final_path), camera_id, metadata.width, metadata.height,
                    metadata.fps, metadata.frame_count, metadata.duration_sec or 0.0,
                    original_file_name=name, size_bytes=size, sha256=digest.hexdigest(),
                    captured_at=captured_at, uploaded_at=_timestamp(),
                )
            app.state.repository.add_video(video)
            if matching:
                owned_path.unlink(missing_ok=True)
            owned_path = None
            return JSONResponse(status_code=201, content=_envelope(
                request, data=_video_payload(video, deduplicated=bool(matching))))
        except ValueError as error:
            raise ApiError("WE0201", f"video cannot be registered: {error}", status_code=422) from error
        except OSError as error:
            raise ApiError("WE0401", f"video storage failed: {error}", status_code=500) from error
        finally:
            if owned_path is not None:
                owned_path.unlink(missing_ok=True)
            await file.close()

    @app.get("/api/work-efficiency/v1/videos/{video_id}", dependencies=[Depends(authorize)])
    async def get_video(request: Request, video_id: str) -> JSONResponse:
        video = _require_registered_video(app.state.repository, video_id)
        return JSONResponse(content=_envelope(request, data=_video_payload(video)))

    @app.get("/api/work-efficiency/v1/videos/{video_id}/preview", dependencies=[Depends(authorize)])
    async def video_preview(
        video_id: str,
        timestamp_sec: Annotated[float, Query(alias="timestampSec", ge=0)] = 0,
    ) -> Response:
        video = _require_registered_video(app.state.repository, video_id)
        if timestamp_sec >= video.duration_sec:
            raise ApiError("WE0101", "timestampSec exceeds video duration", status_code=422)
        return Response(_read_video_preview(video, timestamp_sec), media_type="image/jpeg")

    @app.post("/api/work-efficiency/v1/scenes", dependencies=[Depends(authorize)])
    async def create_scene(
        request: Request,
        file: Annotated[UploadFile, File()],
        scene_name: Annotated[str, Form(alias="sceneName", min_length=1, max_length=128)],
        site_id: Annotated[str, Form(alias="siteId", min_length=1, max_length=64)],
        video_id: Annotated[str | None, Form(alias="videoId")] = None,
        description: Annotated[str | None, Form()] = None,
    ) -> JSONResponse:
        if video_id is not None:
            _require_registered_video(app.state.repository, video_id)
        if file.content_type not in ALLOWED_SCENE_MEDIA_TYPES:
            raise ApiError("WE0101", "scene file must be PNG or JPEG",
                           status_code=422 if video_id is not None else 400)
        content = await file.read(MAX_SCENE_FILE_BYTES + 1)
        if len(content) > MAX_SCENE_FILE_BYTES:
            raise ApiError("WE0101", "scene file exceeds 20 MB",
                           status_code=422 if video_id is not None else 400)
        if not content:
            raise ApiError("WE0101", "scene file must not be empty", status_code=422)
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ApiError("WE0101", "scene file is not a valid image",
                           status_code=422 if video_id is not None else 400)
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
            video_id=video_id,
            description=description,
        )
        app.state.repository.add_scene(scene)
        return JSONResponse(
            status_code=201,
            content=_envelope(
                request,
                data={
                    "sceneId": scene.scene_id,
                    "sceneName": scene.scene_name,
                    "videoId": scene.video_id,
                    "sha256": sha256(content).hexdigest(),
                    "imageUrl": scene.image_url,
                    "width": scene.width,
                    "height": scene.height,
                },
            ),
        )

    @app.get("/files/scenes/{scene_id}", dependencies=[Depends(authorize)])
    async def get_scene_file(scene_id: str) -> Response:
        scene = app.state.repository.get_scene(scene_id)
        if scene is None or scene.analysis_id is not None:
            raise ApiError("WE0101", f"unknown sceneId: {scene_id}", status_code=404)
        return Response(content=scene.content, media_type=scene.media_type)

    @app.post("/api/work-efficiency/v1/configurations", dependencies=[Depends(authorize)])
    async def create_configuration(
        request: Request, body: ConfigurationRequest
    ) -> JSONResponse:
        scene = app.state.repository.get_scene(body.scene_id)
        if scene is None or scene.analysis_id is not None:
            raise ApiError("WE0101", f"unknown sceneId: {body.scene_id}",
                           status_code=404 if body.video_id is not None else 400)
        if body.video_id is not None:
            video = _require_registered_video(app.state.repository, body.video_id)
            if scene.video_id != video.video_id:
                raise ApiError("WE0202", "sceneId is not bound to this videoId", status_code=409)
            if video.camera_id != body.camera_id:
                raise ApiError("WE0201", "video cameraId does not match configuration", status_code=422)
            if any(not (0 <= pair.camera_point.x <= video.width
                        and 0 <= pair.camera_point.y <= video.height)
                   for pair in body.feature_pairs):
                raise ApiError("WE0102", "calibration points must lie within the video frame", status_code=422)
        elif scene.video_id is not None:
            raise ApiError("WE0202", "video-bound scene requires videoId", status_code=409)
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
            "cameraId": body.camera_id,
            "videoId": body.video_id,
            "calibrationVersion": calibration_version,
            "areaVersion": area_version,
            "quality": {
                "meanErrorPx": calibration.mean_error_px,
                "maxErrorPx": calibration.max_error_px,
            },
            "effectiveAreas": effective_areas,
            "status": "ACTIVE",
        }
        save_configuration = (
            app.state.repository.add_historical_configuration
            if body.video_id is not None else app.state.repository.add_configuration
        )
        save_configuration(
            ConfigurationRecord(
                configuration_id=configuration_id,
                calibration_version=calibration_version,
                area_version=area_version,
                configuration_type="HISTORICAL" if body.video_id is not None else "REALTIME",
                payload={
                    "request": body.model_dump(by_alias=True),
                    "homography": calibration.homography,
                    "response": data,
                },
            )
        )
        return JSONResponse(status_code=200 if body.video_id else 201,
                            content=_envelope(request, data=data))

    @app.get("/api/work-efficiency/v1/cameras", dependencies=[Depends(authorize)])
    async def list_cameras(request: Request) -> JSONResponse:
        return JSONResponse(content=_envelope(request, data={"items": [
            {"cameraId": c.camera_id, "status": "REGISTERED"}
            for c in app.state.repository.list_cameras()
        ]}))

    @app.delete("/api/work-efficiency/v1/cameras/{camera_id}", dependencies=[Depends(authorize)])
    async def delete_camera(request: Request, camera_id: str) -> JSONResponse:
        try:
            app.state.repository.remove_camera(camera_id)
        except TaskConflictError as error:
            raise ApiError("WE0203", str(error), status_code=409) from error
        return JSONResponse(content=_envelope(request, data={"cameraId": camera_id, "status": "DELETED"}))

    @app.get("/api/work-efficiency/v1/configurations", dependencies=[Depends(authorize)])
    async def list_configurations(request: Request) -> JSONResponse:
        return JSONResponse(content=_envelope(request, data={"items": [
            {"configurationId": c.configuration_id, "configurationType": c.configuration_type,
             "cameraId": c.payload.get("request", {}).get("cameraId"),
             "sceneId": c.payload.get("request", {}).get("sceneId"),
             "analysisId": c.analysis_id}
            for c in app.state.repository.list_configurations()
        ]}))

    @app.delete("/api/work-efficiency/v1/configurations/{configuration_id}", dependencies=[Depends(authorize)])
    async def delete_configuration(request: Request, configuration_id: str) -> JSONResponse:
        app.state.repository.delete_configuration(configuration_id)
        return JSONResponse(content=_envelope(request, data={"configurationId": configuration_id, "status": "DELETED"}))

    @app.get("/api/work-efficiency/v1/configurations/{configuration_id}", dependencies=[Depends(authorize)])
    async def get_configuration(request: Request, configuration_id: str) -> JSONResponse:
        configuration = app.state.repository.get_configuration(configuration_id)
        if configuration is None or configuration.payload.get("response", {}).get("status") == "DELETED":
            raise ApiError("WE0201", "configuration is not available", status_code=404)
        scene = app.state.repository.get_scene(configuration.payload["request"].get("sceneId", ""))
        return JSONResponse(content=_envelope(request, data={
            "configurationId": configuration_id,
            **configuration.payload["response"], "request": configuration.payload["request"],
            "sceneImageUrl": scene.image_url if scene is not None else None,
        }))

    @app.post(
        "/api/work-efficiency/v1/historical-analyses",
        dependencies=[Depends(authorize)],
    )
    async def create_historical_analysis(request: Request) -> JSONResponse:
        now = _timestamp()
        analysis = HistoricalAnalysisRecord(
            analysis_id=_new_id("hist"),
            preparation_status=HistoricalPreparationStatus.EMPTY,
            created_at=now,
            updated_at=now,
        )
        app.state.repository.add_historical_analysis(analysis)
        return JSONResponse(
            status_code=201,
            content=_envelope(request, data=_historical_analysis_payload(app, analysis)),
        )

    @app.get(
        "/api/work-efficiency/v1/historical-analyses/{analysis_id}",
        dependencies=[Depends(authorize)],
    )
    async def get_historical_analysis(request: Request, analysis_id: str) -> JSONResponse:
        analysis = _require_historical_analysis(app.state.repository, analysis_id)
        return JSONResponse(
            content=_envelope(request, data=_historical_analysis_payload(app, analysis))
        )

    @app.post(
        "/api/work-efficiency/v1/historical-analyses/{analysis_id}/video",
        dependencies=[Depends(authorize)],
    )
    async def upload_historical_video(
        request: Request,
        analysis_id: str,
        file: Annotated[UploadFile, File()],
    ) -> JSONResponse:
        analysis = _require_editable_historical_analysis(
            app.state.repository, analysis_id
        )
        if file.content_type not in {"video/mp4", "application/octet-stream"}:
            raise ApiError("WE0101", "historical video must be an MP4 file")
        video_id = _new_id("video")
        directory: Path = app.state.video_directory / analysis_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{video_id}.mp4"
        try:
            with path.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    destination.write(chunk)
            from ..video import VideoFileSource

            with VideoFileSource(path) as source:
                metadata = source.metadata
            video = VideoRecord(
                video_id=video_id,
                path=str(path),
                camera_id=None,
                width=metadata.width,
                height=metadata.height,
                fps=metadata.fps,
                frame_count=metadata.frame_count or 0,
                duration_sec=metadata.duration_sec or 0.0,
                analysis_id=analysis_id,
            )
            app.state.repository.add_video(video)
        except (OSError, ValueError) as error:
            path.unlink(missing_ok=True)
            raise ApiError("WE0201", f"video cannot be registered: {error}") from error
        _discard_historical_configuration(app.state.repository, analysis)
        status = (
            HistoricalPreparationStatus.ASSETS_READY
            if analysis.scene_id is not None
            else HistoricalPreparationStatus.VIDEO_UPLOADED
        )
        updated = app.state.repository.update_historical_analysis(
            analysis_id,
            expected={analysis.preparation_status},
            updated_at=_timestamp(),
            video_id=video.video_id,
            configuration_id=None,
            preparation_status=status,
        )
        if updated is None:
            path.unlink(missing_ok=True)
            app.state.repository.remove_video(video.video_id)
            raise ApiError("WE0202", "historical analysis changed during upload")
        _remove_replaced_video(app, analysis.video_id)
        return JSONResponse(
            status_code=201,
            content=_envelope(request, data=_historical_video_payload(video)),
        )

    @app.post(
        "/api/work-efficiency/v1/historical-analyses/{analysis_id}/scene",
        dependencies=[Depends(authorize)],
    )
    async def upload_historical_scene(
        request: Request,
        analysis_id: str,
        file: Annotated[UploadFile, File()],
        scene_name: Annotated[str, Form(alias="sceneName", min_length=1, max_length=128)],
    ) -> JSONResponse:
        analysis = _require_editable_historical_analysis(
            app.state.repository, analysis_id
        )
        if analysis.video_id is None:
            raise ApiError("WE0202", "historical video must be uploaded before scene")
        content, width, height = await _decode_scene_upload(file)
        scene = SceneRecord(
            scene_id=_new_id("hist_scene"),
            scene_name=scene_name,
            site_id=analysis_id,
            image_url=f"/files/historical-scenes/{analysis_id}",
            width=width,
            height=height,
            media_type=file.content_type or "image/png",
            content=content,
            analysis_id=analysis_id,
        )
        app.state.repository.add_scene(scene)
        _discard_historical_configuration(app.state.repository, analysis)
        updated = app.state.repository.update_historical_analysis(
            analysis_id,
            expected={analysis.preparation_status},
            updated_at=_timestamp(),
            scene_id=scene.scene_id,
            configuration_id=None,
            preparation_status=HistoricalPreparationStatus.ASSETS_READY,
        )
        if updated is None:
            app.state.repository.remove_scene(scene.scene_id)
            raise ApiError("WE0202", "historical analysis changed during upload")
        if analysis.scene_id is not None:
            app.state.repository.remove_scene(analysis.scene_id)
        return JSONResponse(
            status_code=201,
            content=_envelope(request, data=_historical_scene_payload(scene)),
        )

    @app.get(
        "/files/historical-scenes/{analysis_id}",
        dependencies=[Depends(authorize)],
    )
    async def get_historical_scene_file(analysis_id: str) -> Response:
        analysis = _require_historical_analysis(app.state.repository, analysis_id)
        scene = (
            app.state.repository.get_scene(analysis.scene_id)
            if analysis.scene_id is not None
            else None
        )
        if scene is None:
            raise ApiError("WE0201", "historical scene is not uploaded", status_code=404)
        return Response(content=scene.content, media_type=scene.media_type)

    @app.get(
        "/api/work-efficiency/v1/historical-analyses/{analysis_id}/video/preview",
        dependencies=[Depends(authorize)],
    )
    async def get_historical_video_preview(
        analysis_id: str,
        timestamp_sec: Annotated[float, Query(alias="timestampSec", ge=0)] = 0.0,
    ) -> Response:
        analysis = _require_historical_analysis(app.state.repository, analysis_id)
        video = _historical_video(app.state.repository, analysis)
        if video.duration_sec > 0 and timestamp_sec >= video.duration_sec:
            raise ApiError(
                "WE0101",
                "timestampSec exceeds video duration",
                data={"timestampSec": timestamp_sec, "durationSec": video.duration_sec},
            )
        content = await asyncio.to_thread(_read_video_preview, video, timestamp_sec)
        return Response(content=content, media_type="image/jpeg")

    @app.put(
        "/api/work-efficiency/v1/historical-analyses/{analysis_id}/configuration",
        dependencies=[Depends(authorize)],
    )
    async def put_historical_configuration(
        request: Request,
        analysis_id: str,
        body: HistoricalConfigurationRequest,
    ) -> JSONResponse:
        analysis = _require_editable_historical_analysis(
            app.state.repository, analysis_id
        )
        if analysis.video_id is None or analysis.scene_id is None:
            raise ApiError(
                "WE0202",
                "historical video and scene are required",
                data={"missingResources": _missing_historical_resources(analysis)},
            )
        video = _historical_video(app.state.repository, analysis)
        scene = app.state.repository.get_scene(analysis.scene_id)
        if scene is None:
            raise ApiError("WE0201", "historical scene is unavailable")
        if (
            video.duration_sec > 0
            and body.calibration_frame_timestamp_sec >= video.duration_sec
        ):
            raise ApiError("WE0101", "calibrationFrameTimestampSec exceeds video duration")
        _, calibration = _calculate_historical_configuration(
            body, video, scene, max_mean_reprojection_error_px
        )
        configuration_id = analysis.configuration_id or _new_id("hist_cfg")
        calibration_version = _new_id("cal")
        area_version = _new_id("area")
        data = {
            "historicalConfigurationId": configuration_id,
            "calibrationVersion": calibration_version,
            "areaVersion": area_version,
            "quality": {
                "meanErrorPx": calibration.mean_error_px,
                "maxErrorPx": calibration.max_error_px,
            },
            "preparationStatus": HistoricalPreparationStatus.READY.value,
        }
        payload = body.model_dump(by_alias=True)
        payload["sceneId"] = scene.scene_id
        payload["videoId"] = video.video_id
        app.state.repository.add_historical_configuration(
            ConfigurationRecord(
                configuration_id,
                calibration_version,
                area_version,
                {"request": payload, "homography": calibration.homography, "response": data},
                configuration_type="HISTORICAL",
                analysis_id=analysis_id,
            )
        )
        updated = app.state.repository.update_historical_analysis(
            analysis_id,
            expected={analysis.preparation_status},
            updated_at=_timestamp(),
            configuration_id=configuration_id,
            preparation_status=HistoricalPreparationStatus.READY,
        )
        if updated is None:
            app.state.repository.remove_historical_configuration(configuration_id)
            raise ApiError("WE0202", "historical analysis changed during configuration")
        return JSONResponse(content=_envelope(request, data=data))

    @app.post(
        "/api/work-efficiency/v1/historical-analyses/{analysis_id}/tasks",
        dependencies=[Depends(authorize)],
    )
    async def create_historical_task(
        request: Request,
        analysis_id: str,
        body: HistoricalTaskCreateRequest | None = None,
    ) -> JSONResponse:
        analysis = _require_historical_analysis(app.state.repository, analysis_id)
        missing = _missing_historical_resources(analysis)
        video = (
            app.state.repository.get_video(analysis.video_id)
            if analysis.video_id is not None
            else None
        )
        scene = (
            app.state.repository.get_scene(analysis.scene_id)
            if analysis.scene_id is not None
            else None
        )
        configuration = (
            app.state.repository.get_historical_configuration(analysis.configuration_id)
            if analysis.configuration_id is not None
            else None
        )
        if video is None or video.analysis_id != analysis_id:
            missing.append("VIDEO")
        if scene is None or scene.analysis_id != analysis_id:
            missing.append("SCENE")
        if (
            configuration is None
            or configuration.payload.get("response", {}).get("status") == "DELETED"
            or configuration.configuration_type != "HISTORICAL"
            or configuration.analysis_id != analysis_id
        ):
            missing.append("CONFIGURATION")
        missing = list(dict.fromkeys(missing))
        if missing:
            raise ApiError(
                "WE0202",
                "historical analysis is not ready",
                data={"missingResources": missing},
            )
        now = _timestamp()
        task = TaskRecord(
            task_id=_new_id("task"),
            input_type="HISTORICAL_VIDEO",
            camera_id=None,
            video_id=analysis.video_id,
            configuration_id=analysis.configuration_id or "",
            persist_pose_and_trajectory=(
                True if body is None else body.options.persist_pose_and_trajectory
            ),
            status=TaskStatus.CREATED,
            created_at=now,
            updated_at=now,
            analysis_id=analysis_id,
            configuration_snapshot=deepcopy(configuration),
            recording_started_at=(
                body.recording_started_at.isoformat()
                if body is not None and body.recording_started_at is not None else None
            ),
        )
        try:
            app.state.repository.add_historical_task(analysis_id, task, updated_at=now)
        except TaskConflictError as error:
            raise ApiError("WE0203", str(error), status_code=409) from error
        except ValueError as error:
            raise ApiError("WE0202", str(error), status_code=409) from error
        if app.state.task_manager is not None:
            task = app.state.task_manager.start(task.task_id)
        return JSONResponse(
            status_code=201,
            content=_envelope(request, data=_task_payload(task)),
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
        realtime = body.input_type == "CAMERA_STREAM"
        if (configuration is None
                or configuration.payload.get("response", {}).get("status", "ACTIVE") != "ACTIVE"
                or (realtime and (configuration.configuration_type != "REALTIME"
                                 or configuration.analysis_id is not None))):
            raise ApiError(
                "WE0202", f"configuration is not active for {body.input_type}: {body.configuration_id}",
                status_code=409 if body.input_type == "VIDEO_FILE" else 400,
            )
        _validate_task_resource(app.state.repository, body, configuration)
        recording_started_at = None
        if not realtime:
            video = _require_registered_video(app.state.repository, body.video_id or "")
            recording_started_at = _resolve_recording_start(body.recording_started_at, video)
        now = _timestamp()
        task = TaskRecord(
            task_id=_new_id("task"),
            input_type=body.input_type,
            camera_id=configuration.payload["request"].get("cameraId"),
            video_id=body.video_id,
            configuration_id=body.configuration_id,
            persist_pose_and_trajectory=body.options.persist_pose_and_trajectory,
            status=TaskStatus.CREATED,
            created_at=now,
            updated_at=now,
            recording_started_at=recording_started_at,
            configuration_snapshot=deepcopy(configuration),
        )
        try:
            app.state.repository.claim_task(task, task.camera_id or "")
        except ValueError as error:
            raise ApiError("WE0203", str(error), status_code=409) from error
        if app.state.task_manager is not None:
            task = app.state.task_manager.start(task.task_id)
        return JSONResponse(
            status_code=200 if body.input_type == "VIDEO_FILE" else 201,
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
        data = _task_payload(task)
        data["action"] = body.action
        return JSONResponse(content=_envelope(request, data=data))

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
            "analyzedFrames": snapshot.analyzed_frames,
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

    @app.get("/api/work-efficiency/v1/tasks", dependencies=[Depends(authorize)])
    async def list_saved_tasks(request: Request) -> JSONResponse:
        return JSONResponse(content=_envelope(request, data={"items": [
            _task_payload(task) for task in app.state.repository.list_tasks()]}))

    @app.get("/api/work-efficiency/v1/tasks/{task_id}/frames", dependencies=[Depends(authorize)])
    async def get_saved_frames(
        request: Request, task_id: str,
        after_seq: Annotated[int, Query(alias="afterSeq", ge=-1)] = -1,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> JSONResponse:
        _require_task(app.state.repository, task_id)
        store = app.state.analysis_result_store
        if not hasattr(store, "read_frames"):
            raise ApiError("WE0201", "persistent result store is not enabled")
        return JSONResponse(content=_envelope(request, data=store.read_frames(task_id, after_seq, limit)))

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
                <= _datetime_epoch(datetime.fromisoformat(task.recording_started_at or task.created_at))
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
                if task.status in TERMINAL_TASK_STATUSES:
                    # Terminal status is committed after the final result. Read the
                    # snapshot again so a simultaneous completion cannot skip it.
                    latest = app.state.analysis_result_store.get(task_id).latest_frame
                    if latest is not None and latest.frame_seq > last_sequence:
                        await websocket.send_json(_websocket_frame_payload(task, latest))
                    if task.status == TaskStatus.FAILED:
                        await websocket.send_json({"type": "ERROR", "taskId": task_id,
                            "code": task.failure_code or "WE0301",
                            "message": task.failure_message or "analysis failed"})
                    await websocket.send_json({"type": "TASK_STATUS", "taskId": task_id,
                        "taskStatus": task.status.value, "endedAt": task.updated_at,
                        "error": None if task.status != TaskStatus.FAILED else {
                            "code": task.failure_code or "WE0301",
                            "message": task.failure_message or "analysis failed"}})
                    await websocket.close(code=1000)
                    break
        except WebSocketDisconnect:
            pass
        finally:
            subscription.close()

    web_directory = Path(__file__).resolve().parents[1] / "web"
    if web_directory.is_dir():
        app.mount("/test-ui", StaticFiles(directory=web_directory, html=True), name="test-ui")

    return app


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_task_input(body: TaskCreateRequest) -> None:
    if body.input_type == "VIDEO_FILE":
        if body.video_id is None or body.camera_id is not None:
            raise ApiError("WE0101", "VIDEO_FILE requires videoId and forbids cameraId", status_code=422)
        return
    valid = body.camera_id is not None and body.video_id is None
    if not valid:
        raise ApiError(
            "WE0101",
            "CAMERA_STREAM requires cameraId and does not accept videoId",
        )


def _validate_task_resource(
    repository: InMemoryRepository,
    body: TaskCreateRequest,
    configuration: ConfigurationRecord,
) -> None:
    if body.input_type == "VIDEO_FILE":
        video = _require_registered_video(repository, body.video_id or "")
        if not Path(video.path).is_file():
            raise ApiError("WE0201", "registered video source is not readable", status_code=404)
        return
    configured_camera_id = configuration.payload.get("request", {}).get("cameraId")
    camera = repository.get_camera(body.camera_id or "")
    if camera is None:
        raise ApiError("WE0201", f"camera is not registered: {body.camera_id}")
    source_camera_id = camera.camera_id
    if configured_camera_id != source_camera_id:
        raise ApiError(
            "WE0202",
            "input source and configuration belong to different camera viewpoints",
            data={
                "inputType": body.input_type,
                "sourceId": body.camera_id,
                "cameraId": body.camera_id,
                "configurationId": configuration.configuration_id,
                "sourceCameraId": source_camera_id,
                "configurationCameraId": configured_camera_id,
                "incompatibleReasons": ["CAMERA_ID_MISMATCH"],
                "suggestedAction": "RECALIBRATE",
            },
        )


def _resolve_recording_start(value: datetime | None, video: VideoRecord) -> str:
    if value is None and video.captured_at:
        try:
            value = datetime.fromisoformat(video.captured_at.replace("Z", "+00:00"))
        except ValueError:
            value = None
    if value is None or value.utcoffset() is None:
        raise ApiError("WE0101", "historical video requires recordingStartedAt with timezone",
                       status_code=422)
    return value.isoformat()


def _require_registered_video(repository: InMemoryRepository, video_id: str) -> VideoRecord:
    video = repository.get_video(video_id)
    if video is None or video.analysis_id is not None:
        raise ApiError("WE0201", f"videoId is not registered: {video_id}", status_code=404)
    return video


def _video_payload(video: VideoRecord, *, deduplicated: bool = False) -> dict:
    return {
        "videoId": video.video_id, "cameraId": video.camera_id,
        "originalFileName": video.original_file_name, "contentType": "video/mp4",
        "sizeBytes": video.size_bytes, "sha256": video.sha256,
        "width": video.width, "height": video.height, "fps": video.fps,
        "frameCount": video.frame_count, "durationSec": video.duration_sec,
        "capturedAt": video.captured_at, "uploadedAt": video.uploaded_at,
        "status": "READY", "deduplicated": deduplicated,
        "previewUrl": f"/api/work-efficiency/v1/videos/{video.video_id}/preview",
    }


def _historical_video_payload(video: VideoRecord) -> dict:
    return {
        "videoId": video.video_id,
        "width": video.width,
        "height": video.height,
        "fps": video.fps,
        "frameCount": video.frame_count,
        "durationSec": video.duration_sec,
        "previewUrl": (
            f"/api/work-efficiency/v1/historical-analyses/"
            f"{video.analysis_id}/video/preview"
        ),
    }


def _historical_scene_payload(scene: SceneRecord) -> dict:
    return {
        "sceneId": scene.scene_id,
        "imageUrl": scene.image_url,
        "width": scene.width,
        "height": scene.height,
    }


def _historical_analysis_payload(app: FastAPI, analysis: HistoricalAnalysisRecord) -> dict:
    video = (
        app.state.repository.get_video(analysis.video_id)
        if analysis.video_id is not None
        else None
    )
    scene = (
        app.state.repository.get_scene(analysis.scene_id)
        if analysis.scene_id is not None
        else None
    )
    configuration = (
        app.state.repository.get_historical_configuration(analysis.configuration_id)
        if analysis.configuration_id is not None
        else None
    )
    return {
        "analysisId": analysis.analysis_id,
        "preparationStatus": analysis.preparation_status.value,
        "video": None if video is None else _historical_video_payload(video),
        "scene": None if scene is None else _historical_scene_payload(scene),
        "configuration": None
        if configuration is None
        else {
            "historicalConfigurationId": configuration.configuration_id,
            "calibrationVersion": configuration.calibration_version,
            "areaVersion": configuration.area_version,
            "quality": configuration.payload.get("response", {}).get("quality"),
        },
        "latestTaskId": analysis.latest_task_id,
        "createdAt": analysis.created_at,
        "updatedAt": analysis.updated_at,
    }


def _require_historical_analysis(
    repository: InMemoryRepository, analysis_id: str
) -> HistoricalAnalysisRecord:
    analysis = repository.get_historical_analysis(analysis_id)
    if analysis is None:
        raise ApiError(
            "WE0201", f"historical analysis not found: {analysis_id}", status_code=404
        )
    return analysis


def _require_editable_historical_analysis(
    repository: InMemoryRepository, analysis_id: str
) -> HistoricalAnalysisRecord:
    analysis = _require_historical_analysis(repository, analysis_id)
    if analysis.preparation_status == HistoricalPreparationStatus.LOCKED:
        raise ApiError(
            "WE0202",
            "historical analysis inputs are locked",
            status_code=409,
            data={"suggestedAction": "CREATE_NEW_HISTORICAL_ANALYSIS"},
        )
    return analysis


def _historical_video(
    repository: InMemoryRepository, analysis: HistoricalAnalysisRecord
) -> VideoRecord:
    video = (
        repository.get_video(analysis.video_id)
        if analysis.video_id is not None
        else None
    )
    if video is None or video.analysis_id != analysis.analysis_id:
        raise ApiError("WE0201", "historical video is unavailable", status_code=404)
    return video


def _missing_historical_resources(analysis: HistoricalAnalysisRecord) -> list[str]:
    missing = []
    if analysis.video_id is None:
        missing.append("VIDEO")
    if analysis.scene_id is None:
        missing.append("SCENE")
    if analysis.configuration_id is None:
        missing.append("CONFIGURATION")
    return missing


def _discard_historical_configuration(
    repository: InMemoryRepository, analysis: HistoricalAnalysisRecord
) -> None:
    if analysis.configuration_id is not None:
        repository.remove_historical_configuration(analysis.configuration_id)


def _remove_replaced_video(app: FastAPI, video_id: str | None) -> None:
    if video_id is None:
        return
    old = app.state.repository.get_video(video_id)
    if old is None:
        return
    old_path = Path(old.path).resolve()
    if old_path.is_relative_to(app.state.video_directory):
        old_path.unlink(missing_ok=True)
    app.state.repository.remove_video(video_id)


async def _decode_scene_upload(file: UploadFile) -> tuple[bytes, int, int]:
    if file.content_type not in ALLOWED_SCENE_MEDIA_TYPES:
        raise ApiError("WE0101", "scene file must be PNG or JPEG")
    content = await file.read(MAX_SCENE_FILE_BYTES + 1)
    if len(content) > MAX_SCENE_FILE_BYTES:
        raise ApiError("WE0101", "scene file exceeds 20 MB")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ApiError("WE0101", "scene file is not a valid image")
    height, width = image.shape[:2]
    return content, int(width), int(height)


def _calculate_historical_configuration(
    body: HistoricalConfigurationRequest,
    video: VideoRecord,
    scene: SceneRecord,
    max_mean_reprojection_error_px: float,
) -> tuple[tuple[WorkArea, ...], Any]:
    try:
        if any(
            pair.video_point.x < 0
            or pair.video_point.y < 0
            or pair.video_point.x > video.width
            or pair.video_point.y > video.height
            for pair in body.feature_pairs
        ):
            raise ValueError("calibration video points must lie within the video frame")
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
                    (pair.video_point.x, pair.video_point.y),
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
    return areas, calibration


def _read_video_preview(video: VideoRecord, timestamp_sec: float) -> bytes:
    capture = cv2.VideoCapture(video.path)
    try:
        if not capture.isOpened():
            raise ApiError(
                "WE0201",
                "video cannot be opened for preview",
                status_code=404,
                data={"videoId": video.video_id},
            )
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
        success, frame = capture.read()
        if not success or frame is None:
            raise ApiError(
                "WE0201",
                "video frame cannot be decoded for preview",
                data={"videoId": video.video_id, "timestampSec": timestamp_sec},
            )
        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            raise ApiError(
                "WE0201",
                "video preview cannot be encoded",
                data={"videoId": video.video_id, "timestampSec": timestamp_sec},
            )
        return buffer.tobytes()
    finally:
        capture.release()


def _require_task(repository: InMemoryRepository, task_id: str) -> TaskRecord:
    task = repository.get_task(task_id)
    if task is None:
        raise ApiError("WE0201", f"task not found: {task_id}", status_code=404)
    return task


def _task_payload(task: TaskRecord) -> dict:
    payload = {
        "taskId": task.task_id,
        "inputType": task.input_type,
        "cameraId": task.camera_id,
        "videoId": task.video_id,
        "configurationId": task.configuration_id,
        "status": "QUEUED" if task.input_type == "VIDEO_FILE" and task.status in {
            TaskStatus.CREATED, TaskStatus.STARTING
        } else task.status.value,
        "options": {"persistPoseAndTrajectory": task.persist_pose_and_trajectory},
        "createdAt": task.created_at,
        "updatedAt": task.updated_at,
        "failureCode": task.failure_code,
        "failureMessage": task.failure_message,
        "analysisId": task.analysis_id,
        "recordingStartedAt": task.recording_started_at,
    }
    if task.input_type == "HISTORICAL_VIDEO":
        payload["historicalConfigurationId"] = task.configuration_id
    if task.configuration_snapshot is not None:
        payload["configurationSnapshot"] = {
            "configurationId": task.configuration_snapshot.configuration_id,
            **deepcopy(task.configuration_snapshot.payload),
        }
    return payload


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
    task_started = datetime.fromisoformat(task.recording_started_at or task.created_at)
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
