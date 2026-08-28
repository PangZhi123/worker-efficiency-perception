from __future__ import annotations

import os
from pathlib import Path

from .analysis import AnalysisResultStore, VideoAnalysisWorker, build_yolo_tracker_factory
from .api import create_app
from .api.repository import InMemoryRepository
from .tracker import PoseTrackerConfig


def build_app():
    repository = InMemoryRepository()
    result_store = AnalysisResultStore()
    configured_path = os.getenv("WORK_EFFICIENCY_MODEL_PATH")
    model_path = Path(configured_path) if configured_path else Path("model/yolo11m-pose.pt")
    worker = None
    if model_path.is_file():
        device = os.getenv("WORK_EFFICIENCY_DEVICE")
        tracker_config = PoseTrackerConfig(
            model_path=model_path,
            device=None if device in {None, ""} else device,
            image_size=int(os.getenv("WORK_EFFICIENCY_IMAGE_SIZE", "1280")),
        )
        worker = VideoAnalysisWorker(
            repository,
            build_yolo_tracker_factory(tracker_config),
            result_store,
        )
    return create_app(
        repository=repository,
        analysis_worker=worker,
        analysis_result_store=result_store,
        bearer_token=os.getenv("WORK_EFFICIENCY_BEARER_TOKEN"),
        video_directory=os.getenv("WORK_EFFICIENCY_VIDEO_DIRECTORY", "data/videos"),
    )


app = build_app()
