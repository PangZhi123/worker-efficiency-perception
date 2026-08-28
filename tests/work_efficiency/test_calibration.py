import pytest

from work_efficiency_perception import (
    CalibrationPointPair,
    EngineConfig,
    WorkArea,
    calibrate_homography,
    validate_work_areas,
)
from work_efficiency_perception.geometry import project_point


def test_calibration_maps_camera_pixels_to_normalized_scene_coordinates():
    pairs = (
        CalibrationPointPair((100, 50), (0, 0)),
        CalibrationPointPair((300, 50), (1000, 0)),
        CalibrationPointPair((300, 250), (1000, 500)),
        CalibrationPointPair((100, 250), (0, 500)),
    )

    result = calibrate_homography(pairs, scene_width=1000, scene_height=500)

    assert project_point((200, 150), result.homography) == pytest.approx((0.5, 0.5))
    assert result.mean_error_px == pytest.approx(0, abs=1e-6)
    assert result.max_error_px == pytest.approx(0, abs=1e-6)


def test_calibration_rejects_insufficient_or_duplicate_points():
    pair = CalibrationPointPair((0, 0), (0, 0))
    with pytest.raises(ValueError, match="at least four"):
        calibrate_homography((pair,) * 3, scene_width=100, scene_height=100)

    pairs = (
        pair,
        CalibrationPointPair((0, 0), (100, 0)),
        CalibrationPointPair((100, 100), (100, 100)),
        CalibrationPointPair((0, 100), (0, 100)),
    )
    with pytest.raises(ValueError, match="unique"):
        calibrate_homography(pairs, scene_width=100, scene_height=100)


def test_work_area_validation_rejects_duplicate_ids_and_self_intersection():
    square = WorkArea("zone", ((0, 0), (1, 0), (1, 1), (0, 1)))
    with pytest.raises(ValueError, match="IDs must be unique"):
        validate_work_areas((square, square))

    crossed = WorkArea("crossed", ((0, 0), (1, 1), (0, 1), (1, 0)))
    with pytest.raises(ValueError, match="self-intersects"):
        validate_work_areas((crossed,))


def test_work_area_and_engine_config_reject_invalid_frontend_values():
    with pytest.raises(ValueError, match="normalized"):
        WorkArea("zone", ((0, 0), (2, 0), (0, 1)))
    with pytest.raises(ValueError, match="non-negative"):
        WorkArea("zone", ((0, 0), (1, 0), (0, 1)), required_team_work_duration_sec=-1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        EngineConfig(keypoint_confidence=1.1)
