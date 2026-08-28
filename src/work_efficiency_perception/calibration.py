from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .models import CalibrationPointPair, CalibrationResult, WorkArea


def calibrate_homography(
    point_pairs: Sequence[CalibrationPointPair],
    *,
    scene_width: int,
    scene_height: int,
) -> CalibrationResult:
    """Calculate a camera-pixel to normalized-scene homography.

    Reprojection quality is reported in scene-image pixels so that it remains
    understandable to callers even though the returned matrix targets normalized
    scene coordinates.
    """
    if len(point_pairs) < 4:
        raise ValueError("at least four calibration point pairs are required")
    if scene_width <= 0 or scene_height <= 0:
        raise ValueError("scene dimensions must be greater than zero")

    camera_points = np.asarray([pair.camera_point for pair in point_pairs], dtype=np.float64)
    scene_points_px = np.asarray(
        [pair.scene_point_px for pair in point_pairs], dtype=np.float64
    )
    if np.any(scene_points_px < 0) or np.any(scene_points_px[:, 0] > scene_width) or np.any(
        scene_points_px[:, 1] > scene_height
    ):
        raise ValueError("calibration scene points must lie within the scene image")
    if _contains_duplicate_points(camera_points) or _contains_duplicate_points(scene_points_px):
        raise ValueError("calibration points must be unique")

    homography_px, _ = cv2.findHomography(camera_points, scene_points_px, method=0)
    if homography_px is None or not np.isfinite(homography_px).all():
        raise ValueError("calibration points do not produce a valid homography")
    if abs(np.linalg.det(homography_px)) < 1e-12:
        raise ValueError("calibration homography is degenerate")

    projected = cv2.perspectiveTransform(
        camera_points.reshape(-1, 1, 2), homography_px
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - scene_points_px, axis=1)
    normalization = np.asarray(
        [[1.0 / scene_width, 0.0, 0.0], [0.0, 1.0 / scene_height, 0.0], [0.0, 0.0, 1.0]]
    )
    homography_normalized = normalization @ homography_px
    homography_normalized /= homography_normalized[2, 2]

    return CalibrationResult(
        homography=tuple(tuple(float(value) for value in row) for row in homography_normalized),
        mean_error_px=float(errors.mean()),
        max_error_px=float(errors.max()),
    )


def validate_work_areas(areas: Sequence[WorkArea]) -> None:
    if not areas:
        raise ValueError("at least one work area is required")
    area_ids = [area.area_id for area in areas]
    if len(area_ids) != len(set(area_ids)):
        raise ValueError("work-area IDs must be unique")
    for area in areas:
        if not area.area_id.strip():
            raise ValueError("work-area ID must not be blank")
        if _polygon_self_intersects(area.polygon):
            raise ValueError(f"work-area polygon self-intersects: {area.area_id}")
        if abs(_signed_polygon_area(area.polygon)) < 1e-9:
            raise ValueError(f"work-area polygon has zero area: {area.area_id}")


def _contains_duplicate_points(points: np.ndarray) -> bool:
    return len({(float(x), float(y)) for x, y in points}) != len(points)


def _signed_polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    return sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(polygon, (*polygon[1:], polygon[0]))
    ) / 2.0


def _polygon_self_intersects(polygon: Sequence[tuple[float, float]]) -> bool:
    edge_count = len(polygon)
    for first_index in range(edge_count):
        first = (polygon[first_index], polygon[(first_index + 1) % edge_count])
        for second_index in range(first_index + 1, edge_count):
            if second_index in {first_index, (first_index + 1) % edge_count}:
                continue
            if first_index == 0 and second_index == edge_count - 1:
                continue
            second = (polygon[second_index], polygon[(second_index + 1) % edge_count])
            if _segments_intersect(*first, *second):
                return True
    return False


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    def orientation(
        p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]
    ) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(
        p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]
    ) -> bool:
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    values = (
        orientation(a, b, c),
        orientation(a, b, d),
        orientation(c, d, a),
        orientation(c, d, b),
    )
    if values[0] * values[1] < 0 and values[2] * values[3] < 0:
        return True
    return (
        (abs(values[0]) <= 1e-12 and on_segment(a, c, b))
        or (abs(values[1]) <= 1e-12 and on_segment(a, d, b))
        or (abs(values[2]) <= 1e-12 and on_segment(c, a, d))
        or (abs(values[3]) <= 1e-12 and on_segment(c, b, d))
    )
