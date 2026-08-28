from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .models import Keypoint, PoseObservation


def project_point(
    point_px: tuple[float, float], homography: Sequence[Sequence[float]]
) -> tuple[float, float] | None:
    matrix = np.asarray(homography, dtype=float).reshape(3, 3)
    source = np.asarray([point_px[0], point_px[1], 1.0], dtype=float)
    projected = matrix @ source
    if abs(projected[2]) < 1e-12 or not np.isfinite(projected).all():
        return None
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def person_anchor(observation: PoseObservation, threshold: float) -> tuple[float, float]:
    by_name: dict[str, Keypoint] = {point.name: point for point in observation.keypoints}
    ankles = [
        by_name[name]
        for name in ("left_ankle", "right_ankle")
        if name in by_name and by_name[name].score >= threshold
    ]
    if ankles:
        return (
            sum(point.x for point in ankles) / len(ankles),
            sum(point.y for point in ankles) / len(ankles),
        )
    x1, _, x2, y2 = observation.bbox_xyxy
    return (x1 + x2) / 2.0, y2


def point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    """Boundary-inclusive ray casting without requiring Shapely at runtime."""
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-9 and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
            return True
        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside
