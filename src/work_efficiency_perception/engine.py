from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .calibration import validate_work_areas
from .geometry import person_anchor, point_in_polygon, project_point
from .models import ComplianceInterval, EngineConfig, PoseObservation, WorkArea, ZoneState


@dataclass(slots=True)
class _TrackState:
    zone_state: ZoneState = ZoneState.UNKNOWN
    candidate_area_id: str | None = None
    candidate_since: float = 0.0
    area_id: str | None = None
    last_seen: float = 0.0


class WorkEfficiencyEngine:
    """Stateful projection, zone hysteresis and compliant-duration accumulator."""

    def __init__(self, areas: Sequence[WorkArea], homography: Sequence[Sequence[float]], config: EngineConfig | None = None) -> None:
        validate_work_areas(areas)
        self.areas = tuple(areas)
        self.homography = homography
        self.config = config or EngineConfig()
        self._tracks: dict[int, _TrackState] = {}
        self._qualified_duration = {area.area_id: 0.0 for area in areas}
        self._headcount_compliant: dict[str, bool | None] = {
            area.area_id: None for area in areas
        }
        self._interval_start: dict[str, float | None] = {area.area_id: None for area in areas}
        self._completed_intervals: dict[str, list[ComplianceInterval]] = {
            area.area_id: [] for area in areas
        }
        self._last_timestamp: float | None = None

    def process(self, observations: Sequence[PoseObservation], timestamp_sec: float) -> dict:
        if self._last_timestamp is not None and timestamp_sec < self._last_timestamp:
            raise ValueError("timestamps must be monotonic")
        people = []
        seen = set()
        for observation in observations:
            seen.add(observation.track_id)
            anchor = person_anchor(observation, self.config.keypoint_confidence)
            scene_point = project_point(anchor, self.homography)
            observed_area = next((area.area_id for area in self.areas if scene_point is not None and point_in_polygon(scene_point, area.polygon)), None)
            state = self._update_track(observation.track_id, observed_area, timestamp_sec)
            people.append({
                "trackId": str(observation.track_id),
                "confidence": observation.confidence,
                "bboxPx": list(observation.bbox_xyxy),
                "bboxNorm": [
                    observation.bbox_xyxy[0] / observation.frame_width,
                    observation.bbox_xyxy[1] / observation.frame_height,
                    observation.bbox_xyxy[2] / observation.frame_width,
                    observation.bbox_xyxy[3] / observation.frame_height,
                ],
                "keypoints": [vars_keypoint(point) for point in observation.keypoints],
                "anchorPx": {"x": anchor[0], "y": anchor[1]},
                "scenePointNorm": None if scene_point is None else {"x": scene_point[0], "y": scene_point[1]},
                "areaId": state.area_id,
                "personZoneState": state.zone_state.value,
                "workState": "WORKING" if state.zone_state == ZoneState.INSIDE else ("OUTSIDE" if state.zone_state == ZoneState.OUTSIDE else "UNKNOWN"),
            })
        self._expire_missing(seen, timestamp_sec)
        counts = {area.area_id: 0 for area in self.areas}
        for state in self._tracks.values():
            if state.zone_state == ZoneState.INSIDE and state.area_id in counts:
                counts[state.area_id] += 1
        delta = 0.0 if self._last_timestamp is None else min(timestamp_sec - self._last_timestamp, self.config.max_frame_gap_sec)
        areas_payload = []
        for area in self.areas:
            expected = area.expected_worker_count
            compliant = None if expected is None else counts[area.area_id] >= expected
            # The interval ending at this timestamp belongs to the state observed
            # at the previous timestamp. Using the newly observed state here would
            # credit time before entry/team-size confirmation.
            if self._headcount_compliant[area.area_id]:
                self._qualified_duration[area.area_id] += delta
            self._update_compliance_interval(area.area_id, compliant, timestamp_sec)
            self._headcount_compliant[area.area_id] = compliant
            required = area.required_team_work_duration_sec
            efficiency = "NOT_CONFIGURED" if required is None else ("MET" if self._qualified_duration[area.area_id] >= required else "NOT_MET")
            areas_payload.append({
                "areaId": area.area_id, "actualWorkingCount": counts[area.area_id],
                "expectedWorkerCount": expected, "headcountCompliant": compliant,
                "qualifiedTeamWorkDurationSec": self._qualified_duration[area.area_id],
                "requiredTeamWorkDurationSec": required, "efficiencyStatus": efficiency,
            })
        self._last_timestamp = timestamp_sec
        return {"persons": people, "areas": areas_payload}

    def compliance_intervals(
        self, area_id: str, *, include_open: bool = True
    ) -> tuple[ComplianceInterval, ...]:
        if area_id not in self._completed_intervals:
            raise KeyError(f"unknown work area: {area_id}")
        intervals = list(self._completed_intervals[area_id])
        state = self._headcount_compliant[area_id]
        start = self._interval_start[area_id]
        if include_open and state is not None and start is not None:
            intervals.append(ComplianceInterval(start, None, state))
        return tuple(intervals)

    def _update_compliance_interval(
        self, area_id: str, compliant: bool | None, now: float
    ) -> None:
        previous = self._headcount_compliant[area_id]
        if compliant is None:
            if previous is not None and self._interval_start[area_id] is not None:
                self._completed_intervals[area_id].append(
                    ComplianceInterval(self._interval_start[area_id], now, previous)
                )
            self._interval_start[area_id] = None
        elif previous is None:
            self._interval_start[area_id] = now
        elif compliant != previous:
            self._completed_intervals[area_id].append(
                ComplianceInterval(self._interval_start[area_id], now, previous)
            )
            self._interval_start[area_id] = now

    def _update_track(self, track_id: int, observed_area: str | None, now: float) -> _TrackState:
        state = self._tracks.setdefault(track_id, _TrackState(last_seen=now))
        state.last_seen = now
        if state.zone_state in (ZoneState.UNKNOWN, ZoneState.OUTSIDE):
            if observed_area is None:
                state.zone_state, state.area_id, state.candidate_area_id = ZoneState.OUTSIDE, None, None
            elif state.candidate_area_id != observed_area:
                state.zone_state, state.candidate_area_id, state.candidate_since = ZoneState.ENTERING, observed_area, now
                if self.config.entry_detection_duration_sec == 0:
                    state.zone_state, state.area_id, state.candidate_area_id = ZoneState.INSIDE, observed_area, None
            elif now - state.candidate_since >= self.config.entry_detection_duration_sec:
                state.zone_state, state.area_id, state.candidate_area_id = ZoneState.INSIDE, observed_area, None
        elif state.zone_state == ZoneState.ENTERING:
            if observed_area != state.candidate_area_id:
                state.zone_state, state.candidate_area_id = ZoneState.OUTSIDE, None
            elif now - state.candidate_since >= self.config.entry_detection_duration_sec:
                state.zone_state, state.area_id, state.candidate_area_id = ZoneState.INSIDE, observed_area, None
        elif state.zone_state == ZoneState.INSIDE:
            if observed_area == state.area_id:
                state.candidate_area_id = None
            else:
                state.zone_state, state.candidate_since = ZoneState.LEAVING, now
        elif state.zone_state == ZoneState.LEAVING:
            if observed_area == state.area_id:
                state.zone_state = ZoneState.INSIDE
            elif now - state.candidate_since >= self.config.exit_detection_duration_sec:
                state.zone_state, state.area_id = ZoneState.OUTSIDE, None
        return state

    def _expire_missing(self, seen: set[int], now: float) -> None:
        for track_id, state in list(self._tracks.items()):
            if track_id not in seen and now - state.last_seen > self.config.lost_track_tolerance_sec:
                del self._tracks[track_id]


def vars_keypoint(point: object) -> dict:
    return {"name": point.name, "x": point.x, "y": point.y, "score": point.score}
