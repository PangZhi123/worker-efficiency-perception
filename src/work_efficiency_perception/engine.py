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
    entered_from_outside: bool = False


@dataclass(slots=True)
class _TeamSlot:
    area_id: str
    track_id: int | None
    expires_at: float | None = None


class WorkEfficiencyEngine:
    """Stateful projection, zone hysteresis and compliant-duration accumulator."""

    def __init__(self, areas: Sequence[WorkArea], homography: Sequence[Sequence[float]], config: EngineConfig | None = None) -> None:
        validate_work_areas(areas)
        self.areas = tuple(areas)
        self.homography = homography
        self.config = config or EngineConfig()
        self._tracks: dict[int, _TrackState] = {}
        self._team_slots: list[_TeamSlot] = []
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
        seen = {observation.track_id for observation in observations}
        # Once an outside observation starts departure, losing the detection
        # must not turn that departure into an indefinite in-area absence.
        for track_id, state in self._tracks.items():
            confirmed_at = state.candidate_since + self.config.exit_detection_duration_sec
            if (track_id not in seen and state.zone_state == ZoneState.LEAVING
                    and timestamp_sec >= confirmed_at):
                state.zone_state, state.area_id = ZoneState.OUTSIDE, None
                state.entered_from_outside = True
                for slot in self._team_slots:
                    if slot.track_id == track_id:
                        slot.track_id = None
                        slot.expires_at = confirmed_at + self.config.team_exit_grace_sec
        self._advance_time(timestamp_sec)
        people = []
        projected = []
        for observation in observations:
            anchor = person_anchor(observation, self.config.keypoint_confidence)
            scene_point = project_point(anchor, self.homography)
            observed_area = next((area.area_id for area in self.areas if scene_point is not None and point_in_polygon(scene_point, area.polygon)), None)
            projected.append((observation, anchor, scene_point, observed_area))

        # Process all exits before assigning entries, independently of detection order.
        for observation, _, scene_point, observed_area in projected:
            state = self._tracks.get(observation.track_id)
            previous_area = None if state is None else state.area_id
            if scene_point is None:
                # A failed projection is not evidence of crossing a boundary.
                continue
            state = self._update_track(observation.track_id, observed_area, timestamp_sec)
            if previous_area is not None and state.zone_state == ZoneState.OUTSIDE:
                for slot in self._team_slots:
                    if slot.track_id == observation.track_id:
                        slot.track_id = None
                        slot.expires_at = timestamp_sec + self.config.team_exit_grace_sec

        for observation, _, _, _ in projected:
            state = self._tracks.get(observation.track_id, _TrackState())
            if state.zone_state == ZoneState.INSIDE:
                self._assign_team_slot(observation.track_id, state, seen, timestamp_sec)

        # Rebuild person payloads after every group reassignment has completed.
        for observation, anchor, scene_point, observed_area in projected:
            state = self._tracks.get(observation.track_id, _TrackState())
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
                "workState": "WORKING" if state.zone_state in (ZoneState.INSIDE, ZoneState.LEAVING) else ("OUTSIDE" if state.zone_state == ZoneState.OUTSIDE else "UNKNOWN"),
            })
        self._expire_missing(seen, timestamp_sec)
        self._team_slots = [
            slot for slot in self._team_slots
            if slot.expires_at is None or slot.expires_at > timestamp_sec
        ]
        counts = self._team_counts()
        areas_payload = []
        for area in self.areas:
            expected = area.expected_worker_count
            compliant = None if expected is None else counts[area.area_id] >= expected
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

    def _team_counts(self) -> dict[str, int]:
        counts = {area.area_id: 0 for area in self.areas}
        for slot in self._team_slots:
            counts[slot.area_id] += 1
        return counts

    def _advance_time(self, now: float) -> None:
        """Split accounting at grace deadlines, including between sampled frames."""
        if self._last_timestamp is None:
            return
        cursor = self._last_timestamp
        credit_until = min(now, cursor + self.config.max_frame_gap_sec)
        deadlines = sorted({
            slot.expires_at for slot in self._team_slots
            if slot.expires_at is not None and slot.expires_at <= now
        })
        for boundary in [*deadlines, now]:
            delta = max(0.0, min(boundary, credit_until) - min(cursor, credit_until))
            for area in self.areas:
                if self._headcount_compliant[area.area_id]:
                    self._qualified_duration[area.area_id] += delta
            cursor = boundary
            self._team_slots = [
                slot for slot in self._team_slots
                if slot.expires_at is None or slot.expires_at > boundary
            ]
            counts = self._team_counts()
            for area in self.areas:
                expected = area.expected_worker_count
                compliant = None if expected is None else counts[area.area_id] >= expected
                self._update_compliance_interval(area.area_id, compliant, boundary)
                self._headcount_compliant[area.area_id] = compliant

    def _assign_team_slot(
        self, track_id: int, state: _TrackState, seen: set[int], now: float
    ) -> None:
        if any(slot.track_id == track_id for slot in self._team_slots):
            return
        # A new group member first fills an unexpired temporary absence.
        grace = next((slot for slot in sorted(
            self._team_slots, key=lambda slot: slot.expires_at or float("inf")
        ) if slot.area_id == state.area_id and slot.expires_at is not None
            and slot.expires_at > now), None)
        # Reappearing detections without an observed entry reuse an unseen slot.
        # An actual outside-to-inside crossing is additional headcount instead.
        missing = None
        if grace is None and not state.entered_from_outside:
            missing = next((slot for slot in self._team_slots
                if slot.area_id == state.area_id and slot.expires_at is None
                and slot.track_id not in seen), None)
        slot = grace or missing
        if slot is None:
            self._team_slots.append(_TeamSlot(state.area_id, track_id))
        else:
            old_track_id = slot.track_id
            slot.track_id, slot.expires_at = track_id, None
            if old_track_id is not None:
                self._tracks.pop(old_track_id, None)

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
                state.entered_from_outside = True
            elif state.candidate_area_id != observed_area:
                state.zone_state, state.candidate_area_id, state.candidate_since = ZoneState.ENTERING, observed_area, now
                if self.config.entry_detection_duration_sec == 0:
                    state.zone_state, state.area_id, state.candidate_area_id = ZoneState.INSIDE, observed_area, None
            elif now - state.candidate_since >= self.config.entry_detection_duration_sec:
                state.zone_state, state.area_id, state.candidate_area_id = ZoneState.INSIDE, observed_area, None
        elif state.zone_state == ZoneState.ENTERING:
            if observed_area != state.candidate_area_id:
                state.zone_state, state.candidate_area_id = ZoneState.OUTSIDE, None
                state.entered_from_outside = True
            elif now - state.candidate_since >= self.config.entry_detection_duration_sec:
                state.zone_state, state.area_id, state.candidate_area_id = ZoneState.INSIDE, observed_area, None
        elif state.zone_state == ZoneState.INSIDE:
            if observed_area == state.area_id:
                state.candidate_area_id = None
            else:
                state.zone_state, state.candidate_since = ZoneState.LEAVING, now
                if self.config.exit_detection_duration_sec == 0:
                    state.zone_state, state.area_id = ZoneState.OUTSIDE, None
                    state.entered_from_outside = True
        elif state.zone_state == ZoneState.LEAVING:
            if observed_area == state.area_id:
                state.zone_state = ZoneState.INSIDE
            elif now - state.candidate_since >= self.config.exit_detection_duration_sec:
                state.zone_state, state.area_id = ZoneState.OUTSIDE, None
                state.entered_from_outside = True
        return state

    def _expire_missing(self, seen: set[int], now: float) -> None:
        retained = {slot.track_id for slot in self._team_slots if slot.track_id is not None}
        for track_id, state in list(self._tracks.items()):
            if (track_id not in seen and track_id not in retained
                    and now - state.last_seen > self.config.lost_track_tolerance_sec):
                del self._tracks[track_id]


def vars_keypoint(point: object) -> dict:
    return {"name": point.name, "x": point.x, "y": point.y, "score": point.score}
