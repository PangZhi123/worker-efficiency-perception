import pytest

from work_efficiency_perception import EngineConfig, PoseObservation, WorkArea, WorkEfficiencyEngine
from work_efficiency_perception.models import Keypoint


def observation(track_id: int, time: float, x: float, y: float) -> PoseObservation:
    points = (Keypoint("left_ankle", x, y, 0.9), Keypoint("right_ankle", x, y, 0.9))
    return PoseObservation(int(time), time, track_id, (x - 5, y - 20, x + 5, y), 0.9, points, 100, 100)


def test_entry_hysteresis_and_qualified_duration():
    area = WorkArea("zone", ((0, 0), (1, 0), (1, 1), (0, 1)), 1, 2)
    engine = WorkEfficiencyEngine([area], [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]], EngineConfig(entry_detection_duration_sec=1, exit_detection_duration_sec=1))
    assert engine.process([observation(7, 0, 50, 50)], 0)["areas"][0]["actualWorkingCount"] == 0
    result = engine.process([observation(7, 1, 50, 50)], 1)
    assert result["areas"][0]["actualWorkingCount"] == 1
    assert result["areas"][0]["qualifiedTeamWorkDurationSec"] == 0
    result = engine.process([observation(7, 2, 50, 50)], 2)
    assert result["areas"][0]["qualifiedTeamWorkDurationSec"] == 1
    assert result["areas"][0]["efficiencyStatus"] == "NOT_MET"
    result = engine.process([observation(7, 3, 50, 50)], 3)
    assert result["areas"][0]["qualifiedTeamWorkDurationSec"] == 2
    assert result["areas"][0]["efficiencyStatus"] == "MET"


def test_qualified_duration_uses_previous_interval_state_on_exit():
    area = WorkArea("zone", ((0, 0), (1, 0), (1, 1), (0, 1)), 1, None)
    engine = WorkEfficiencyEngine(
        [area],
        [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]],
        EngineConfig(entry_detection_duration_sec=0, exit_detection_duration_sec=0,
                     team_exit_grace_sec=0),
    )

    first = engine.process([observation(1, 0, 50, 50)], 0)["areas"][0]
    assert first["headcountCompliant"] is True
    assert first["qualifiedTeamWorkDurationSec"] == 0

    exited = engine.process([observation(1, 1, 150, 150)], 1)["areas"][0]
    assert exited["headcountCompliant"] is False
    assert exited["qualifiedTeamWorkDurationSec"] == 1

    still_outside = engine.process([observation(1, 2, 150, 150)], 2)["areas"][0]
    assert still_outside["qualifiedTeamWorkDurationSec"] == 1
    completed, open_interval = engine.compliance_intervals("zone")
    assert (completed.start_sec, completed.end_sec, completed.headcount_compliant) == (
        0,
        1,
        True,
    )
    assert (open_interval.start_sec, open_interval.end_sec, open_interval.headcount_compliant) == (
        1,
        None,
        False,
    )


def test_short_exit_does_not_drop_worker():
    area = WorkArea("zone", ((0, 0), (1, 0), (1, 1), (0, 1)), 1, None)
    engine = WorkEfficiencyEngine([area], [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]], EngineConfig(entry_detection_duration_sec=0, exit_detection_duration_sec=2))
    engine.process([observation(1, 0, 50, 50)], 0)
    assert engine.process([observation(1, 1, 150, 150)], 1)["persons"][0]["personZoneState"] == "LEAVING"
    assert engine.process([observation(1, 2, 50, 50)], 2)["persons"][0]["personZoneState"] == "INSIDE"


def team_engine(**settings):
    config = dict(entry_detection_duration_sec=0, exit_detection_duration_sec=0,
                  max_frame_gap_sec=10000)
    config.update(settings)
    return WorkEfficiencyEngine(
        [WorkArea("zone", ((0, 0), (1, 0), (1, 1), (0, 1)), 1, None)],
        [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]], EngineConfig(**config),
    )


def test_missing_inside_is_retained_indefinitely_and_new_id_reuses_slot():
    engine = team_engine()
    engine.process([observation(1, 0, 50, 50)], 0)
    missing = engine.process([], 7200)["areas"][0]
    assert missing["actualWorkingCount"] == 1
    assert missing["qualifiedTeamWorkDurationSec"] == 7200
    returned = engine.process([observation(99, 7201, 50, 50)], 7201)
    assert returned["areas"][0]["actualWorkingCount"] == 1
    assert returned["persons"][0]["workState"] == "WORKING"


def test_exit_grace_expires_exactly_and_splits_duration_and_intervals():
    engine = team_engine()
    engine.process([observation(1, 0, 50, 50)], 0)
    assert engine.process([observation(1, 10, 150, 150)], 10)["areas"][0]["actualWorkingCount"] == 1
    assert engine.process([], 1209)["areas"][0]["actualWorkingCount"] == 1
    expired = engine.process([], 1211)["areas"][0]
    assert expired["actualWorkingCount"] == 0
    assert expired["qualifiedTeamWorkDurationSec"] == 1210
    assert [(i.start_sec, i.end_sec, i.headcount_compliant)
            for i in engine.compliance_intervals("zone")] == [
        (0, 1210, True), (1210, None, False),
    ]


@pytest.mark.parametrize("return_id", [1, 99])
def test_return_or_replacement_during_grace_does_not_double_count(return_id):
    engine = team_engine()
    engine.process([observation(1, 0, 50, 50)], 0)
    engine.process([observation(1, 10, 150, 150)], 10)
    returned = engine.process([observation(return_id, 100, 50, 50)], 100)
    assert returned["areas"][0]["actualWorkingCount"] == 1
    assert engine.process([], 1300)["areas"][0]["actualWorkingCount"] == 1


def test_return_at_deadline_requires_entry_confirmation_again():
    engine = team_engine(entry_detection_duration_sec=2)
    engine.process([observation(1, 0, 50, 50)], 0)
    engine.process([observation(1, 2, 50, 50)], 2)
    engine.process([observation(1, 10, 150, 150)], 10)
    assert engine.process([observation(1, 1210, 50, 50)], 1210)["areas"][0]["actualWorkingCount"] == 0
    assert engine.process([observation(1, 1212, 50, 50)], 1212)["areas"][0]["actualWorkingCount"] == 1


def test_visible_people_and_observed_new_entries_are_not_merged():
    engine = team_engine()
    engine.process([observation(1, 0, 50, 50)], 0)
    assert engine.process([observation(1, 1, 50, 50), observation(2, 1, 60, 50)], 1)["areas"][0]["actualWorkingCount"] == 2
    # Both old members are now occluded; a third is visibly crossing into the area.
    engine.process([observation(3, 2, 150, 150)], 2)
    assert engine.process([observation(3, 3, 50, 50)], 3)["areas"][0]["actualWorkingCount"] == 3


@pytest.mark.parametrize("reverse", [False, True])
def test_same_frame_departure_and_replacement_are_order_independent(reverse):
    engine = team_engine()
    engine.process([observation(1, 0, 50, 50)], 0)
    people = [observation(1, 1, 150, 150), observation(2, 1, 50, 50)]
    result = engine.process(people[::-1] if reverse else people, 1)
    assert result["areas"][0]["actualWorkingCount"] == 1


def test_multiple_departures_expire_independently_and_replacement_fills_oldest():
    engine = team_engine()
    engine.process([observation(1, 0, 50, 50), observation(2, 0, 60, 50)], 0)
    engine.process([observation(1, 10, 150, 150), observation(2, 10, 60, 50)], 10)
    engine.process([observation(2, 20, 150, 150)], 20)
    assert engine.process([observation(3, 100, 50, 50)], 100)["areas"][0]["actualWorkingCount"] == 2
    assert engine.process([], 1210)["areas"][0]["actualWorkingCount"] == 2
    assert engine.process([], 1220)["areas"][0]["actualWorkingCount"] == 1


def test_disappearing_after_observed_exit_is_not_retained_forever():
    engine = team_engine(exit_detection_duration_sec=5)
    engine.process([observation(1, 0, 50, 50)], 0)
    engine.process([observation(1, 10, 150, 150)], 10)
    assert engine.process([], 1214)["areas"][0]["actualWorkingCount"] == 1
    result = engine.process([], 1216)["areas"][0]
    assert result["actualWorkingCount"] == 0
    assert result["qualifiedTeamWorkDurationSec"] == 1215


def test_frame_gap_still_caps_credited_time():
    engine = team_engine(max_frame_gap_sec=2)
    engine.process([observation(1, 0, 50, 50)], 0)
    result = engine.process([], 7200)["areas"][0]
    assert result["actualWorkingCount"] == 1
    assert result["qualifiedTeamWorkDurationSec"] == 2


def test_known_departed_member_does_not_absorb_another_missing_member_after_expiry():
    engine = team_engine()
    engine.process([observation(1, 0, 50, 50), observation(2, 0, 60, 50)], 0)
    engine.process([observation(1, 10, 150, 150)], 10)
    # Keep observing the outside track while the other member remains occluded.
    engine.process([observation(1, 1210, 150, 150)], 1210)
    returned = engine.process([observation(1, 1211, 50, 50)], 1211)
    assert returned["areas"][0]["actualWorkingCount"] == 2


def test_replacement_cannot_consume_grace_from_a_different_area():
    engine = WorkEfficiencyEngine(
        [WorkArea("a", ((0, 0), (0.4, 0), (0.4, 1), (0, 1)), 1),
         WorkArea("b", ((0.6, 0), (1, 0), (1, 1), (0.6, 1)), 1)],
        [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]],
        EngineConfig(entry_detection_duration_sec=0, exit_detection_duration_sec=0),
    )
    engine.process([observation(1, 0, 20, 50)], 0)
    engine.process([observation(1, 1, 150, 150)], 1)
    result = engine.process([observation(2, 2, 80, 50)], 2)
    assert [area["actualWorkingCount"] for area in result["areas"]] == [1, 1]
    expired = engine.process([], 1201)
    assert [area["actualWorkingCount"] for area in expired["areas"]] == [0, 1]
