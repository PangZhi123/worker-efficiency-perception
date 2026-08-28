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
        EngineConfig(entry_detection_duration_sec=0, exit_detection_duration_sec=0),
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
