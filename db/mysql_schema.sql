-- Initial MySQL 8 schema for work-efficiency perception.
-- Runtime code still uses the in-memory repository until database settings are confirmed.

CREATE TABLE we_scene (
    scene_id VARCHAR(64) PRIMARY KEY,
    site_id VARCHAR(64) NOT NULL,
    scene_name VARCHAR(128) NOT NULL,
    image_url VARCHAR(512) NOT NULL,
    width INT UNSIGNED NOT NULL,
    height INT UNSIGNED NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    INDEX idx_we_scene_site (site_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE we_camera (
    camera_id VARCHAR(64) PRIMARY KEY,
    stream_url VARCHAR(2048) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'REGISTERED',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE we_video (
    video_id VARCHAR(64) PRIMARY KEY,
    camera_id VARCHAR(64) NULL,
    storage_path VARCHAR(1024) NOT NULL,
    width INT UNSIGNED NOT NULL,
    height INT UNSIGNED NOT NULL,
    fps DOUBLE NOT NULL,
    frame_count BIGINT UNSIGNED NOT NULL,
    duration_sec DOUBLE NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    CONSTRAINT fk_we_video_camera FOREIGN KEY (camera_id) REFERENCES we_camera(camera_id),
    INDEX idx_we_video_camera (camera_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE we_configuration (
    configuration_id VARCHAR(64) PRIMARY KEY,
    camera_id VARCHAR(64) NOT NULL,
    scene_id VARCHAR(64) NOT NULL,
    calibration_version VARCHAR(64) NOT NULL,
    area_version VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL,
    homography_json JSON NOT NULL,
    algorithm_json JSON NOT NULL,
    mean_error_px DOUBLE NOT NULL,
    max_error_px DOUBLE NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    CONSTRAINT fk_we_configuration_scene FOREIGN KEY (scene_id) REFERENCES we_scene(scene_id),
    CONSTRAINT fk_we_configuration_camera FOREIGN KEY (camera_id) REFERENCES we_camera(camera_id),
    UNIQUE KEY uk_we_configuration_calibration (calibration_version),
    UNIQUE KEY uk_we_configuration_area (area_version),
    INDEX idx_we_configuration_camera (camera_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE we_work_area (
    configuration_id VARCHAR(64) NOT NULL,
    area_id VARCHAR(64) NOT NULL,
    polygon_json JSON NOT NULL,
    expected_worker_count INT UNSIGNED NULL,
    required_team_work_duration_sec DOUBLE NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (configuration_id, area_id),
    CONSTRAINT fk_we_work_area_configuration
        FOREIGN KEY (configuration_id) REFERENCES we_configuration(configuration_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE we_task (
    task_id VARCHAR(64) PRIMARY KEY,
    input_type VARCHAR(20) NOT NULL,
    camera_id VARCHAR(64) NULL,
    video_id VARCHAR(64) NULL,
    configuration_id VARCHAR(64) NOT NULL,
    status VARCHAR(20) NOT NULL,
    processed_frames BIGINT UNSIGNED NOT NULL DEFAULT 0,
    progress DOUBLE NULL,
    current_timestamp_sec DOUBLE NULL,
    failure_code VARCHAR(16) NULL,
    failure_message VARCHAR(1000) NULL,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    CONSTRAINT fk_we_task_configuration
        FOREIGN KEY (configuration_id) REFERENCES we_configuration(configuration_id),
    CONSTRAINT fk_we_task_camera FOREIGN KEY (camera_id) REFERENCES we_camera(camera_id),
    CONSTRAINT fk_we_task_video FOREIGN KEY (video_id) REFERENCES we_video(video_id),
    INDEX idx_we_task_status_updated (status, updated_at),
    INDEX idx_we_task_camera (camera_id),
    INDEX idx_we_task_video (video_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE we_area_snapshot (
    snapshot_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    task_id VARCHAR(64) NOT NULL,
    area_id VARCHAR(64) NOT NULL,
    frame_seq BIGINT UNSIGNED NOT NULL,
    video_timestamp_sec DOUBLE NOT NULL,
    captured_at DATETIME(6) NULL,
    actual_working_count INT UNSIGNED NOT NULL,
    expected_worker_count INT UNSIGNED NULL,
    headcount_compliant BOOLEAN NULL,
    qualified_team_work_duration_sec DOUBLE NOT NULL,
    required_team_work_duration_sec DOUBLE NULL,
    efficiency_status VARCHAR(20) NOT NULL,
    CONSTRAINT fk_we_area_snapshot_task FOREIGN KEY (task_id) REFERENCES we_task(task_id),
    UNIQUE KEY uk_we_area_snapshot_frame (task_id, area_id, frame_seq),
    INDEX idx_we_area_snapshot_time (task_id, area_id, video_timestamp_sec)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE we_compliance_interval (
    interval_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    task_id VARCHAR(64) NOT NULL,
    area_id VARCHAR(64) NOT NULL,
    start_sec DOUBLE NOT NULL,
    end_sec DOUBLE NULL,
    headcount_compliant BOOLEAN NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    CONSTRAINT fk_we_compliance_interval_task FOREIGN KEY (task_id) REFERENCES we_task(task_id),
    INDEX idx_we_compliance_interval_range (task_id, area_id, start_sec)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE we_pose_snapshot (
    task_id VARCHAR(64) NOT NULL,
    frame_seq BIGINT UNSIGNED NOT NULL,
    track_id VARCHAR(64) NOT NULL,
    video_timestamp_sec DOUBLE NOT NULL,
    captured_at DATETIME(6) NULL,
    confidence DOUBLE NOT NULL,
    bbox_json JSON NOT NULL,
    keypoints_json JSON NOT NULL,
    scene_point_json JSON NULL,
    area_id VARCHAR(64) NULL,
    person_zone_state VARCHAR(16) NOT NULL,
    work_state VARCHAR(16) NOT NULL,
    PRIMARY KEY (task_id, frame_seq, track_id),
    CONSTRAINT fk_we_pose_snapshot_task FOREIGN KEY (task_id) REFERENCES we_task(task_id),
    INDEX idx_we_pose_snapshot_time (task_id, video_timestamp_sec)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE we_trajectory_point (
    task_id VARCHAR(64) NOT NULL,
    track_id VARCHAR(64) NOT NULL,
    trajectory_seq BIGINT UNSIGNED NOT NULL,
    video_timestamp_sec DOUBLE NOT NULL,
    scene_x DOUBLE NOT NULL,
    scene_y DOUBLE NOT NULL,
    area_id VARCHAR(64) NULL,
    PRIMARY KEY (task_id, track_id, trajectory_seq),
    CONSTRAINT fk_we_trajectory_task FOREIGN KEY (task_id) REFERENCES we_task(task_id),
    INDEX idx_we_trajectory_time (task_id, video_timestamp_sec)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
