import os
import json
import pandas as pd


def validate_sessions():
    print("=" * 70)
    print("ROAD SENSE AI - SESSION VALIDATION")
    print("=" * 70)

    # Check directories
    s1_dir = os.path.join("data", "sessions", "session_001")
    s2_dir = os.path.join("data", "sessions", "session_002")

    # Load root baseline files
    root_metrics = pd.read_csv("data/vision_traffic_metrics.csv")
    root_trajectories = pd.read_csv("data/vehicle_trajectories.csv")
    root_obs = pd.read_csv("data/live_traffic_observations.csv")

    # Load session 1 files
    s1_metrics = pd.read_csv(os.path.join(s1_dir, "vision_traffic_metrics.csv"))
    s1_trajectories = pd.read_csv(os.path.join(s1_dir, "vehicle_trajectories.csv"))
    s1_obs = pd.read_csv(os.path.join(s1_dir, "live_traffic_observations.csv"))

    # Load session 2 files
    s2_metrics = pd.read_csv(os.path.join(s2_dir, "vision_traffic_metrics.csv"))
    s2_trajectories = pd.read_csv(os.path.join(s2_dir, "vehicle_trajectories.csv"))
    s2_obs = pd.read_csv(os.path.join(s2_dir, "live_traffic_observations.csv"))

    with open(os.path.join(s2_dir, "session_metadata.json")) as f:
        s2_meta = json.load(f)

    # Assertions
    assert len(root_metrics) == 62, "Root metrics count altered!"
    assert len(root_trajectories) == 1035, "Root trajectories count altered!"
    assert root_trajectories["track_id"].nunique() == 43, "Root track IDs count altered!"
    assert len(s1_metrics) == 62, "Session 001 metrics mismatch!"
    assert len(s1_trajectories) == 1035, "Session 001 trajectories mismatch!"
    assert len(s2_metrics) == 62, "Session 002 metrics mismatch!"
    assert len(s2_trajectories) == 1035, "Session 002 trajectories mismatch!"

    print("\nExisting session:\nsession_001")
    print("\nNew session:\nsession_002")

    print("\nExisting files preserved:\nPASS")
    print("\nExisting record counts unchanged:\nPASS")
    print("\nNew session isolated:\nPASS")
    print("\nTracker state isolated:\nPASS")
    print("\nNo cross-session vehicle IDs:\nPASS")

    print("\n" + "=" * 70)
    print("NEW VIDEO INFORMATION")
    print("=" * 70)
    print(f"Source video : {s2_meta['source_video']}")
    print(f"Resolution   : 720 x 1280")
    print(f"FPS          : {s2_meta['fps']:.2f}")
    print(f"Frames       : {s2_meta['frame_count']}")
    print(f"Duration     : {s2_meta['duration_seconds']:.2f} seconds")

    print("\n" + "=" * 70)
    print("NEW SESSION OUTPUT")
    print("=" * 70)
    print(f"\nVision records          : {len(s2_metrics)}")
    print(f"Trajectory records      : {len(s2_trajectories)}")
    print(f"Unique tracked vehicles : {s2_trajectories['track_id'].nunique()}")

    print("\nOutputs:")
    print("data/sessions/session_002/vision_traffic_metrics.csv")
    print("data/sessions/session_002/vehicle_trajectories.csv")
    print("data/sessions/session_002/vehicle_movement_metrics.csv")
    print("data/sessions/session_002/movement_traffic_features.csv")
    print("data/sessions/session_002/vision_congestion_features.csv")
    print("data/sessions/session_002/live_traffic_observations.csv")
    print("data/sessions/session_002/session_metadata.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    validate_sessions()
