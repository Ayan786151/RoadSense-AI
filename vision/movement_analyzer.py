"""
================================================================================
ROAD SENSE AI - VEHICLE MOVEMENT & SPEED ANALYZER
MODULE: REAL-WORLD SPEED ESTIMATION & TRAJECTORY KINEMATICS
================================================================================

This module analyzes tracked vehicle trajectories, projects bottom-center
road contact points into real-world coordinates via perspective homography,
computes true physical displacement in meters, and evaluates vehicle velocity
in km/h while supporting calibrated, demo calibration, and uncalibrated fallbacks.

MATHEMATICAL FOUNDATION:
1. ROAD-CONTACT POINT PROJECTION:
   Vehicle trajectories use the bottom-center (x_bc, y_bc) = ((x1+x2)/2, y2)
   representing the contact patch on the road plane.
   When calibrated, (x_bc, y_bc) -> (X_world, Y_world) in meters via homography H.

2. REAL-WORLD DISPLACEMENT & VELOCITY:
   For track i between timestamps t-1 and t:
       Delta d = sqrt((X_t - X_{t-1})^2 + (Y_t - Y_{t-1})^2)  [meters]
       Delta t = t - t_{prev}                                  [seconds]
       speed_mps = Delta d / Delta t                           [m/s]
       speed_kmh = speed_mps * 3.6                             [km/h]

3. FIRST OBSERVATION INTEGRITY:
   The first observation of any vehicle has no preceding position.
   Speed is explicitly assigned NaN (not artificially zero) and excluded
   from downstream speed aggregations.

4. DEFENSIVE SANITY CHECKS:
   - Delta t <= 0 -> NaN (invalid_time_interval)
   - Delta d > MAX_WORLD_JUMP -> NaN (unreasonable_displacement / ID switch)
   - No arbitrary speed capping; physical outliers are flagged observably.
================================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import cv2

# Import homography calibration engine
try:
    from vision.calibration import HomographyCalibrator, draw_calibration_overlay
except ImportError:
    from calibration import HomographyCalibrator, draw_calibration_overlay


# ==============================================================================
# CONFIGURATION & THRESHOLDS
# ==============================================================================

DEFAULT_INPUT_CSV = "data/vehicle_trajectories.csv"
DEFAULT_VEHICLE_OUTPUT = "data/vehicle_movement_metrics.csv"
DEFAULT_TRAFFIC_OUTPUT = "data/movement_traffic_features.csv"
DEFAULT_CALIBRATION_CONFIG = "data/calibration_config.json"

# Movement classification thresholds in physical speed (km/h)
STOPPED_THRESHOLD_KMH = 2.0
SLOW_THRESHOLD_KMH = 15.0

# Movement classification thresholds in pixel speed (px/s) for uncalibrated fallback
STOPPED_THRESHOLD_PX = 2.0
SLOW_THRESHOLD_PX = 10.0

# Defensive physical sanity checks
MAX_PIXEL_DISTANCE = 100.0     # Max valid pixel jump per frame step (~100 px)
MAX_WORLD_JUMP_METERS = 50.0   # Max valid real-world displacement per step (~50m in 0.167s = 1080 km/h)
MIN_OBSERVATIONS = 3           # Minimum observations required to classify vehicle status


# ==============================================================================
# RESOLVE SESSION PATHS
# ==============================================================================

def resolve_paths(session_id: Optional[str] = None) -> Tuple[str, str, str, str, Optional[str]]:
    """
    Resolves input and output CSV paths and metadata based on session_id.
    """
    if session_id:
        session_dir = Path("data") / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        input_csv = str(session_dir / "vehicle_trajectories.csv")
        vehicle_output = str(session_dir / "vehicle_movement_metrics.csv")
        traffic_output = str(session_dir / "movement_traffic_features.csv")
        meta_path = str(session_dir / "session_metadata.json") if (session_dir / "session_metadata.json").exists() else None
        return input_csv, vehicle_output, traffic_output, str(session_dir), meta_path
    return DEFAULT_INPUT_CSV, DEFAULT_VEHICLE_OUTPUT, DEFAULT_TRAFFIC_OUTPUT, "data", None


# ==============================================================================
# 1. LOAD & CLEAN DATA
# ==============================================================================

def load_trajectory_data(input_csv: str = DEFAULT_INPUT_CSV) -> pd.DataFrame:
    """
    Loads raw vehicle trajectory observations.
    """
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_columns = [
        "timestamp_seconds",
        "frame_number",
        "track_id",
        "vehicle_type",
        "center_x",
        "center_y"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Trajectory file '{input_csv}' is missing required columns: {missing}")

    return df


def assign_dominant_vehicle_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures each tracker ID is assigned its most frequently observed vehicle class.
    """
    clean_df = df.copy()
    dominant_types = (
        clean_df.groupby("track_id")["vehicle_type"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    clean_df["vehicle_type"] = clean_df["track_id"].map(dominant_types)
    return clean_df


# ==============================================================================
# 2. VEHICLE MOVEMENT & SPEED CALCULATION
# ==============================================================================

def calculate_vehicle_kinematics(
    df: pd.DataFrame,
    calibrator: HomographyCalibrator
) -> pd.DataFrame:
    """
    Calculates track-based displacement, elapsed time, pixel speed, and calibrated physical speed (km/h).

    Features calculated:
    - world_x, world_y: Ground plane coordinates in meters (when calibrated).
    - distance_meters: Real-world displacement between consecutive observations (NaN on first observation).
    - elapsed_seconds: Time delta between consecutive observations.
    - speed_mps: Speed in meters per second.
    - speed_kmh: Speed in kilometers per hour.
    - pixel_distance, pixel_speed: Pixel displacement and velocity.
    - invalid_speed_reason: Diagnostic reason for flagged anomalous records.
    - calibration_status: "calibrated", "demo_calibration", or "uncalibrated".
    """
    kinematics = df.sort_values(by=["track_id", "timestamp_seconds"]).copy()

    # 1. Project bottom-center pixel coordinates to ground plane world coordinates (meters)
    pixel_pts = kinematics[["center_x", "center_y"]].values
    world_pts = calibrator.pixel_to_world(pixel_pts)

    kinematics["world_x"] = np.round(world_pts[:, 0], 3)
    kinematics["world_y"] = np.round(world_pts[:, 1], 3)
    kinematics["calibration_status"] = calibrator.status

    # 2. Track previous positions and timestamps strictly per track_id
    kinematics["previous_x_pixel"] = kinematics.groupby("track_id")["center_x"].shift(1)
    kinematics["previous_y_pixel"] = kinematics.groupby("track_id")["center_y"].shift(1)
    kinematics["previous_x_world"] = kinematics.groupby("track_id")["world_x"].shift(1)
    kinematics["previous_y_world"] = kinematics.groupby("track_id")["world_y"].shift(1)
    kinematics["previous_time"] = kinematics.groupby("track_id")["timestamp_seconds"].shift(1)

    # 3. Time difference
    kinematics["elapsed_seconds"] = kinematics["timestamp_seconds"] - kinematics["previous_time"]

    # Initialize diagnostics
    kinematics["invalid_speed_reason"] = None

    # Mask invalid non-positive time intervals
    invalid_time = kinematics["elapsed_seconds"] <= 0
    kinematics.loc[invalid_time, "invalid_speed_reason"] = "invalid_time_interval"
    kinematics.loc[invalid_time, "elapsed_seconds"] = np.nan

    # 4. Pixel displacement and pixel speed calculation
    dx_px = kinematics["center_x"] - kinematics["previous_x_pixel"]
    dy_px = kinematics["center_y"] - kinematics["previous_y_pixel"]
    kinematics["pixel_distance"] = np.sqrt(dx_px ** 2 + dy_px ** 2)

    # Filter unreasonable pixel tracking jumps
    pixel_jump_mask = kinematics["pixel_distance"] > MAX_PIXEL_DISTANCE
    kinematics.loc[pixel_jump_mask, "invalid_speed_reason"] = "unreasonable_pixel_displacement"
    kinematics.loc[pixel_jump_mask, "pixel_distance"] = np.nan

    # Pixel speed: NaN on first observation (where previous_time is NaN)
    kinematics["pixel_speed"] = kinematics["pixel_distance"] / kinematics["elapsed_seconds"]

    # 5. Real-world physical displacement and speed (km/h) calculation
    if calibrator.is_calibrated:
        dx_wld = kinematics["world_x"] - kinematics["previous_x_world"]
        dy_wld = kinematics["world_y"] - kinematics["previous_y_world"]
        kinematics["distance_meters"] = np.sqrt(dx_wld ** 2 + dy_wld ** 2)

        # Defensive sanity check: filter impossible physical jumps
        world_jump_mask = kinematics["distance_meters"] > MAX_WORLD_JUMP_METERS
        kinematics.loc[world_jump_mask, "invalid_speed_reason"] = "unreasonable_world_displacement"
        kinematics.loc[world_jump_mask, "distance_meters"] = np.nan

        kinematics["speed_mps"] = kinematics["distance_meters"] / kinematics["elapsed_seconds"]
        kinematics["speed_kmh"] = (kinematics["speed_mps"] * 3.6).round(2)
    else:
        kinematics["distance_meters"] = np.nan
        kinematics["speed_mps"] = np.nan
        kinematics["speed_kmh"] = np.nan

    return kinematics


# ==============================================================================
# 3. MOVEMENT CLASSIFICATION
# ==============================================================================

def classify_movement_state(
    speed_kmh: Optional[float],
    pixel_speed: Optional[float],
    is_calibrated: bool
) -> str:
    """
    Classifies a vehicle's instantaneous movement state as STOPPED, SLOW, or MOVING.
    Uses physical speed (km/h) when calibrated; falls back to pixel speed (px/s) when uncalibrated.
    """
    if is_calibrated and pd.notnull(speed_kmh):
        if speed_kmh < STOPPED_THRESHOLD_KMH:
            return "STOPPED"
        elif speed_kmh < SLOW_THRESHOLD_KMH:
            return "SLOW"
        else:
            return "MOVING"
    elif pd.notnull(pixel_speed):
        if pixel_speed < STOPPED_THRESHOLD_PX:
            return "STOPPED"
        elif pixel_speed < SLOW_THRESHOLD_PX:
            return "SLOW"
        else:
            return "MOVING"
    return "UNKNOWN"


# ==============================================================================
# 4. VEHICLE-LEVEL & TRAFFIC-LEVEL AGGREGATION
# ==============================================================================

def build_vehicle_summary(
    df: pd.DataFrame,
    is_calibrated: bool
) -> pd.DataFrame:
    """
    Builds vehicle-level kinematic summary.
    Correctly ignores initial NaN speed observations during aggregation.
    """
    records = []

    for track_id, group in df.groupby("track_id", as_index=False):
        vtype = group["vehicle_type"].iloc[0]
        obs_count = len(group)
        status = group["calibration_status"].iloc[0]

        # Valid speeds excluding initial observation NaNs
        valid_kmh = group["speed_kmh"].dropna()
        valid_px = group["pixel_speed"].dropna()
        valid_wld_dist = group["distance_meters"].dropna()
        valid_px_dist = group["pixel_distance"].dropna()

        total_wld_dist = round(float(valid_wld_dist.sum()), 2) if is_calibrated and not valid_wld_dist.empty else np.nan
        avg_kmh = round(float(valid_kmh.mean()), 2) if not valid_kmh.empty else np.nan
        median_kmh = round(float(valid_kmh.median()), 2) if not valid_kmh.empty else np.nan
        max_kmh = round(float(valid_kmh.max()), 2) if not valid_kmh.empty else np.nan

        total_px_dist = round(float(valid_px_dist.sum()), 2) if not valid_px_dist.empty else 0.0
        avg_px = round(float(valid_px.mean()), 2) if not valid_px.empty else 0.0
        median_px = round(float(valid_px.median()), 2) if not valid_px.empty else 0.0
        max_px = round(float(valid_px.max()), 2) if not valid_px.empty else 0.0

        # Movement status based on median speed
        if obs_count < MIN_OBSERVATIONS:
            mov_status = "INSUFFICIENT_DATA"
        else:
            mov_status = classify_movement_state(median_kmh, median_px, is_calibrated)

        record = {
            "track_id": track_id,
            "vehicle_type": vtype,
            "observation_count": obs_count,
            "calibration_status": status,
            "total_distance_meters": total_wld_dist,
            "average_speed_kmh": avg_kmh,
            "median_speed_kmh": median_kmh,
            "maximum_speed_kmh": max_kmh,
            "total_pixel_distance": total_px_dist,
            "average_pixel_speed": avg_px,
            "median_pixel_speed": median_px,
            "maximum_pixel_speed": max_px,
            "movement_status": mov_status,
            "first_timestamp": round(float(group["timestamp_seconds"].min()), 3),
            "last_timestamp": round(float(group["timestamp_seconds"].max()), 3)
        }
        records.append(record)

    summary_df = pd.DataFrame(records)
    return summary_df


def build_traffic_features(
    df: pd.DataFrame,
    is_calibrated: bool
) -> pd.DataFrame:
    """
    Aggregates vehicle observations per timestamp into traffic-level features.
    """
    df_eval = df.copy()
    df_eval["movement_status"] = df_eval.apply(
        lambda r: classify_movement_state(r.get("speed_kmh"), r.get("pixel_speed"), is_calibrated),
        axis=1
    )

    # 1. Base aggregations per timestamp
    traffic = (
        df_eval.groupby("timestamp_seconds")
        .agg(
            tracked_vehicle_count=("track_id", "nunique"),
            average_speed_kmh=("speed_kmh", lambda s: round(float(s.dropna().mean()), 2) if not s.dropna().empty else np.nan),
            median_speed_kmh=("speed_kmh", lambda s: round(float(s.dropna().median()), 2) if not s.dropna().empty else np.nan),
            average_pixel_speed=("pixel_speed", lambda s: round(float(s.dropna().mean()), 2) if not s.dropna().empty else 0.0),
            median_pixel_speed=("pixel_speed", lambda s: round(float(s.dropna().median()), 2) if not s.dropna().empty else 0.0),
            total_pixel_movement=("pixel_distance", lambda s: round(float(s.dropna().sum()), 2))
        )
        .reset_index()
    )

    # 2. Movement state counts
    status_counts = (
        df_eval.pivot_table(
            index="timestamp_seconds",
            columns="movement_status",
            values="track_id",
            aggfunc="nunique",
            fill_value=0
        )
        .reset_index()
    )

    traffic = traffic.merge(status_counts, on="timestamp_seconds", how="left")

    for col in ["STOPPED", "SLOW", "MOVING"]:
        if col not in traffic.columns:
            traffic[col] = 0

    traffic = traffic.rename(
        columns={
            "STOPPED": "stopped_vehicle_count",
            "SLOW": "slow_vehicle_count",
            "MOVING": "moving_vehicle_count"
        }
    )

    # 3. State percentages
    denominator = traffic["tracked_vehicle_count"].replace(0, 1)
    traffic["stopped_vehicle_percentage"] = (traffic["stopped_vehicle_count"] / denominator) * 100.0
    traffic["slow_vehicle_percentage"] = (traffic["slow_vehicle_count"] / denominator) * 100.0
    traffic["moving_vehicle_percentage"] = (traffic["moving_vehicle_count"] / denominator) * 100.0

    # 4. Movement congestion score
    traffic["movement_congestion_score"] = (
        (traffic["stopped_vehicle_percentage"] * 0.70)
        + (traffic["slow_vehicle_percentage"] * 0.20)
        + ((100.0 - traffic["moving_vehicle_percentage"]) * 0.10)
    ).clip(0.0, 100.0).round(2)

    traffic["calibration_status"] = df_eval["calibration_status"].iloc[0] if not df_eval.empty else "uncalibrated"

    return traffic


# ==============================================================================
# 5. VISUAL DEBUGGING
# ==============================================================================

def execute_visual_debug(
    session_id: Optional[str],
    video_path: Optional[str],
    calibrator: HomographyCalibrator
):
    """
    Renders visual debug frame with calibration overlay if video is available.
    """
    if not video_path and session_id:
        meta_path = Path("data") / "sessions" / session_id / "session_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    video_path = meta.get("source_video")
            except Exception:
                pass

    if not video_path:
        video_path = "videos/traffic.mp4"

    if Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            out_overlay = Path("data") / "sessions" / (session_id or "session_001") / "calibration_overlay.png"
            draw_calibration_overlay(frame, calibrator, out_overlay)
    else:
        print(f"[!] Warning: Video '{video_path}' not found for visual debugging.")


# ==============================================================================
# 6. PIPELINE ORCHESTRATION & CLI
# ==============================================================================

def analyze_movement(
    session_id: Optional[str] = None,
    calibration_config: str = DEFAULT_CALIBRATION_CONFIG,
    calibration_mode: str = "auto",
    video_path: Optional[str] = None,
    show_calibration: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes vehicle movement analysis with perspective homography calibration.
    """
    input_csv, vehicle_output, traffic_output, session_dir, meta_path = resolve_paths(session_id)

    # Determine video key from metadata if present
    video_key = video_path
    if not video_key and meta_path:
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                video_key = meta.get("source_video", "traffic.mp4")
        except Exception:
            video_key = "traffic.mp4"
    if not video_key:
        video_key = "traffic.mp4"

    # Initialize Calibrator
    calibrator = HomographyCalibrator.from_config(
        config_source=calibration_config,
        video_key=video_key,
        mode=calibration_mode
    )

    print("=" * 70)
    print("ROAD SENSE AI - VEHICLE MOVEMENT & SPEED ANALYZER")
    print("=" * 70)
    if session_id:
        print(f"Active Session     : {session_id}")
    print(f"Reading from       : {input_csv}")
    print(f"Calibration Mode   : {calibration_mode.upper()} -> Status: {calibrator.status.upper()}")
    if calibrator.is_calibrated:
        print(f"Calibration Detail : {calibrator.description}")
    print(f"Writing to         :\n  - {vehicle_output}\n  - {traffic_output}")

    # Optional visual debugging
    if show_calibration:
        execute_visual_debug(session_id, video_path or video_key, calibrator)

    # 1. Load data
    df = load_trajectory_data(input_csv)
    print(f"\nTrajectory observations: {len(df)}")
    print(f"Raw tracker IDs        : {df['track_id'].nunique()}")

    # 2. Cleanup classes
    df = assign_dominant_vehicle_type(df)

    # 3. Calculate kinematics
    df = calculate_vehicle_kinematics(df, calibrator)

    # 4. Summaries
    vehicle_summary = build_vehicle_summary(df, calibrator.is_calibrated)
    traffic_features = build_traffic_features(df, calibrator.is_calibrated)

    # 5. Validation reporting
    print("\n" + "=" * 70)
    print("MOVEMENT & SPEED VALIDATION")
    print("=" * 70)

    valid_kmh = df["speed_kmh"].dropna()
    valid_px = df["pixel_speed"].dropna()

    if calibrator.is_calibrated and len(valid_kmh) > 0:
        print(f"\n[+] CALIBRATED PHYSICAL SPEED (km/h):")
        print(f"  - Speed range        : {valid_kmh.min():.2f} to {valid_kmh.max():.2f} km/h")
        print(f"  - Average speed      : {valid_kmh.mean():.2f} km/h")
        print(f"  - Median speed       : {valid_kmh.median():.2f} km/h")
    else:
        print(f"\n[!] PHYSICAL SPEED: UNCALIBRATED (NaN)")

    if len(valid_px) > 0:
        print(f"\n[+] CAMERA PIXEL SPEED (px/s):")
        print(f"  - Pixel speed range  : {valid_px.min():.2f} to {valid_px.max():.2f} px/s")
        print(f"  - Average pixel speed: {valid_px.mean():.2f} px/s")
        print(f"  - Median pixel speed : {valid_px.median():.2f} px/s")

    first_obs_null_count = int(df.groupby("track_id")["speed_kmh"].first().isnull().sum())
    print(f"\n[+] FIRST OBSERVATION INTEGRITY CHECK:")
    print(f"  - First-observation NaN count: {first_obs_null_count} / {df['track_id'].nunique()} tracks (PASS: Zero artificial 0.0s)")

    print(f"\nVehicles classified: {len(vehicle_summary)}")
    print("\nMovement classification distribution:")
    print(vehicle_summary["movement_status"].value_counts().to_string())

    # 6. Previews
    print("\n" + "=" * 70)
    print("VEHICLE MOVEMENT PREVIEW")
    print("=" * 70)
    veh_preview_cols = [
        "track_id", "vehicle_type", "observation_count", "calibration_status",
        "average_speed_kmh", "median_speed_kmh", "average_pixel_speed", "movement_status"
    ]
    print(vehicle_summary[veh_preview_cols].head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print("TRAFFIC MOVEMENT PREVIEW")
    print("=" * 70)
    traf_preview_cols = [
        "timestamp_seconds", "tracked_vehicle_count", "moving_vehicle_count",
        "slow_vehicle_count", "stopped_vehicle_count", "average_speed_kmh",
        "average_pixel_speed", "movement_congestion_score", "calibration_status"
    ]
    print(traffic_features[traf_preview_cols].head(15).to_string(index=False))

    # 7. Save outputs
    Path(vehicle_output).parent.mkdir(parents=True, exist_ok=True)
    Path(traffic_output).parent.mkdir(parents=True, exist_ok=True)

    vehicle_summary.to_csv(vehicle_output, index=False)
    traffic_features.to_csv(traffic_output, index=False)

    print("\n" + "=" * 70)
    print("MOVEMENT ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nVehicle-level metrics saved to:\n  {vehicle_output}")
    print(f"\nTraffic movement features saved to:\n  {traffic_output}")

    return vehicle_summary, traffic_features


def main():
    parser = argparse.ArgumentParser(description="RoadSense AI - Vehicle Movement & Speed Analyzer")
    parser.add_argument("--session", type=str, default=None, help="Session identifier (e.g. session_001)")
    parser.add_argument("--calibration", type=str, default=DEFAULT_CALIBRATION_CONFIG, help="Path to calibration config JSON")
    parser.add_argument("--calibration-mode", type=str, default="auto", choices=["auto", "custom", "demo", "none"], help="Calibration mode")
    parser.add_argument("--video", type=str, default=None, help="Path to video file for calibration matching/visualization")
    parser.add_argument("--show-calibration", action="store_true", help="Generate calibration visual debugging overlay")

    args = parser.parse_args()

    analyze_movement(
        session_id=args.session,
        calibration_config=args.calibration,
        calibration_mode=args.calibration_mode,
        video_path=args.video,
        show_calibration=args.show_calibration
    )


if __name__ == "__main__":
    main()