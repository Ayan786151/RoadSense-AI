"""
================================================================================
ROAD SENSE AI - LIVE TRAFFIC OBSERVATION BUILDER
MODULE: LIVE DATA STANDARDIZATION & OBSERVATION LAYER (Step 5 -> Intermediate)
================================================================================

This module acts as the bridge between the computer vision pipeline and the
temporal machine learning risk prediction system.

It ingests camera-derived features (vehicle detection, trajectory tracking,
speed distribution, and movement congestion scores) and transforms them into
a standardized live traffic observation dataset.

ARCHITECTURAL PRINCIPLES:
1. SEMANTIC SEPARATION:
   - Camera observations provide local, camera-frame-level measurements
     (vehicle counts, pixel speeds, movement congestion).
   - They DO NOT directly measure geographic population density, road capacity,
     or city-wide temporal dynamics.
   - A normalized vehicle count is explicitly designated as `vehicle_density_proxy`
     rather than true geographic vehicle density.
2. COMBINED CAMERA SIGNAL:
   - Fuses vision-based congestion and trajectory movement congestion into a
     unified `camera_congestion` indicator.
3. PRESERVATION OF VISION GROUNDING:
   - All underlying raw computer vision metrics (timestamps, vehicle counts,
     speed distributions, state classifications) are preserved for auditing
     and downstream aggregation.
4. SESSION ISOLATION & BACKWARD COMPATIBILITY:
   - Supports processing within isolated session directories (data/sessions/{session_id}/)
   - Defaults to root data/ paths when no session is specified.
================================================================================
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION & FILE PATHS
# ==============================================================================

DEFAULT_INPUT_CSV = "data/vision_congestion_features.csv"
DEFAULT_OUTPUT_CSV = "data/live_traffic_observations.csv"

# Required columns from vision feature fusion
REQUIRED_INPUT_COLUMNS = [
    "timestamp_seconds",
    "vehicle_count",
    "vehicle_density_score",
    "tracked_vehicle_count",
    "moving_vehicle_count",
    "slow_vehicle_count",
    "stopped_vehicle_count",
    "moving_vehicle_percentage",
    "slow_vehicle_percentage",
    "stopped_vehicle_percentage",
    "average_pixel_speed",
    "median_pixel_speed",
    "movement_congestion_score",
    "vision_congestion_score",
    "traffic_state"
]


# ==============================================================================
# RESOLVE SESSION PATHS
# ==============================================================================

def resolve_paths(session_id: str = None):
    """
    Resolves input and output CSV paths based on session_id.
    """
    if session_id:
        session_dir = Path("data") / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        input_csv = str(session_dir / "vision_congestion_features.csv")
        output_csv = str(session_dir / "live_traffic_observations.csv")
        return input_csv, output_csv, str(session_dir)
    return DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_CSV, "data"


# ==============================================================================
# 1. DATA INGESTION & VALIDATION
# ==============================================================================

def load_vision_features(filepath: str = DEFAULT_INPUT_CSV) -> pd.DataFrame:
    """
    Loads camera-derived features from the feature fusion stage and validates
    that all required columns are present.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Required vision input file not found: {filepath}\n"
            "Please ensure the computer vision pipeline (feature_fusion.py) has run."
        )

    df = pd.read_csv(filepath)

    # Normalize column names for robustness
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Validate presence of required columns
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input file '{filepath}' is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )

    return df


# ==============================================================================
# 2. CONGESTION CLASSIFICATION LOGIC
# ==============================================================================

def classify_camera_traffic_state(congestion_score: float) -> str:
    """
    Classifies traffic state based on camera congestion score using standard
    project thresholds (matching vision feature fusion):
    - SEVERE   : >= 75.0
    - HIGH     : >= 55.0
    - MODERATE : >= 30.0
    - LOW      : < 30.0
    """
    if congestion_score >= 75.0:
        return "SEVERE"
    elif congestion_score >= 55.0:
        return "HIGH"
    elif congestion_score >= 30.0:
        return "MODERATE"
    else:
        return "LOW"


# ==============================================================================
# 3. LIVE TRAFFIC OBSERVATION BUILDER
# ==============================================================================

def build_traffic_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw camera features into a standardized live traffic observation dataset.
    """
    obs = df.copy()

    # 1. Enforce numeric data types for numerical metrics
    numeric_cols = [
        "timestamp_seconds",
        "vehicle_count",
        "vehicle_density_score",
        "tracked_vehicle_count",
        "moving_vehicle_count",
        "slow_vehicle_count",
        "stopped_vehicle_count",
        "moving_vehicle_percentage",
        "slow_vehicle_percentage",
        "stopped_vehicle_percentage",
        "average_pixel_speed",
        "median_pixel_speed",
        "movement_congestion_score",
        "vision_congestion_score"
    ]

    for col in numeric_cols:
        obs[col] = pd.to_numeric(obs[col], errors="coerce")

    # 2. Combined camera congestion signal
    obs["camera_congestion"] = (
        0.5 * obs["vision_congestion_score"]
        + 0.5 * obs["movement_congestion_score"]
    ).clip(0.0, 100.0).round(2)

    # 3. Vehicle density proxy
    max_vehicle_count = obs["vehicle_count"].max()
    if max_vehicle_count > 0:
        obs["vehicle_density_proxy"] = (
            (obs["vehicle_count"] / max_vehicle_count) * 100.0
        ).clip(0.0, 100.0).round(2)
    else:
        obs["vehicle_density_proxy"] = 0.0

    # 4. Standardized camera traffic state classification
    obs["camera_traffic_state"] = obs["camera_congestion"].apply(classify_camera_traffic_state)

    # 5. Arrange columns in clean, logical order
    ordered_columns = [
        "timestamp_seconds",
        "vehicle_count",
        "tracked_vehicle_count",
        "moving_vehicle_count",
        "slow_vehicle_count",
        "stopped_vehicle_count",
        "moving_vehicle_percentage",
        "slow_vehicle_percentage",
        "stopped_vehicle_percentage",
        "average_speed_kmh",
        "median_speed_kmh",
        "average_pixel_speed",
        "median_pixel_speed",
        "vehicle_density_score",
        "vehicle_density_proxy",
        "movement_congestion_score",
        "vision_congestion_score",
        "camera_congestion",
        "traffic_state",
        "camera_traffic_state",
        "calibration_status"
    ]

    avail_ordered = [c for c in ordered_columns if c in obs.columns]
    remaining_cols = [c for c in obs.columns if c not in avail_ordered]
    obs = obs[avail_ordered + remaining_cols]
    return obs


# ==============================================================================
# 4. VALIDATION & AUDIT REPORTING
# ==============================================================================

def validate_observations(df: pd.DataFrame) -> dict:
    """
    Validates output observations dataset and returns an audit dictionary.
    """
    core_cols = [
        "timestamp_seconds",
        "vehicle_count",
        "tracked_vehicle_count",
        "moving_vehicle_count",
        "slow_vehicle_count",
        "stopped_vehicle_count",
        "average_pixel_speed",
        "median_pixel_speed",
        "vehicle_density_proxy",
        "movement_congestion_score",
        "vision_congestion_score",
        "camera_congestion",
        "traffic_state",
        "camera_traffic_state"
    ]
    core_nulls = int(df[core_cols].isnull().sum().sum())
    total_nulls = int(df.isnull().sum().sum())

    audit = {
        "num_records": len(df),
        "core_nulls": core_nulls,
        "total_nulls": total_nulls,
        "is_valid": (core_nulls == 0) and len(df) > 0,
        "vehicle_count_min": float(df["vehicle_count"].min()),
        "vehicle_count_max": float(df["vehicle_count"].max()),
        "density_proxy_min": float(df["vehicle_density_proxy"].min()),
        "density_proxy_max": float(df["vehicle_density_proxy"].max()),
        "vision_congestion_min": float(df["vision_congestion_score"].min()),
        "vision_congestion_max": float(df["vision_congestion_score"].max()),
        "movement_congestion_min": float(df["movement_congestion_score"].min()),
        "movement_congestion_max": float(df["movement_congestion_score"].max()),
        "camera_congestion_min": float(df["camera_congestion"].min()),
        "camera_congestion_max": float(df["camera_congestion"].max()),
        "camera_traffic_state_dist": df["camera_traffic_state"].value_counts().to_dict(),
        "vision_traffic_state_dist": df["traffic_state"].value_counts().to_dict()
    }

    return audit


def print_audit_report(audit: dict, df: pd.DataFrame, output_path: str):
    """
    Prints a formatted, detailed inspection report to stdout.
    """
    print("\n" + "=" * 70)
    print("ROAD SENSE AI - LIVE TRAFFIC OBSERVATION AUDIT REPORT")
    print("=" * 70)

    print(f"\n[1] DATASET SUMMARY:")
    print(f"  - Total Input / Output Records : {audit['num_records']}")
    print(f"  - Total Null Values            : {audit['total_nulls']}")
    print(f"  - Validation Status            : {'PASSED (Zero Nulls)' if audit['is_valid'] else 'FAILED'}")

    print(f"\n[2] VEHICLE COUNT & DENSITY PROXY:")
    print(f"  - Vehicle Count Range          : {audit['vehicle_count_min']:.0f} to {audit['vehicle_count_max']:.0f} vehicles")
    print(f"  - Vehicle Density Proxy Range  : {audit['density_proxy_min']:.2f}% to {audit['density_proxy_max']:.2f}%")

    print(f"\n[3] CONGESTION SIGNALS:")
    print(f"  - Vision Congestion Range      : {audit['vision_congestion_min']:.2f} to {audit['vision_congestion_max']:.2f}")
    print(f"  - Movement Congestion Range    : {audit['movement_congestion_min']:.2f} to {audit['movement_congestion_max']:.2f}")
    print(f"  - Combined Camera Congestion   : {audit['camera_congestion_min']:.2f} to {audit['camera_congestion_max']:.2f}")

    print(f"\n[4] TRAFFIC STATE DISTRIBUTIONS:")
    print("  Camera Traffic State (Combined):")
    for state, count in audit["camera_traffic_state_dist"].items():
        pct = (count / audit['num_records']) * 100
        print(f"    * {state:<10} : {count:>3} records ({pct:>5.1f}%)")

    print("\n  Original Vision Traffic State:")
    for state, count in audit["vision_traffic_state_dist"].items():
        pct = (count / audit['num_records']) * 100
        print(f"    * {state:<10} : {count:>3} records ({pct:>5.1f}%)")

    print("\n" + "=" * 70)
    print("GENERATED OBSERVATIONS PREVIEW (First 10 Records)")
    print("=" * 70)
    preview_cols = [
        "timestamp_seconds",
        "vehicle_count",
        "vehicle_density_proxy",
        "moving_vehicle_percentage",
        "stopped_vehicle_percentage",
        "median_pixel_speed",
        "movement_congestion_score",
        "vision_congestion_score",
        "camera_congestion",
        "camera_traffic_state"
    ]
    print(df[preview_cols].head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"Output saved to: {output_path}")
    print("=" * 70 + "\n")


# ==============================================================================
# 5. PERSISTENCE & PIPELINE EXECUTION
# ==============================================================================

def save_observations(df: pd.DataFrame, output_path: str = DEFAULT_OUTPUT_CSV) -> str:
    """
    Saves the standardized observation dataset to CSV.
    """
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return str(target_path)


def run_observation_builder(session_id: str = None):
    input_csv, output_csv, session_dir = resolve_paths(session_id)

    print("=" * 70)
    print("ROAD SENSE AI - LIVE TRAFFIC OBSERVATION BUILDER")
    print("=" * 70)
    if session_id:
        print(f"Active Session : {session_id}")

    # 1. Ingest
    print(f"\n[+] Loading vision features from: {input_csv}")
    vision_df = load_vision_features(input_csv)
    print(f"    Loaded {len(vision_df)} records with {len(vision_df.columns)} columns.")

    # 2. Build standardized observations
    print("\n[+] Building standardized live traffic observations...")
    observations_df = build_traffic_observations(vision_df)

    # 3. Validate & Audit
    audit = validate_observations(observations_df)

    # 4. Save
    output_path = save_observations(observations_df, output_csv)

    # 5. Report
    print_audit_report(audit, observations_df, output_path)

    return observations_df


def main():
    parser = argparse.ArgumentParser(description="RoadSense AI - Live Traffic Observation Builder")
    parser.add_argument("--session", type=str, default=None, help="Session identifier (e.g. session_002)")

    args = parser.parse_args()

    run_observation_builder(session_id=args.session)


if __name__ == "__main__":
    main()
