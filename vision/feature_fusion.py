"""
================================================================================
ROAD SENSE AI - VISION FEATURE FUSION
MODULE: MULTI-MODAL VISION & MOVEMENT FEATURE FUSION
================================================================================

This module performs relational feature fusion combining vehicle detection
density signals from vehicle_detector.py and movement kinematics / speed features
from movement_analyzer.py.

ARCHITECTURAL PRINCIPLES:
1. RELATIONAL MERGING:
   Merges vision and movement streams on the exact relational timestamp key
   `timestamp_seconds` (using inner join), rather than fragile positional slicing.
2. SPEED FEATURE PROPAGATION:
   Passes calibrated speed features (`average_speed_kmh`, `median_speed_kmh`,
   `calibration_status`) alongside pixel speed metrics downstream to live
   observation builders.
3. CONGESTION SIGNAL BLEND:
   Fuses vehicle occupancy density (40% weight) and movement congestion (60% weight)
   into `vision_congestion_score` and classifies traffic states.
================================================================================
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np


# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

DEFAULT_VISION_INPUT = "data/vision_traffic_metrics.csv"
DEFAULT_MOVEMENT_INPUT = "data/movement_traffic_features.csv"
DEFAULT_OUTPUT_FILE = "data/vision_congestion_features.csv"

# Weight given to vehicle-density/count signal
DENSITY_WEIGHT = 0.40

# Weight given to movement behavior
MOVEMENT_WEIGHT = 0.60


# ==============================================================================
# RESOLVE SESSION PATHS
# ==============================================================================

def resolve_paths(session_id: Optional[str] = None) -> Tuple[str, str, str, str]:
    """
    Resolves input and output CSV paths based on session_id.
    """
    if session_id:
        session_dir = Path("data") / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        vision_input = str(session_dir / "vision_traffic_metrics.csv")
        movement_input = str(session_dir / "movement_traffic_features.csv")
        output_file = str(session_dir / "vision_congestion_features.csv")
        return vision_input, movement_input, output_file, str(session_dir)
    return DEFAULT_VISION_INPUT, DEFAULT_MOVEMENT_INPUT, DEFAULT_OUTPUT_FILE, "data"


# ==============================================================================
# 1. LOAD & NORMALIZE INPUTS
# ==============================================================================

def load_inputs(
    vision_path: str = DEFAULT_VISION_INPUT,
    movement_path: str = DEFAULT_MOVEMENT_INPUT
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw vision traffic metrics and movement traffic features.
    """
    if not Path(vision_path).exists():
        raise FileNotFoundError(f"Missing vision metrics file: {vision_path}")

    if not Path(movement_path).exists():
        raise FileNotFoundError(f"Missing movement features file: {movement_path}")

    vision = pd.read_csv(vision_path)
    movement = pd.read_csv(movement_path)

    return vision, movement


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names to lowercase snake_case."""
    df_clean = df.copy()
    df_clean.columns = (
        df_clean.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df_clean


def find_vehicle_count_column(df: pd.DataFrame) -> str:
    """Finds the vehicle count column name."""
    possible_columns = [
        "vehicle_count",
        "vehicles_detected",
        "detected_vehicles",
        "vehicle_density",
        "tracked_vehicle_count"
    ]
    for col in possible_columns:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a vehicle-count column in vision metrics. Available: {list(df.columns)}")


# ==============================================================================
# 2. PREPARE STREAMS
# ==============================================================================

def prepare_vision_data(vision: pd.DataFrame) -> pd.DataFrame:
    """Prepares vision metrics and calculates vehicle density score (0-100)."""
    vision_clean = normalize_columns(vision)
    count_col = find_vehicle_count_column(vision_clean)

    vision_clean["vehicle_count"] = pd.to_numeric(vision_clean[count_col], errors="coerce")
    vision_clean["timestamp_seconds"] = pd.to_numeric(vision_clean["timestamp_seconds"], errors="coerce").round(3)

    min_count = vision_clean["vehicle_count"].min()
    max_count = vision_clean["vehicle_count"].max()

    if max_count > min_count:
        vision_clean["vehicle_density_score"] = (
            (vision_clean["vehicle_count"] - min_count) / (max_count - min_count)
        ) * 100.0
    else:
        vision_clean["vehicle_density_score"] = 0.0

    return vision_clean


def prepare_movement_data(movement: pd.DataFrame) -> pd.DataFrame:
    """Prepares and validates movement traffic features."""
    mov_clean = normalize_columns(movement)
    mov_clean["timestamp_seconds"] = pd.to_numeric(mov_clean["timestamp_seconds"], errors="coerce").round(3)

    required = [
        "timestamp_seconds",
        "tracked_vehicle_count",
        "moving_vehicle_percentage",
        "slow_vehicle_percentage",
        "stopped_vehicle_percentage",
        "average_pixel_speed",
        "median_pixel_speed",
        "movement_congestion_score"
    ]

    missing = [c for c in required if c not in mov_clean.columns]
    if missing:
        raise ValueError(f"Movement feature stream is missing required columns: {missing}")

    return mov_clean


# ==============================================================================
# 3. RELATIONAL MERGE & CONGESTION FUSION
# ==============================================================================

def merge_features(vision: pd.DataFrame, movement: pd.DataFrame) -> pd.DataFrame:
    """
    Performs relational key merging on timestamp_seconds.
    """
    # Select columns to merge
    vision_cols = ["timestamp_seconds", "vehicle_count", "vehicle_density_score"]
    # Preserve optional vehicle composition if available
    for col in ["cars", "motorcycles", "buses", "trucks"]:
        if col in vision.columns:
            vision_cols.append(col)

    v_subset = vision[[c for c in vision_cols if c in vision.columns]].drop_duplicates(subset=["timestamp_seconds"])
    m_subset = movement.drop_duplicates(subset=["timestamp_seconds"])

    # Relational merge on timestamp_seconds
    merged = pd.merge(v_subset, m_subset, on="timestamp_seconds", how="inner")

    if merged.empty:
        raise ValueError(
            "Relational merge on 'timestamp_seconds' produced 0 records. "
            "Verify timestamp alignment between vision and movement outputs."
        )

    # Calculate fused vision congestion score
    merged["vision_congestion_score"] = (
        (merged["vehicle_density_score"] * DENSITY_WEIGHT)
        + (merged["movement_congestion_score"] * MOVEMENT_WEIGHT)
    ).clip(0.0, 100.0).round(2)

    def classify_state(score: float) -> str:
        if score >= 75.0:
            return "SEVERE"
        elif score >= 55.0:
            return "HIGH"
        elif score >= 30.0:
            return "MODERATE"
        else:
            return "LOW"

    merged["traffic_state"] = merged["vision_congestion_score"].apply(classify_state)

    # Standard column ordering
    primary_cols = [
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
        "average_speed_kmh",
        "median_speed_kmh",
        "average_pixel_speed",
        "median_pixel_speed",
        "movement_congestion_score",
        "vision_congestion_score",
        "traffic_state",
        "calibration_status"
    ]

    ordered = [c for c in primary_cols if c in merged.columns]
    remaining = [c for c in merged.columns if c not in ordered]
    fused_df = merged[ordered + remaining]

    return fused_df


# ==============================================================================
# 4. VALIDATION REPORTING
# ==============================================================================

def validate(fused: pd.DataFrame):
    """Prints feature fusion audit details."""
    print("\n" + "=" * 70)
    print("VISION FEATURE VALIDATION")
    print("=" * 70)

    print(f"\nRecords generated: {len(fused)}")
    print(f"Vehicle count range: {fused['vehicle_count'].min():.0f} to {fused['vehicle_count'].max():.0f}")
    print(f"Vision congestion range: {fused['vision_congestion_score'].min():.2f} to {fused['vision_congestion_score'].max():.2f}")

    if "average_speed_kmh" in fused.columns and fused["average_speed_kmh"].notnull().any():
        valid_speeds = fused["average_speed_kmh"].dropna()
        print(f"Calibrated speed range (km/h): {valid_speeds.min():.2f} to {valid_speeds.max():.2f} km/h (Mean: {valid_speeds.mean():.2f} km/h)")
        print(f"Calibration status: {fused['calibration_status'].iloc[0] if 'calibration_status' in fused.columns else 'N/A'}")
    else:
        print("Calibrated speed: UNCALIBRATED (NaN)")

    print("\nTraffic state distribution:")
    print(fused["traffic_state"].value_counts().to_string())


# ==============================================================================
# 5. PIPELINE ORCHESTRATION & CLI
# ==============================================================================

def fuse_features(session_id: Optional[str] = None) -> pd.DataFrame:
    """Executes end-to-end feature fusion pipeline."""
    vision_input, movement_input, output_file, session_dir = resolve_paths(session_id)

    print("=" * 70)
    print("ROAD SENSE AI - VISION FEATURE FUSION")
    print("=" * 70)
    if session_id:
        print(f"Active Session : {session_id}")
    print(f"Vision Input   : {vision_input}")
    print(f"Movement Input : {movement_input}")
    print(f"Writing to     : {output_file}")

    # 1. Ingest
    vision_raw, movement_raw = load_inputs(vision_input, movement_input)
    print(f"\nVision records   : {len(vision_raw)}")
    print(f"Movement records : {len(movement_raw)}")

    # 2. Prepare
    vision_prep = prepare_vision_data(vision_raw)
    movement_prep = prepare_movement_data(movement_raw)

    # 3. Merge relationally
    fused = merge_features(vision_prep, movement_prep)
    print(f"Relational fused records : {len(fused)}")

    # 4. Validate
    validate(fused)

    # 5. Preview
    print("\n" + "=" * 70)
    print("FUSED FEATURE PREVIEW")
    print("=" * 70)

    preview_cols = [
        "timestamp_seconds",
        "vehicle_count",
        "tracked_vehicle_count",
        "average_speed_kmh",
        "average_pixel_speed",
        "movement_congestion_score",
        "vision_congestion_score",
        "traffic_state"
    ]
    avail_preview = [c for c in preview_cols if c in fused.columns]
    print(fused[avail_preview].head(15).to_string(index=False))

    # 6. Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("FEATURE FUSION COMPLETE")
    print("=" * 70)
    print(f"\nOutput saved to:\n  {output_file}")

    return fused


def main():
    parser = argparse.ArgumentParser(description="RoadSense AI - Vision Feature Fusion")
    parser.add_argument("--session", type=str, default=None, help="Session identifier (e.g. session_001)")

    args = parser.parse_args()

    fuse_features(session_id=args.session)


if __name__ == "__main__":
    main()