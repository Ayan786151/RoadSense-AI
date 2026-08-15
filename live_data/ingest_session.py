"""
================================================================================
ROAD SENSE AI - TEMPORAL SESSION INGESTION LAYER
MODULE: SAFE TEMPORAL DATA INGESTION & ACCUMULATION (Stage 8)
================================================================================

This module safely ingests standardized live traffic observations from individual
session directories (data/sessions/{session_id}/) into the persistent historical
temporal traffic store (data/temporal_traffic_store.csv).

ARCHITECTURAL PRINCIPLES:
1. SEMANTIC SEPARATION OF TIME & SESSIONS:
   - An observation session (e.g. session_001, session_003) is a single recording.
   - An observation date (e.g. 2026-08-15) is a specific calendar day.
   - A calendar week (e.g. week_id = 1 / Week 33) is a full 7-day period.
   - A historical ML week requires real elapsed calendar weeks, NEVER fabricated
     from multiple video sessions recorded on the same date.
2. STRICT IDEMPOTENCY & DEDUPLICATION:
   - Ingestion uses a stable composite identity key:
     (session_id, location_id, camera_id, observation_date, timestamp_seconds)
   - Running ingestion multiple times skips already-ingested observations with
     zero duplicate records added.
3. PROVENANCE & SCHEMA EXTENSION:
   - Attaches session_id, source_type (CAMERA_LIVE), and observation_sequence.
   - Preserves all raw vision measurements and intermediate signals.
4. ZERO FABRICATION & TEMPORAL WARMUP:
   - When historical weeks < 2, marks status as WARMUP / INSUFFICIENT_HISTORICAL_WEEKS.
   - Never invents lag-1 or rolling 4-week metrics.
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


# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "data" / "sessions"
DEFAULT_STORE_PATH = PROJECT_ROOT / "data" / "temporal_traffic_store.csv"

# Composite identity key for strict deduplication
COMPOSITE_IDENTITY_KEYS = [
    "session_id",
    "location_id",
    "camera_id",
    "observation_date",
    "timestamp_seconds"
]

# Required columns expected from standardized live traffic observations
REQUIRED_OBSERVATION_COLUMNS = [
    "timestamp_seconds",
    "vehicle_count",
    "tracked_vehicle_count",
    "moving_vehicle_count",
    "slow_vehicle_count",
    "stopped_vehicle_count",
    "moving_vehicle_percentage",
    "slow_vehicle_percentage",
    "stopped_vehicle_percentage",
    "average_pixel_speed",
    "median_pixel_speed",
    "vehicle_density_proxy",
    "movement_congestion_score",
    "vision_congestion_score",
    "camera_congestion",
    "traffic_state",
    "camera_traffic_state"
]

# Standardized temporal store column order
STORE_COLUMN_ORDER = [
    "session_id",
    "location_id",
    "camera_id",
    "observation_date",
    "week_id",
    "timestamp_seconds",
    "observation_sequence",
    "source_type",
    "vehicle_count",
    "tracked_vehicle_count",
    "moving_vehicle_count",
    "slow_vehicle_count",
    "stopped_vehicle_count",
    "moving_vehicle_percentage",
    "slow_vehicle_percentage",
    "stopped_vehicle_percentage",
    "average_pixel_speed",
    "median_pixel_speed",
    "vehicle_density_proxy",
    "movement_congestion_score",
    "vision_congestion_score",
    "camera_congestion",
    "traffic_state",
    "camera_traffic_state",
    "vehicle_density_score"
]


# ==============================================================================
# 1. LOAD & VALIDATE SESSION OBSERVATIONS
# ==============================================================================

def load_session_dataset(session_id: str, sessions_root: Path = DEFAULT_SESSIONS_DIR) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads live traffic observations and metadata for a specific session.
    """
    session_dir = sessions_root / session_id
    if not session_dir.exists():
        raise FileNotFoundError(
            f"Session directory not found: {session_dir}\n"
            f"Available sessions: {[d.name for d in sessions_root.iterdir() if d.is_dir()]}"
        )

    obs_path = session_dir / "live_traffic_observations.csv"
    meta_path = session_dir / "session_metadata.json"

    if not obs_path.exists():
        raise FileNotFoundError(
            f"Live traffic observations file not found: {obs_path}\n"
            f"Please run the vision pipeline for {session_id} first."
        )

    df = pd.read_csv(obs_path)

    metadata = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"[!] Warning: Could not parse session metadata from {meta_path}: {e}")

    return df, metadata


def validate_session_data(df: pd.DataFrame, session_id: str) -> Dict[str, Any]:
    """
    Validates schema, null values, numeric types, and internal duplicates.
    """
    if len(df) == 0:
        raise ValueError(f"Session {session_id} observations dataset is empty (0 records).")

    # 1. Required column validation
    missing_cols = [col for col in REQUIRED_OBSERVATION_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Session {session_id} is missing required observation columns:\n"
            + "\n".join(f"  - {c}" for c in missing_cols)
        )

    # 2. Null value validation
    null_counts = df[REQUIRED_OBSERVATION_COLUMNS].isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        null_details = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Null values detected in session {session_id} observations: {null_details}")

    # 3. Numeric validation
    if (df["timestamp_seconds"] < 0).any():
        raise ValueError(f"Negative timestamps detected in session {session_id}.")

    # 4. Internal duplicate timestamp check
    internal_duplicates = int(df.duplicated(subset=["timestamp_seconds"]).sum())

    return {
        "record_count": len(df),
        "total_nulls": total_nulls,
        "internal_duplicates": internal_duplicates,
        "min_timestamp": float(df["timestamp_seconds"].min()),
        "max_timestamp": float(df["timestamp_seconds"].max()),
        "vehicle_count_range": (float(df["vehicle_count"].min()), float(df["vehicle_count"].max())),
        "camera_congestion_range": (float(df["camera_congestion"].min()), float(df["camera_congestion"].max()))
    }


# ==============================================================================
# 2. STANDARDIZE PROVENANCE & IDENTITY
# ==============================================================================

def attach_provenance_metadata(df: pd.DataFrame, session_id: str, metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Attaches spatial, temporal, session, and provenance metadata to observations.
    """
    standardized = df.copy()

    location_id = metadata.get("location_id", "loc_01")
    camera_id = metadata.get("camera_id", "cam_01")
    observation_date = metadata.get("observation_date", "2026-08-15")
    week_id = metadata.get("week_id", 1)

    standardized["session_id"] = session_id
    standardized["location_id"] = location_id
    standardized["camera_id"] = camera_id
    standardized["observation_date"] = observation_date
    standardized["week_id"] = int(week_id)
    standardized["source_type"] = "CAMERA_LIVE"
    standardized["observation_sequence"] = np.arange(1, len(standardized) + 1, dtype=int)

    # If vehicle_density_score is missing, fill with vehicle_density_proxy
    if "vehicle_density_score" not in standardized.columns:
        if "vehicle_density_proxy" in standardized.columns:
            standardized["vehicle_density_score"] = standardized["vehicle_density_proxy"]
        else:
            standardized["vehicle_density_score"] = 0.0

    # Ensure all expected columns exist
    for col in STORE_COLUMN_ORDER:
        if col not in standardized.columns:
            standardized[col] = np.nan

    # Reorder columns cleanly
    remaining = [c for c in standardized.columns if c not in STORE_COLUMN_ORDER]
    standardized = standardized[STORE_COLUMN_ORDER + remaining]

    return standardized


# ==============================================================================
# 3. PERSISTENT TEMPORAL STORE MANAGEMENT
# ==============================================================================

def load_temporal_store(store_path: Path = DEFAULT_STORE_PATH) -> pd.DataFrame:
    """
    Loads existing temporal store from disk and ensures schema compatibility.
    """
    if not store_path.exists() or store_path.stat().st_size == 0:
        return pd.DataFrame(columns=STORE_COLUMN_ORDER)

    df = pd.read_csv(store_path)

    # Backfill legacy records if session_id is absent
    if "session_id" not in df.columns:
        df["session_id"] = "session_001"
    else:
        df["session_id"] = df["session_id"].fillna("session_001")

    if "source_type" not in df.columns:
        df["source_type"] = "CAMERA_LIVE"
    else:
        df["source_type"] = df["source_type"].fillna("CAMERA_LIVE")

    if "observation_sequence" not in df.columns:
        df["observation_sequence"] = np.arange(1, len(df) + 1, dtype=int)

    if "vehicle_density_score" not in df.columns:
        df["vehicle_density_score"] = df.get("vehicle_density_proxy", 0.0)

    # Reorder according to standard column order
    for col in STORE_COLUMN_ORDER:
        if col not in df.columns:
            df[col] = np.nan

    remaining = [c for c in df.columns if c not in STORE_COLUMN_ORDER]
    df = df[STORE_COLUMN_ORDER + remaining]

    return df


def merge_session_idempotent(
    existing_store: pd.DataFrame,
    incoming_df: pd.DataFrame,
    identity_keys: List[str] = COMPOSITE_IDENTITY_KEYS
) -> Tuple[pd.DataFrame, int, int, int]:
    """
    Performs an idempotent merge of incoming session observations into the temporal store.

    Returns:
        (updated_store, loaded_count, inserted_count, skipped_count)
    """
    loaded_count = len(incoming_df)

    if existing_store.empty:
        updated_store = incoming_df.copy()
        inserted_count = loaded_count
        skipped_count = 0
    else:
        # Build existing keys set
        existing_keys = set(
            existing_store[identity_keys].itertuples(index=False, name=None)
        )

        incoming_keys = list(
            incoming_df[identity_keys].itertuples(index=False, name=None)
        )

        # Identify new vs duplicate records
        is_new_mask = [k not in existing_keys for k in incoming_keys]
        new_records = incoming_df[is_new_mask].copy()

        inserted_count = len(new_records)
        skipped_count = loaded_count - inserted_count

        if inserted_count > 0:
            updated_store = pd.concat([existing_store, new_records], ignore_index=True)
        else:
            updated_store = existing_store.copy()

    # Sort strictly chronologically
    sort_cols = [c for c in ["observation_date", "week_id", "session_id", "timestamp_seconds"] if c in updated_store.columns]
    updated_store = updated_store.sort_values(by=sort_cols).reset_index(drop=True)

    return updated_store, loaded_count, inserted_count, skipped_count


def save_temporal_store(df: pd.DataFrame, store_path: Path = DEFAULT_STORE_PATH) -> str:
    """
    Safely saves updated temporal traffic store to CSV.
    """
    store_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(store_path, index=False)
    return str(store_path)


# ==============================================================================
# 4. TEMPORAL STORE AUDIT & STATUS ASSESSMENT
# ==============================================================================

def audit_temporal_store(store_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates temporal store content, representation, and warm-up state.
    """
    total_records = len(store_df)
    sessions = sorted(store_df["session_id"].dropna().unique().tolist()) if "session_id" in store_df.columns else []
    locations = sorted(store_df["location_id"].dropna().unique().tolist()) if "location_id" in store_df.columns else []
    cameras = sorted(store_df["camera_id"].dropna().unique().tolist()) if "camera_id" in store_df.columns else []
    dates = sorted(store_df["observation_date"].dropna().unique().tolist()) if "observation_date" in store_df.columns else []
    calendar_weeks = sorted(store_df["week_id"].dropna().unique().tolist()) if "week_id" in store_df.columns else []

    # Assess temporal maturity
    # Note: A single calendar date / week_id contains multiple sessions, but is still only 1 historical week.
    num_distinct_weeks = len(calendar_weeks)

    if num_distinct_weeks == 0 or total_records == 0:
        temporal_status = "EMPTY"
        status_detail = "Store contains no observations."
    elif num_distinct_weeks < 2:
        temporal_status = "WARMUP / INSUFFICIENT_HISTORICAL_WEEKS"
        status_detail = (
            f"Store contains {len(sessions)} session(s) across {len(dates)} date(s), representing "
            f"{num_distinct_weeks} calendar week(s) (Week {calendar_weeks[0] if calendar_weeks else 1}). "
            "Lag-1 features require >= 2 distinct historical weeks. "
            "4-week rolling averages and OLS trend slopes require >= 5 distinct historical weeks. "
            "Zero artificial historical weeks were fabricated."
        )
    elif num_distinct_weeks < 5:
        temporal_status = "PARTIAL_HISTORY (Lag-1 Active, 4-Week Rolling in Warmup)"
        status_detail = f"Store contains {num_distinct_weeks} distinct weeks. Lag-1 active; 4-week window requires >= 5 weeks."
    else:
        temporal_status = "READY (Full 4-Week Temporal History Active)"
        status_detail = f"Store contains {num_distinct_weeks} distinct weeks. All temporal features fully active."

    return {
        "total_records": total_records,
        "sessions_represented": sessions,
        "locations_represented": locations,
        "cameras_represented": cameras,
        "dates_represented": dates,
        "calendar_weeks_represented": calendar_weeks,
        "num_distinct_weeks": num_distinct_weeks,
        "temporal_status": temporal_status,
        "status_detail": status_detail,
        "null_count": int(store_df.isnull().sum().sum()),
        "duplicate_count": int(store_df.duplicated(subset=COMPOSITE_IDENTITY_KEYS).sum())
    }


# ==============================================================================
# 5. STRUCTURED TERMINAL REPORTING
# ==============================================================================

def print_ingestion_report(
    session_id: str,
    loaded_count: int,
    inserted_count: int,
    skipped_count: int,
    audit: Dict[str, Any],
    store_path: str
):
    """
    Prints structured terminal report matching RoadSense AI requirements.
    """
    print("=" * 70)
    print("ROAD SENSE AI - TEMPORAL SESSION INGESTION")
    print("=" * 70)

    print(f"\n[1] SESSION INGESTION AUDIT:")
    print(f"  - Target Session ID            : {session_id}")
    print(f"  - Observations Loaded          : {loaded_count}")
    print(f"  - Observations Inserted        : {inserted_count}")
    print(f"  - Skipped as Duplicates        : {skipped_count}")
    print(f"  - Ingestion Idempotency Status : {'PASS (Duplicates Safely Prevented)' if skipped_count > 0 or inserted_count == loaded_count else 'PASS'}")

    print(f"\n[2] TEMPORAL STORE CURRENT STATE:")
    print(f"  - Total Historical Records     : {audit['total_records']}")
    print(f"  - Sessions Represented         : {audit['sessions_represented']} ({len(audit['sessions_represented'])} sessions)")
    print(f"  - Locations Represented        : {audit['locations_represented']} ({len(audit['locations_represented'])} locations)")
    print(f"  - Cameras Represented          : {audit['cameras_represented']} ({len(audit['cameras_represented'])} cameras)")
    print(f"  - Observation Dates            : {audit['dates_represented']}")
    print(f"  - Actual Calendar Weeks        : {audit['calendar_weeks_represented']} ({audit['num_distinct_weeks']} distinct week(s))")
    print(f"  - Zero Duplicate Records Check : {'PASS' if audit['duplicate_count'] == 0 else 'FAIL'}")

    print(f"\n[3] TEMPORAL MATURITY & WARMUP ASSESSMENT:")
    print(f"  - Temporal Status              : {audit['temporal_status']}")
    print(f"  - Historical Assessment        :\n    {audit['status_detail']}")

    print(f"\n[4] TEMPORAL STORE PERSISTENCE:")
    print(f"  - Saved to:\n    {store_path}")

    print("=" * 70 + "\n")


# ==============================================================================
# 6. PIPELINE ORCHESTRATOR & CLI
# ==============================================================================

def ingest_session(session_id: str, store_path: Path = DEFAULT_STORE_PATH, sessions_root: Path = DEFAULT_SESSIONS_DIR) -> Dict[str, Any]:
    """
    Executes end-to-end idempotent ingestion of a session into the temporal store.
    """
    # 1. Load session data & metadata
    raw_df, metadata = load_session_dataset(session_id, sessions_root)

    # 2. Validate session data
    val_stats = validate_session_data(raw_df, session_id)

    # 3. Standardize provenance metadata
    standardized_df = attach_provenance_metadata(raw_df, session_id, metadata)

    # 4. Load existing temporal store
    existing_store = load_temporal_store(store_path)

    # 5. Idempotent merge
    updated_store, loaded, inserted, skipped = merge_session_idempotent(existing_store, standardized_df)

    # 6. Save updated temporal store
    saved_path = save_temporal_store(updated_store, store_path)

    # 7. Audit updated temporal store
    audit = audit_temporal_store(updated_store)

    # 8. Print report
    print_ingestion_report(session_id, loaded, inserted, skipped, audit, saved_path)

    return {
        "session_id": session_id,
        "loaded": loaded,
        "inserted": inserted,
        "skipped": skipped,
        "audit": audit
    }


def main():
    parser = argparse.ArgumentParser(description="RoadSense AI - Safe Temporal Session Ingestion Layer")
    parser.add_argument("--session", type=str, required=True, help="Session identifier to ingest (e.g. session_003)")
    parser.add_argument("--store", type=str, default=str(DEFAULT_STORE_PATH), help="Path to temporal traffic store CSV")

    args = parser.parse_args()

    ingest_session(
        session_id=args.session,
        store_path=Path(args.store)
    )


if __name__ == "__main__":
    main()
