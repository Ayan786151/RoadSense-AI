"""
================================================================================
ROADSENSE AI — TEMPORAL DATASET ORGANIZER (WEEKS & DAY OF THE WEEK)
MODULE: backend/temporal_organizer.py
================================================================================

This module ingests raw CSV files dropped into backend/input_csv/, automatically
detects timestamps/dates, organizes records by:
1. Calendar / ISO Weeks (Week 1 to Week 52)
2. Day of the Week (Monday through Sunday)
3. Time of Day (Morning Rush, Midday, Evening Rush, Night)
4. Weekend vs. Weekday

It partitions the data into structured subfolders (by_week/, by_day_of_week/)
and creates consolidated master datasets ready for machine learning model training.
================================================================================
"""

import os
import glob
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
INPUT_CSV_DIR = os.path.join(BACKEND_DIR, "input_csv")
ORGANIZED_DIR = os.path.join(BACKEND_DIR, "organized_data")
BY_WEEK_DIR = os.path.join(ORGANIZED_DIR, "by_week")
BY_DAY_DIR = os.path.join(ORGANIZED_DIR, "by_day_of_week")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Mapping for 1-based Sunday convention (1=Sunday, 2=Monday, ..., 7=Saturday)
SUNDAY_CONVENTION_MAP = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday"
}


def find_input_csv_files(search_dir: Optional[str] = None) -> List[str]:
    """Finds all CSV files placed in the input folder."""
    target_dir = search_dir or INPUT_CSV_DIR
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return []
    
    files = glob.glob(os.path.join(target_dir, "*.csv"))
    # Also check parent data/input_csv if exists
    alt_dir = os.path.join(PROJECT_ROOT, "data", "input_csv")
    if os.path.exists(alt_dir):
        files.extend([f for f in glob.glob(os.path.join(alt_dir, "*.csv")) if f not in files])
    return files


def detect_date_and_temporal_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Intelligently identifies timestamp, date, week, and day_of_week columns.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    
    # Check timestamp/date candidates
    date_candidates = [
        "crash_date", "rash_date", "date", "timestamp", "datetime",
        "date_time", "crash_dt", "incident_date", "time_stamp"
    ]
    date_col = None
    for cand in date_candidates:
        if cand in cols_lower:
            date_col = cols_lower[cand]
            break
            
    # Check existing week column
    week_col = None
    for cand in ["week", "crash_week", "calendar_week", "iso_week"]:
        if cand in cols_lower:
            week_col = cols_lower[cand]
            break
            
    # Check existing day of week column
    day_col = None
    for cand in ["crash_day_of_week", "day_of_week", "dayofweek", "day_name", "weekday", "day"]:
        if cand in cols_lower:
            day_col = cols_lower[cand]
            break

    # Entity / Zone column candidates
    entity_col = None
    for cand in ["beat_of_occurrence", "beat", "zone_id", "zone", "location", "corridor", "station_id"]:
        if cand in cols_lower:
            entity_col = cols_lower[cand]
            break

    # Target column candidates
    target_col = None
    for cand in ["incident_occurred", "crash", "accident", "target", "label", "is_incident", "injuries_total", "has_crash"]:
        if cand in cols_lower:
            target_col = cols_lower[cand]
            break

    return {
        "date_col": date_col,
        "week_col": week_col,
        "day_col": day_col,
        "entity_col": entity_col,
        "target_col": target_col
    }


def categorize_time_of_day(hour: int) -> str:
    """Classifies an hour into an urban traffic time-of-day window."""
    if 6 <= hour < 10:
        return "Morning_Rush"
    elif 10 <= hour < 16:
        return "Midday"
    elif 16 <= hour < 20:
        return "Evening_Rush"
    else:
        return "Night"


def organize_csv_dataset(
    csv_path: str,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingests a raw CSV, organizes records by Week (1-52) and Day of Week (Mon-Sun),
    creates partitioned files, and exports master datasets.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    target_out_dir = output_dir or ORGANIZED_DIR
    by_week_dir = os.path.join(target_out_dir, "by_week")
    by_day_dir = os.path.join(target_out_dir, "by_day_of_week")

    os.makedirs(target_out_dir, exist_ok=True)
    os.makedirs(by_week_dir, exist_ok=True)
    os.makedirs(by_day_dir, exist_ok=True)

    print(f"\n[+] Ingesting CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    - Raw dimensions: {len(df):,} rows x {len(df.columns)} columns")

    temporal_info = detect_date_and_temporal_columns(df)
    date_col = temporal_info["date_col"]
    week_col = temporal_info["week_col"]
    day_col = temporal_info["day_col"]

    df_proc = df.copy()

    # Case 1: Date column is present -> parse into complete temporal hierarchy
    if date_col:
        print(f"    - Parsing timestamps from column: '{date_col}'")
        df_proc["_dt_parsed"] = pd.to_datetime(df_proc[date_col], errors="coerce")
        df_proc = df_proc.dropna(subset=["_dt_parsed"]).sort_values("_dt_parsed").reset_index(drop=True)

        df_proc["week"] = df_proc["_dt_parsed"].dt.isocalendar().week.astype(int)
        df_proc["day_of_week"] = df_proc["_dt_parsed"].dt.dayofweek.astype(int) # 0=Monday, 6=Sunday
        df_proc["day_name"] = df_proc["_dt_parsed"].dt.day_name()
        df_proc["month"] = df_proc["_dt_parsed"].dt.month.astype(int)
        df_proc["hour"] = df_proc["_dt_parsed"].dt.hour.astype(int)
        df_proc["time_of_day"] = df_proc["hour"].apply(categorize_time_of_day)
        df_proc["is_weekend"] = df_proc["day_of_week"].apply(lambda d: 1 if d in [5, 6] else 0)
        df_proc.drop(columns=["_dt_parsed"], inplace=True)

    # Case 2: Week and Day columns already exist separately
    elif week_col and day_col:
        print(f"    - Using existing temporal columns: week='{week_col}', day='{day_col}'")
        df_proc["week"] = pd.to_numeric(df_proc[week_col], errors="coerce").fillna(1).astype(int)
        
        # Check day format (numeric or string)
        if pd.api.types.is_numeric_dtype(df_proc[day_col]):
            unique_days = sorted(df_proc[day_col].dropna().unique())
            if max(unique_days) <= 7 and min(unique_days) >= 1:
                # Likely 1=Sunday or 1=Monday convention
                df_proc["day_name"] = df_proc[day_col].map(SUNDAY_CONVENTION_MAP).fillna("Monday")
                df_proc["day_of_week"] = df_proc["day_name"].apply(lambda n: DAY_NAMES.index(n) if n in DAY_NAMES else 0)
            else:
                df_proc["day_of_week"] = df_proc[day_col].astype(int) % 7
                df_proc["day_name"] = df_proc["day_of_week"].apply(lambda d: DAY_NAMES[d])
        else:
            df_proc["day_name"] = df_proc[day_col].astype(str).str.capitalize()
            df_proc["day_of_week"] = df_proc["day_name"].apply(lambda n: DAY_NAMES.index(n) if n in DAY_NAMES else 0)

        df_proc["is_weekend"] = df_proc["day_of_week"].apply(lambda d: 1 if d in [5, 6] else 0)
        if "time_of_day" not in df_proc.columns:
            df_proc["time_of_day"] = "All_Day"

    # Case 3: Only week column exists -> assign default/uniform day distribution
    elif week_col:
        print(f"    - Found week column '{week_col}'; synthesizing day-of-week breakdown")
        df_proc["week"] = pd.to_numeric(df_proc[week_col], errors="coerce").fillna(1).astype(int)
        np.random.seed(42)
        df_proc["day_of_week"] = np.random.choice(range(7), size=len(df_proc))
        df_proc["day_name"] = df_proc["day_of_week"].apply(lambda d: DAY_NAMES[d])
        df_proc["is_weekend"] = df_proc["day_of_week"].apply(lambda d: 1 if d in [5, 6] else 0)
        df_proc["time_of_day"] = "All_Day"

    else:
        print("    - No timestamp column detected; synthesizing 52-week & 7-day structure")
        np.random.seed(42)
        df_proc["week"] = np.random.choice(range(1, 53), size=len(df_proc))
        df_proc["day_of_week"] = np.random.choice(range(7), size=len(df_proc))
        df_proc["day_name"] = df_proc["day_of_week"].apply(lambda d: DAY_NAMES[d])
        df_proc["is_weekend"] = df_proc["day_of_week"].apply(lambda d: 1 if d in [5, 6] else 0)
        df_proc["time_of_day"] = "All_Day"

    # Ensure binary target exists or synthesize from incident/casualty counts
    target_col = temporal_info["target_col"]
    if target_col and target_col in df_proc.columns:
        if not pd.api.types.is_numeric_dtype(df_proc[target_col]):
            df_proc["incident_occurred"] = df_proc[target_col].astype(str).str.lower().isin(["1", "true", "yes", "y", "crash", "incident", "accident"]).astype(int)
        else:
            if df_proc[target_col].max() > 1:
                df_proc["incident_occurred"] = (df_proc[target_col] > 0).astype(int)
            else:
                df_proc["incident_occurred"] = df_proc[target_col].astype(int)
    else:
        # Default: if raw incident records without explicit non-incident rows, each row is an event
        df_proc["incident_occurred"] = 1

    # Sort sequentially by Week and Day of Week
    df_proc = df_proc.sort_values(by=["week", "day_of_week"]).reset_index(drop=True)

    # 1. Export Consolidated Master Organized Dataset
    master_path = os.path.join(target_out_dir, "master_organized_panel.csv")
    df_proc.to_csv(master_path, index=False)
    print(f"[+] Saved Master Organized Dataset to: {master_path}")

    # 2. Partition by Week (Weeks 1 to 52)
    weeks_present = sorted(df_proc["week"].unique())
    for w in weeks_present:
        week_df = df_proc[df_proc["week"] == w]
        w_file = os.path.join(by_week_dir, f"week_{w:02d}.csv")
        week_df.to_csv(w_file, index=False)

    print(f"[+] Partitioned into {len(weeks_present)} Weekly CSV files in: {by_week_dir}")

    # 3. Partition by Day of Week (Monday to Sunday)
    day_counts = {}
    for day_idx, d_name in enumerate(DAY_NAMES):
        day_df = df_proc[df_proc["day_name"] == d_name]
        d_file = os.path.join(by_day_dir, f"{d_name}.csv")
        day_df.to_csv(d_file, index=False)
        day_counts[d_name] = {
            "records": len(day_df),
            "incidents": int(day_df["incident_occurred"].sum()),
            "incident_rate": round(float(day_df["incident_occurred"].mean() * 100), 2) if len(day_df) > 0 else 0.0
        }

    print(f"[+] Partitioned into 7 Day-of-Week CSV files in: {by_day_dir}")

    # 4. Generate Weekly Aggregation Summary Matrix (Pivot: Week x Day of Week)
    weekly_day_pivot = df_proc.pivot_table(
        index="week",
        columns="day_name",
        values="incident_occurred",
        aggfunc=["count", "sum"],
        fill_value=0
    )
    pivot_path = os.path.join(target_out_dir, "weekly_day_distribution_matrix.csv")
    weekly_day_pivot.to_csv(pivot_path)

    # 5. Build Summary Statistics Dictionary
    summary = {
        "source_file": os.path.basename(csv_path),
        "total_records": len(df_proc),
        "total_columns": len(df_proc.columns),
        "weeks_count": len(weeks_present),
        "min_week": int(min(weeks_present)),
        "max_week": int(max(weeks_present)),
        "total_incidents": int(df_proc["incident_occurred"].sum()),
        "overall_incident_rate_pct": round(float(df_proc["incident_occurred"].mean() * 100), 2),
        "day_of_week_breakdown": day_counts,
        "master_csv_path": master_path,
        "by_week_dir": by_week_dir,
        "by_day_dir": by_day_dir,
        "columns": list(df_proc.columns)
    }

    summary_json_path = os.path.join(target_out_dir, "organization_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Exported Organization Summary to: {summary_json_path}")
    return df_proc, summary


if __name__ == "__main__":
    import sys
    # Find any CSV in input_csv or take argument
    input_files = find_input_csv_files()
    target_csv = sys.argv[1] if len(sys.argv) > 1 else (input_files[0] if input_files else None)
    
    if not target_csv:
        print("[-] No CSV files found in backend/input_csv/. Please drop a CSV file there.")
    else:
        df_res, summ = organize_csv_dataset(target_csv)
        print("\n" + "=" * 60)
        print(" DATASET ORGANIZATION COMPLETE ".center(60, "="))
        print("=" * 60)
        print(f"Total Rows:     {summ['total_records']:,}")
        print(f"Weeks Covered:  Week {summ['min_week']} to Week {summ['max_week']} ({summ['weeks_count']} weeks)")
        print("\nRecords by Day of the Week:")
        for d, vals in summ["day_of_week_breakdown"].items():
            print(f"  {d:<10}: {vals['records']:>5} rows | {vals['incidents']:>4} incidents ({vals['incident_rate']}%)")
        print("=" * 60)
