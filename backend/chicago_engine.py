"""
================================================================================
ROADSENSE AI — FULL-SPAN CHICAGO POLICE CRASH ENGINE & CONTINUOUS TIMELINE
MODULE: backend/chicago_engine.py
================================================================================

This module processes the ENTIRE Chicago Data Portal traffic crash dataset
across all chronological years and weeks (not restricted to a single 52-week cycle):
1. Ingests all 400,000+ crash reports spanning 2016 through 2026 across 275 police beats.
2. Constructs a continuous multi-year chronological calendar grid (over 540 weeks).
3. Explicitly fills every quiet week where no accident occurred with its exact date range.
4. Highlights weeks with crashes in RED with exact timestamps and primary contributory causes.
5. Computes continuous 4-week temporal momentum and trains a Supervised ML Risk Forecaster.
================================================================================
"""

import os
import glob
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

INPUT_CSV_DIR = os.path.join(BACKEND_DIR, "input_csv")
ORGANIZED_DIR = os.path.join(BACKEND_DIR, "organized_data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

CHICAGO_MODEL_PATH = os.path.join(MODELS_DIR, "chicago_risk_model.pkl")
CHICAGO_GRID_CSV = os.path.join(ORGANIZED_DIR, "chicago_full_continuous_grid.csv")
CHICAGO_SUMMARY_JSON = os.path.join(ORGANIZED_DIR, "chicago_zone_intelligence.json")

RANDOM_SEED = 42

try:
    from backend.chicago_beats_reference import (
        resolve_chicago_zone_name,
        EXACT_CHICAGO_BEAT_DETAILS,
        CPD_DISTRICT_PROFILES
    )
except ImportError:
    from chicago_beats_reference import (
        resolve_chicago_zone_name,
        EXACT_CHICAGO_BEAT_DETAILS,
        CPD_DISTRICT_PROFILES
    )

# Alias for backward compatibility
get_chicago_zone_name = resolve_chicago_zone_name
CHICAGO_BEAT_NAMES = EXACT_CHICAGO_BEAT_DETAILS


def find_chicago_csv() -> str:
    """Locates the primary Chicago crash dataset CSV (prioritizing the latest 1M+ row export)."""
    priority_files = [
        os.path.join(INPUT_CSV_DIR, "Traffic_Crashes_-_Crashes_20260901.csv"),
        os.path.join(INPUT_CSV_DIR, "Traffic_Crashes_-_Crashes_20260830.csv"),
    ]
    for pf in priority_files:
        if os.path.exists(pf) and os.path.getsize(pf) > 100000:
            return pf

    candidates = [
        os.path.join(INPUT_CSV_DIR, "benchmark_traffic_crashes.csv"),
        os.path.join(INPUT_CSV_DIR, "chicago_police_crashes_raw.csv"),
        os.path.join(DATA_DIR, "chicago_police_crashes_raw.csv")
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 2000:
            return p
            
    from data.crash_dataset_adapter import generate_benchmark_police_crash_dataset
    out_p = os.path.join(INPUT_CSV_DIR, "benchmark_traffic_crashes.csv")
    os.makedirs(INPUT_CSV_DIR, exist_ok=True)
    df = generate_benchmark_police_crash_dataset()
    df.to_csv(out_p, index=False)
    return out_p


def get_week_date_range(year: int, week_num: int) -> Tuple[str, str, str]:
    """Computes start date, end date, and label for an ISO calendar week."""
    first_day = datetime(year, 1, 1)
    first_monday = first_day + timedelta(days=((7 - first_day.weekday()) % 7))
    if first_day.weekday() <= 3:
        first_monday = first_day - timedelta(days=first_day.weekday())
        
    target_monday = first_monday + timedelta(weeks=(week_num - 1))
    target_sunday = target_monday + timedelta(days=6)
    
    start_str = target_monday.strftime("%Y-%m-%d")
    end_str = target_sunday.strftime("%Y-%m-%d")
    readable_label = f"{target_monday.strftime('%b %d')} - {target_sunday.strftime('%b %d, %Y')}"
    return start_str, end_str, readable_label


def generate_chicago_continuous_full_grid(csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingests the FULL Chicago dataset, generating the continuous chronological grid
    across ALL years and weeks present in the data, extracting real empirical streets.
    """
    target_csv = csv_path or find_chicago_csv()
    os.makedirs(ORGANIZED_DIR, exist_ok=True)
    
    print(f"[+] Ingesting Full Chicago CSV: {target_csv} ({os.path.getsize(target_csv)/(1024*1024):.1f} MB)")
    
    # Load required columns including street location section
    use_cols = [
        "CRASH_DATE", "POSTED_SPEED_LIMIT", "WEATHER_CONDITION",
        "LIGHTING_CONDITION", "FIRST_CRASH_TYPE", "INTERSECTION_RELATED_I",
        "PRIM_CONTRIBUTORY_CAUSE", "BEAT_OF_OCCURRENCE",
        "STREET_NO", "STREET_DIRECTION", "STREET_NAME"
    ]
    df_raw = pd.read_csv(target_csv, usecols=lambda c: c in use_cols)
    
    date_col = "CRASH_DATE" if "CRASH_DATE" in df_raw.columns else df_raw.columns[0]
    beat_col = "BEAT_OF_OCCURRENCE" if "BEAT_OF_OCCURRENCE" in df_raw.columns else df_raw.columns[1]
    
    df_raw["_dt"] = pd.to_datetime(df_raw[date_col], format="mixed", errors="coerce")
    df_raw = df_raw.dropna(subset=["_dt", beat_col]).copy()
    
    # Clean beat codes
    df_raw["beat_id"] = df_raw[beat_col].astype(str).str.replace(".0", "", regex=False).str.strip()
    
    # Filter realistic years (2016 to present)
    df_raw["year"] = df_raw["_dt"].dt.year.astype(int)
    df_raw = df_raw[(df_raw["year"] >= 2016) & (df_raw["year"] <= 2026)].copy()
    df_raw["week"] = df_raw["_dt"].dt.isocalendar().week.astype(int)
    
    # Extract empirical streets from CSV street section
    if "STREET_NAME" in df_raw.columns:
        s_dir = df_raw["STREET_DIRECTION"].fillna("").astype(str).str.strip() if "STREET_DIRECTION" in df_raw.columns else ""
        s_name = df_raw["STREET_NAME"].fillna("").astype(str).str.strip()
        df_raw["_street_full"] = (s_dir + " " + s_name).str.strip()
    else:
        df_raw["_street_full"] = ""

    # Pre-compute empirical street corridors directly from the dataset
    beat_empirical_streets = {}
    for b, b_group in df_raw.groupby("beat_id"):
        top_s = b_group["_street_full"].value_counts()
        valid_streets = [s.title() for s in top_s.index if s and s != " " and len(s) > 2 and s.lower() != "unknown"]
        if valid_streets:
            beat_empirical_streets[b] = " & ".join(valid_streets[:2])
        else:
            beat_empirical_streets[b] = ""
    
    # Standardize weather to upper
    if "WEATHER_CONDITION" in df_raw.columns:
        df_raw["WEATHER_CONDITION"] = df_raw["WEATHER_CONDITION"].fillna("CLEAR").astype(str).str.upper()
    else:
        df_raw["WEATHER_CONDITION"] = "CLEAR"

    # Compute year_week sequence
    df_raw["year_week"] = df_raw["year"].astype(str) + "-W" + df_raw["week"].astype(str).str.zfill(2)
    
    all_periods = sorted(df_raw["year_week"].unique())
    all_beats = sorted(df_raw["beat_id"].unique())
    
    print(f"[+] Ingested {len(df_raw):,} crashes across {len(all_beats)} beats and {len(all_periods)} continuous weeks.")

    # Pre-compute beat physical properties with empirical street names from CSV
    beat_meta = {}
    for b in all_beats:
        b_rows = df_raw[df_raw["beat_id"] == b]
        info = resolve_chicago_zone_name(b)
        # Use authentic Chicago beat name exactly as displayed on the frontend
        zone_display_name = info["name"]

        spd = float(b_rows["POSTED_SPEED_LIMIT"].median()) if "POSTED_SPEED_LIMIT" in b_rows.columns else 30.0
        inter = float((b_rows["INTERSECTION_RELATED_I"] == "Y").mean()) if "INTERSECTION_RELATED_I" in b_rows.columns else 0.48
        wthr = str(b_rows["WEATHER_CONDITION"].mode()[0]) if not b_rows["WEATHER_CONDITION"].empty else "CLEAR"
        
        beat_meta[b] = {
            "name": zone_display_name,
            "district": info["district"],
            "zone_type": info["type"],
            "posted_speed_limit": spd,
            "intersection_ratio": round(inter, 3),
            "dominant_weather": wthr,
            "total_crashes": len(b_rows)
        }

    # Aggregate weekly crashes by (beat_id, year_week)
    grouped_crashes = df_raw.groupby(["beat_id", "year_week"]).agg(
        crash_count=("_dt", "count"),
        first_crash_timestamp=("_dt", "min"),
        last_crash_timestamp=("_dt", "max"),
        primary_cause=("PRIM_CONTRIBUTORY_CAUSE", lambda x: x.mode()[0] if not x.empty and not pd.isna(x.mode()[0]) else "Traffic Collision"),
        weather=("WEATHER_CONDITION", lambda x: x.mode()[0] if not x.empty else "CLEAR")
    ).reset_index()

    # Build Complete Cartesian Grid: all_beats x all_periods
    cartesian_index = pd.MultiIndex.from_product([all_beats, all_periods], names=["beat_id", "year_week"]).to_frame().reset_index(drop=True)
    
    # Merge crashes onto Cartesian grid
    grid = pd.merge(cartesian_index, grouped_crashes, on=["beat_id", "year_week"], how="left")
    
    # Fill in quiet safe weeks (0 crashes)
    grid["crash_count"] = grid["crash_count"].fillna(0).astype(int)
    grid["incident_occurred"] = (grid["crash_count"] > 0).astype(int)
    grid["status"] = grid["incident_occurred"].apply(lambda x: "CRASH REPORTED" if x == 1 else "SAFE CORRIDOR (NO ACCIDENTS)")
    grid["color_code"] = grid["incident_occurred"].apply(lambda x: "#ef4444" if x == 1 else "#22c55e")
    
    # Parse Year and Week number
    grid["year"] = grid["year_week"].apply(lambda x: int(x.split("-W")[0]))
    grid["week"] = grid["year_week"].apply(lambda x: int(x.split("-W")[1]))

    # Compute date ranges
    date_range_cache = {}
    for p in all_periods:
        yr = int(p.split("-W")[0])
        wk = int(p.split("-W")[1])
        s_d, e_d, r_lbl = get_week_date_range(yr, wk)
        date_range_cache[p] = (s_d, e_d, r_lbl)

    grid["start_date"] = grid["year_week"].apply(lambda p: date_range_cache[p][0])
    grid["end_date"] = grid["year_week"].apply(lambda p: date_range_cache[p][1])
    grid["date_range"] = grid["year_week"].apply(lambda p: date_range_cache[p][2])

    # Assign beat physical properties
    grid["zone_id"] = grid["beat_id"]
    grid["zone_name"] = grid["beat_id"].apply(lambda b: beat_meta[b]["name"])
    grid["district"] = grid["beat_id"].apply(lambda b: beat_meta[b]["district"])
    grid["zone_type"] = grid["beat_id"].apply(lambda b: beat_meta[b]["zone_type"])
    grid["posted_speed_limit"] = grid["beat_id"].apply(lambda b: beat_meta[b]["posted_speed_limit"])
    grid["intersection_ratio"] = grid["beat_id"].apply(lambda b: beat_meta[b]["intersection_ratio"])
    
    # Fill weather and causes
    grid["weather_condition"] = grid.apply(
        lambda r: r["weather"] if pd.notna(r["weather"]) else beat_meta[r["beat_id"]]["dominant_weather"],
        axis=1
    )
    grid["primary_causes"] = grid["primary_cause"].fillna("None - Zero Incidents")
    grid["crash_timestamps"] = grid["first_crash_timestamp"].dt.strftime("%Y-%m-%d %H:%M").fillna("None (Safe)")

    # Sort sequentially
    grid = grid.sort_values(["beat_id", "year_week"]).reset_index(drop=True)

    # Compute Vectorized 4-Week Rolling Momentum per Beat
    g = grid.groupby("beat_id")["crash_count"]
    grid["crashes_lag1"] = g.shift(1).fillna(0).astype(float)
    grid["crashes_rolling4w_avg"] = g.shift(1).rolling(4, min_periods=1).mean().fillna(0).round(2)
    
    l1 = g.shift(1).fillna(0)
    l2 = g.shift(2).fillna(0)
    l3 = g.shift(3).fillna(0)
    l4 = g.shift(4).fillna(0)
    grid["crashes_trend4w_slope"] = ((3.0 * l1 + 1.0 * l2 - 1.0 * l3 - 3.0 * l4) / 10.0).round(3)

    # Save to disk
    grid.to_csv(CHICAGO_GRID_CSV, index=False)
    print(f"[+] Saved Full Multi-Year Continuous Grid ({len(grid):,} zone-weeks) to: {CHICAGO_GRID_CSV}")

    # Summary Statistics
    total_cells = len(grid)
    accident_cells = int(grid["incident_occurred"].sum())
    safe_cells = total_cells - accident_cells

    zone_stats = {}
    for b in all_beats:
        z_df = grid[grid["beat_id"] == b]
        z_acc = int(z_df["incident_occurred"].sum())
        zone_stats[b] = {
            "name": beat_meta[b]["name"],
            "district": beat_meta[b]["district"],
            "total_crashes": int(z_df["crash_count"].sum()),
            "accident_weeks": z_acc,
            "safe_weeks": int(len(z_df) - z_acc),
            "safe_week_pct": round(((len(z_df) - z_acc) / len(z_df)) * 100.0, 1),
            "avg_weekly_crashes": round(float(z_df["crash_count"].mean()), 2)
        }

    summary = {
        "source_dataset": os.path.basename(target_csv),
        "total_zones": len(all_beats),
        "total_weeks": len(all_periods),
        "min_period": all_periods[0],
        "max_period": all_periods[-1],
        "total_timeline_observations": total_cells,
        "accident_weeks_total": accident_cells,
        "safe_weeks_total": safe_cells,
        "safe_weeks_pct": round((safe_cells / total_cells) * 100.0, 1),
        "accident_weeks_pct": round((accident_cells / total_cells) * 100.0, 1),
        "all_periods": all_periods,
        "zone_breakdown": zone_stats,
        "grid_csv_path": CHICAGO_GRID_CSV
    }

    with open(CHICAGO_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return grid, summary


def train_chicago_risk_model(df_grid: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains Supervised ML Risk Forecaster on the Full Continuous Grid:
    - Train split: First 80% of chronological weeks (e.g. 2016 through 2024)
    - Test split: Latest 20% of chronological weeks (e.g. 2025 to 2026)
    Evaluates real accuracy, precision, recall, and confusion matrix.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    all_periods = sorted(df_grid["year_week"].unique())
    split_idx = int(len(all_periods) * 0.80)
    
    train_periods = all_periods[:split_idx]
    test_periods = all_periods[split_idx:]

    cat_cols = ["weather_condition", "district", "zone_type"]
    num_cols = [
        "posted_speed_limit", "intersection_ratio",
        "crashes_lag1", "crashes_rolling4w_avg", "crashes_trend4w_slope"
    ]
    all_features = cat_cols + num_cols

    train_mask = df_grid["year_week"].isin(train_periods)
    test_mask = df_grid["year_week"].isin(test_periods)

    X_train = df_grid.loc[train_mask, all_features]
    y_train = df_grid.loc[train_mask, "incident_occurred"].astype(int)

    X_test = df_grid.loc[test_mask, all_features]
    y_test = df_grid.loc[test_mask, "incident_occurred"].astype(int)
    test_df = df_grid[test_mask].copy()

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=140,
        max_depth=6,
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)

    test_probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X_test)
    test_preds = (test_probs >= 0.50).astype(int)

    acc = float(accuracy_score(y_test, test_preds)) * 100.0
    rec = float(recall_score(y_test, test_preds, zero_division=0)) * 100.0
    prec = float(precision_score(y_test, test_preds, zero_division=0)) * 100.0
    f1 = float(f1_score(y_test, test_preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, test_probs)) if (y_test.nunique() > 1) else 0.5
    cm = confusion_matrix(y_test, test_preds, labels=[0, 1])

    joblib.dump(pipe, CHICAGO_MODEL_PATH)

    test_df["predicted_risk_prob"] = np.round(test_probs, 3)
    test_df["predicted_warning"] = test_preds
    test_df["prediction_correct"] = (test_df["incident_occurred"] == test_preds).astype(int)

    return {
        "model_path": CHICAGO_MODEL_PATH,
        "accuracy_pct": round(acc, 2),
        "recall_pct": round(rec, 2),
        "precision_pct": round(prec, 2),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "test_periods": test_periods,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1])
        },
        "test_df": test_df,
        "pipeline": pipe
    }


def get_or_create_chicago_grid() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Loads or creates the full continuous Chicago grid and model results."""
    if os.path.exists(CHICAGO_GRID_CSV) and os.path.exists(CHICAGO_SUMMARY_JSON):
        df_grid = pd.read_csv(CHICAGO_GRID_CSV)
        with open(CHICAGO_SUMMARY_JSON, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        df_grid, summary = generate_chicago_continuous_full_grid()

    model_eval = train_chicago_risk_model(df_grid)
    return df_grid, summary, model_eval


if __name__ == "__main__":
    grid, summ = generate_chicago_continuous_full_grid()
    ev = train_chicago_risk_model(grid)
    print("\n" + "=" * 80)
    print(" FULL MULTI-YEAR CHICAGO POLICE CRASH ENGINE COMPLETED ".center(80, "="))
    print("=" * 80)
    print(f"Total Zones:                 {summ['total_zones']}")
    print(f"Chronological Timeline:      {summ['min_period']} to {summ['max_period']} ({summ['total_weeks']} weeks)")
    print(f"Total Timeline Observations: {summ['total_timeline_observations']:,} zone-weeks")
    print(f"Accident Weeks (RED):        {summ['accident_weeks_total']:,} ({summ['accident_weeks_pct']}%)")
    print(f"Safe Weeks (GREEN):          {summ['safe_weeks_total']:,} ({summ['safe_weeks_pct']}%)")
    print(f"Test Accuracy:               {ev['accuracy_pct']}%")
    print(f"Test Recall:                 {ev['recall_pct']}%")
    print(f"Confusion Matrix:            {ev['confusion_matrix']}")
    print("=" * 80 + "\n")
