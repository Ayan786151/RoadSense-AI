"""
================================================================================
ROADSENSE AI — POLICE CRASH DATASET ADAPTER & TEMPORAL ROLLING ENGINE
MODULE: data/crash_dataset_adapter.py
================================================================================

This module ingests raw municipal crash datasets containing the 18 specific factors:
- Temporal: CRASH_DATE / RASH_DATE, CRASH_MONTH, CRASH_DAY_OF_WEEK
- Spatial: BEAT_OF_OCCURRENCE, LATITUDE, LONGITUDE
- Road Geometry: POSTED_SPEED_LIMIT, TRAFFICWAY_TYPE, ALIGNMENT, INTERSECTION_RELATED_I, ROAD_DEFECT
- Environmental: WEATHER_CONDITION, LIGHTING_CONDITION, ROADWAY_SURFACE_COND
- Kinematic & Impact: FIRST_CRASH_TYPE, PRIM_CONTRIBUTORY_CAUSE, NUM_UNITS, INJURIES_TOTAL

It structures the records, computes 4-week temporal momentum per beat zone,
stores the organized dataset in SQLite (data/road_sense_master.db),
and provides seamless feeding into the 4-week ML reinforcement engine.
================================================================================
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR if os.path.basename(SCRIPT_DIR) == "traffic_sim" else os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "road_sense_master.db")
CRASH_PROCESSED_CSV = os.path.join(DATA_DIR, "crash_temporal_4w_features.csv")


# ==============================================================================
# 0. 18-FACTOR GLOSSARY & DOMAIN EXPLANATIONS
# ==============================================================================

CRASH_FACTOR_DEFINITIONS = {
    "CRASH_DATE": {
        "category": "Temporal Dimension",
        "type": "Timestamp / Date",
        "unit": "YYYY-MM-DD HH:MM:SS",
        "description": "Exact calendar date and timestamp of the police crash report."
    },
    "RASH_DATE": {
        "category": "Temporal Dimension",
        "type": "Timestamp / Date",
        "unit": "YYYY-MM-DD HH:MM:SS",
        "description": "Exact calendar date and timestamp of the police crash report (alias for CRASH_DATE)."
    },
    "POSTED_SPEED_LIMIT": {
        "category": "Road Geometry & Speed",
        "type": "Numeric",
        "unit": "mph / km/h",
        "description": "Legally posted maximum speed limit on the roadway segment (e.g. 25, 30, 45, 55)."
    },
    "WEATHER_CONDITION": {
        "category": "Environmental & Ambient",
        "type": "Categorical",
        "unit": "Weather State",
        "description": "Atmospheric condition at crash moment: CLEAR, RAIN, SNOW, FOG/SMOKE/HAZE, SEVERE RAIN."
    },
    "LIGHTING_CONDITION": {
        "category": "Environmental & Ambient",
        "type": "Categorical",
        "unit": "Lighting State",
        "description": "Ambient visibility: DAYLIGHT, DARKNESS LIGHTED ROAD, DARKNESS, DUSK, DAWN."
    },
    "FIRST_CRASH_TYPE": {
        "category": "Kinematic & Impact Archetype",
        "type": "Categorical",
        "unit": "Collision Type",
        "description": "Primary vehicle impact pattern: REAR END, SIDESWIPE SAME DIRECTION, TURNING, PEDESTRIAN, ANGLE."
    },
    "TRAFFICWAY_TYPE": {
        "category": "Roadway Infrastructure",
        "type": "Categorical",
        "unit": "Roadway Layout",
        "description": "Physical roadway layout: DIVIDED - W/MEDIAN, ONE-WAY, FOUR WAY INTERSECTION, NOT DIVIDED."
    },
    "ALIGNMENT": {
        "category": "Road Geometry & Alignment",
        "type": "Categorical",
        "unit": "Geometry Type",
        "description": "Horizontal and vertical roadway profile: STRAIGHT AND LEVEL, CURVE LEVEL, STRAIGHT ON GRADE."
    },
    "ROADWAY_SURFACE_COND": {
        "category": "Environmental & Ambient",
        "type": "Categorical",
        "unit": "Surface Friction",
        "description": "Pavement tire-grip state: DRY, WET, SNOW OR SLUSH, ICE, SAND/MUD/DIRT."
    },
    "ROAD_DEFECT": {
        "category": "Infrastructure Quality",
        "type": "Categorical",
        "unit": "Defect State",
        "description": "Physical damage on pavement: NO DEFECTS, RUT/HOLES/POTHOLES, WORN SURFACE, DEBRIS ON ROAD."
    },
    "INTERSECTION_RELATED_I": {
        "category": "Road Geometry & Intersection",
        "type": "Binary / Categorical",
        "unit": "Y / N Flag",
        "description": "Whether the crash occurred at or within the influence zone of a traffic intersection (Y/N)."
    },
    "PRIM_CONTRIBUTORY_CAUSE": {
        "category": "Human & Environmental Root Cause",
        "type": "Categorical",
        "unit": "Cause Archetype",
        "description": "Investigating officer's determination of root cause: FAILING TO YIELD, FOLLOWING TOO CLOSELY, SPEEDING, WEATHER."
    },
    "BEAT_OF_OCCURRENCE": {
        "category": "Spatial & Patrol Sector",
        "type": "Categorical / Integer",
        "unit": "Beat ID",
        "description": "Police patrol beat sector code representing the municipal geographic zone."
    },
    "NUM_UNITS": {
        "category": "Kinematic & Impact Archetype",
        "type": "Integer",
        "unit": "Units Count",
        "description": "Total number of vehicles, bicycles, or pedestrian entities involved in the incident."
    },
    "CRASH_MONTH": {
        "category": "Temporal Dimension",
        "type": "Integer (1-12)",
        "unit": "Month",
        "description": "Calendar month of occurrence (1 = January to 12 = December) capturing seasonal variance."
    },
    "INJURIES_TOTAL": {
        "category": "Safety & Casualty Target",
        "type": "Integer",
        "unit": "Injuries Count",
        "description": "Total number of individuals sustaining injuries (fatal, incapacitating, or non-incapacitating)."
    },
    "CRASH_DAY_OF_WEEK": {
        "category": "Temporal Dimension",
        "type": "Integer (1-7)",
        "unit": "Day of Week",
        "description": "Day of the week (1 = Sunday, 2 = Monday, ... 7 = Saturday) capturing weekend surge patterns."
    },
    "LATITUDE": {
        "category": "Spatial Coordinates",
        "type": "Float",
        "unit": "Degrees N",
        "description": "GPS Latitude coordinate of the crash location."
    },
    "LONGITUDE": {
        "category": "Spatial Coordinates",
        "type": "Float",
        "unit": "Degrees E",
        "description": "GPS Longitude coordinate of the crash location."
    }
}


# ==============================================================================
# 1. GENERATE REALISTIC POLICE CRASH DATASET ACCORDING TO THIS EXACT SCHEMA
# ==============================================================================

def generate_benchmark_police_crash_dataset(num_records: int = 5000) -> pd.DataFrame:
    """
    Generates a realistic municipal police crash dataset matching the exact 18-column schema
    across 20 police beats over 52 consecutive calendar weeks.
    """
    np.random.seed(42)
    beats = [f"BEAT_{i:03d}" for i in range(101, 121)] # 20 Beats
    
    weather_choices = ["CLEAR", "RAIN", "SNOW", "FOG/SMOKE/HAZE", "SEVERE RAIN"]
    weather_p = [0.70, 0.18, 0.06, 0.03, 0.03]

    lighting_choices = ["DAYLIGHT", "DARKNESS, LIGHTED ROAD", "DARKNESS", "DUSK", "DAWN"]
    lighting_p = [0.58, 0.28, 0.08, 0.03, 0.03]

    trafficway_choices = ["DIVIDED - W/MEDIAN", "NOT DIVIDED", "ONE-WAY", "FOUR WAY INTERSECTION", "T-INTERSECTION", "ALLEY"]
    trafficway_p = [0.35, 0.30, 0.15, 0.12, 0.05, 0.03]

    surface_choices = ["DRY", "WET", "SNOW OR SLUSH", "ICE", "SAND, MUD, DIRT"]
    surface_p = [0.72, 0.20, 0.05, 0.02, 0.01]

    defect_choices = ["NO DEFECTS", "RUT, HOLES", "WORN SURFACE", "SHOULDER DEFECT", "DEBRIS ON ROAD"]
    defect_p = [0.82, 0.09, 0.05, 0.02, 0.02]

    crash_types = ["REAR END", "SIDESWIPE SAME DIRECTION", "TURNING", "ANGLE", "PARKED MOTOR VEHICLE", "PEDESTRIAN", "FIXED OBJECT"]
    crash_p = [0.30, 0.22, 0.16, 0.14, 0.10, 0.05, 0.03]

    causes = [
        "FAILING TO YIELD RIGHT-OF-WAY",
        "FOLLOWING TOO CLOSELY",
        "DRIVING SKILLS/KNOWLEDGE/EXPERIENCE",
        "WEATHER",
        "DISREGARDING TRAFFIC SIGNALS",
        "IMPROPER OVERTAKING/PASSING",
        "PHYSICAL CONDITION OF DRIVER"
    ]
    cause_p = [0.28, 0.24, 0.18, 0.12, 0.08, 0.06, 0.04]

    speed_limits = [20, 25, 30, 35, 40, 45, 50, 55]
    speed_p = [0.05, 0.35, 0.30, 0.12, 0.08, 0.05, 0.03, 0.02]

    # Generate dates across 52 weeks (Year 2026)
    start_date = pd.Timestamp("2026-01-01")
    records = []

    for i in range(num_records):
        # Pick day in year (1 to 364)
        day_offset = int(np.random.randint(0, 364))
        crash_dt = start_date + pd.Timedelta(days=day_offset, hours=int(np.random.randint(0, 24)), minutes=int(np.random.randint(0, 60)))
        
        month = crash_dt.month
        day_of_week = (crash_dt.dayofweek + 1) # 1 = Sunday convention
        
        beat = np.random.choice(beats)
        beat_idx = int(beat.split("_")[1]) - 100
        
        # Spatial Coordinates (around urban metro coordinates)
        lat = 41.8781 + (beat_idx * 0.015) + np.random.normal(0, 0.003)
        lon = -87.6298 + (beat_idx * 0.012) + np.random.normal(0, 0.003)

        weather = np.random.choice(weather_choices, p=weather_p)
        lighting = np.random.choice(lighting_choices, p=lighting_p)
        trafficway = np.random.choice(trafficway_choices, p=trafficway_p)
        alignment = np.random.choice(["STRAIGHT AND LEVEL", "CURVE, LEVEL", "STRAIGHT ON GRADE", "CURVE ON GRADE"], p=[0.85, 0.08, 0.05, 0.02])
        surface = np.random.choice(surface_choices, p=surface_p) if weather == "CLEAR" else np.random.choice(["WET", "SNOW OR SLUSH", "ICE"], p=[0.75, 0.18, 0.07])
        defect = np.random.choice(defect_choices, p=defect_p)
        intersection_flag = np.random.choice(["Y", "N"], p=[0.48, 0.52])
        first_crash = np.random.choice(crash_types, p=crash_p)
        prim_cause = np.random.choice(causes, p=cause_p)
        speed_lim = np.random.choice(speed_limits, p=speed_p)
        num_units = np.random.choice([1, 2, 3, 4], p=[0.12, 0.76, 0.09, 0.03])
        
        # Injuries (Higher risk in rain, high speed, or intersection)
        injury_prob = 0.12 + (0.15 if weather in ["RAIN", "SEVERE RAIN"] else 0.0) + (0.10 if speed_lim >= 45 else 0.0) + (0.08 if intersection_flag == "Y" else 0.0)
        has_injury = (np.random.rand() < injury_prob)
        injuries_total = int(np.random.choice([1, 2, 3], p=[0.75, 0.20, 0.05])) if has_injury else 0

        records.append({
            "CRASH_DATE": crash_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "POSTED_SPEED_LIMIT": speed_lim,
            "WEATHER_CONDITION": weather,
            "LIGHTING_CONDITION": lighting,
            "FIRST_CRASH_TYPE": first_crash,
            "TRAFFICWAY_TYPE": trafficway,
            "ALIGNMENT": alignment,
            "ROADWAY_SURFACE_COND": surface,
            "ROAD_DEFECT": defect,
            "INTERSECTION_RELATED_I": intersection_flag,
            "PRIM_CONTRIBUTORY_CAUSE": prim_cause,
            "BEAT_OF_OCCURRENCE": beat,
            "NUM_UNITS": num_units,
            "CRASH_MONTH": month,
            "INJURIES_TOTAL": injuries_total,
            "CRASH_DAY_OF_WEEK": day_of_week,
            "LATITUDE": round(lat, 6),
            "LONGITUDE": round(lon, 6)
        })

    df = pd.DataFrame(records).sort_values("CRASH_DATE").reset_index(drop=True)
    return df


# ==============================================================================
# 2. TRANSFORM RAW CRASH CSV INTO 4-WEEK ROLLING PANEL TIME-SERIES
# ==============================================================================

def transform_crash_records_to_4week_panel(
    df_raw: pd.DataFrame,
    date_col: str = "CRASH_DATE",
    beat_col: str = "BEAT_OF_OCCURRENCE"
) -> pd.DataFrame:
    """
    Transforms individual raw crash incident rows into a structured Weekly Panel Dataset
    per Police Beat, automatically engineering 4-week temporal momentum features
    and 5th-week environmental circumstances.
    """
    df = df_raw.copy()
    
    # Handle alias
    if date_col not in df.columns and "RASH_DATE" in df.columns:
        date_col = "RASH_DATE"
    if beat_col not in df.columns:
        # Auto-detect beat or location
        for c in ["BEAT", "BEAT_OF_OCCURRENCE", "ZONE_ID", "LOCATION"]:
            if c in df.columns:
                beat_col = c
                break

    # Parse Dates and extract Year-Week
    df["dt_parsed"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["dt_parsed"]).sort_values("dt_parsed").reset_index(drop=True)

    # Compute calendar week index (1 to 52)
    df["week"] = df["dt_parsed"].dt.isocalendar().week.astype(int)
    df["beat_id"] = df[beat_col].astype(str)

    # Aggregate weekly summary metrics per Beat
    beats = sorted(df["beat_id"].unique())
    weeks = list(range(1, 53))

    # Create complete Cartesian grid of (beat × week)
    grid = pd.MultiIndex.from_product([beats, weeks], names=["beat_id", "week"]).to_frame().reset_index(drop=True)

    # Weekly aggregates
    weekly_agg = df.groupby(["beat_id", "week"]).agg(
        crash_count=("dt_parsed", "count"),
        injuries_total=("INJURIES_TOTAL", "sum"),
        avg_speed_limit=("POSTED_SPEED_LIMIT", "mean"),
        max_speed_limit=("POSTED_SPEED_LIMIT", "max"),
        dominant_weather=("WEATHER_CONDITION", lambda x: x.mode()[0] if not x.empty else "CLEAR"),
        dominant_lighting=("LIGHTING_CONDITION", lambda x: x.mode()[0] if not x.empty else "DAYLIGHT"),
        dominant_surface=("ROADWAY_SURFACE_COND", lambda x: x.mode()[0] if not x.empty else "DRY"),
        dominant_trafficway=("TRAFFICWAY_TYPE", lambda x: x.mode()[0] if not x.empty else "DIVIDED - W/MEDIAN"),
        dominant_alignment=("ALIGNMENT", lambda x: x.mode()[0] if not x.empty else "STRAIGHT AND LEVEL"),
        dominant_defect=("ROAD_DEFECT", lambda x: x.mode()[0] if not x.empty else "NO DEFECTS"),
        intersection_crash_ratio=("INTERSECTION_RELATED_I", lambda x: (x == "Y").mean() if not x.empty else 0.0),
        avg_num_units=("NUM_UNITS", "mean"),
        latitude=("LATITUDE", "mean"),
        longitude=("LONGITUDE", "mean")
    ).reset_index()

    # Merge onto complete grid
    panel = pd.merge(grid, weekly_agg, on=["beat_id", "week"], how="left")
    panel["crash_count"] = panel["crash_count"].fillna(0).astype(int)
    panel["injuries_total"] = panel["injuries_total"].fillna(0).astype(int)
    panel["incident_occurred"] = (panel["crash_count"] > 0).astype(int)
    panel["severe_injury_occurred"] = (panel["injuries_total"] > 0).astype(int)

    # Fill defaults for circumstances
    panel["avg_speed_limit"] = panel["avg_speed_limit"].fillna(30.0)
    panel["dominant_weather"] = panel["dominant_weather"].fillna("CLEAR")
    panel["dominant_lighting"] = panel["dominant_lighting"].fillna("DAYLIGHT")
    panel["dominant_surface"] = panel["dominant_surface"].fillna("DRY")
    panel["dominant_trafficway"] = panel["dominant_trafficway"].fillna("DIVIDED - W/MEDIAN")
    panel["dominant_alignment"] = panel["dominant_alignment"].fillna("STRAIGHT AND LEVEL")
    panel["dominant_defect"] = panel["dominant_defect"].fillna("NO DEFECTS")
    panel["intersection_crash_ratio"] = panel["intersection_crash_ratio"].fillna(0.0)
    panel["avg_num_units"] = panel["avg_num_units"].fillna(2.0)
    panel["latitude"] = panel.groupby("beat_id")["latitude"].transform(lambda x: x.fillna(x.dropna().mean() if not x.dropna().empty else 41.8781))
    panel["longitude"] = panel.groupby("beat_id")["longitude"].transform(lambda x: x.fillna(x.dropna().mean() if not x.dropna().empty else -87.6298))

    # Sort sequentially
    panel = panel.sort_values(["beat_id", "week"]).reset_index(drop=True)

    # Engineer 4-Week Rolling Temporal Features per Beat
    records = []
    for beat, b_df in panel.groupby("beat_id"):
        b_df = b_df.sort_values("week").reset_index(drop=True)
        crashes = b_df["crash_count"].values
        injuries = b_df["injuries_total"].values

        for idx, row in b_df.iterrows():
            w = int(row["week"])
            row_dict = row.to_dict()

            if idx >= 4:
                # 4-Week Feeder History (t-4 to t-1)
                hist_c = crashes[idx-4:idx]
                hist_inj = injuries[idx-4:idx]

                row_dict["crashes_lag1"] = float(hist_c[-1])
                row_dict["injuries_lag1"] = float(hist_inj[-1])
                row_dict["crashes_rolling4w_avg"] = float(np.mean(hist_c))
                row_dict["injuries_rolling4w_avg"] = float(np.mean(hist_inj))
                
                # Closed-form OLS linear slope
                row_dict["crashes_trend4w_slope"] = float((3*hist_c[3] + hist_c[2] - hist_c[1] - 3*hist_c[0]) / 10.0)
                row_dict["injuries_trend4w_slope"] = float((3*hist_inj[3] + hist_inj[2] - hist_inj[1] - 3*hist_inj[0]) / 10.0)
                row_dict["crashes_delta"] = float(row_dict["crash_count"] - hist_c[-1])
            else:
                row_dict["crashes_lag1"] = float(row_dict["crash_count"])
                row_dict["injuries_lag1"] = float(row_dict["injuries_total"])
                row_dict["crashes_rolling4w_avg"] = float(row_dict["crash_count"])
                row_dict["injuries_rolling4w_avg"] = float(row_dict["injuries_total"])
                row_dict["crashes_trend4w_slope"] = 0.0
                row_dict["injuries_trend4w_slope"] = 0.0
                row_dict["crashes_delta"] = 0.0

            records.append(row_dict)

    df_out = pd.DataFrame(records)

    # Save to SQLite table 'police_crash_4w_temporal'
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df_out.to_sql("police_crash_4w_temporal", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    df_out.to_csv(CRASH_PROCESSED_CSV, index=False)
    print(f"[+] Successfully converted Crash Dataset into 4-Week Temporal Panel ({len(df_out):,} beat-weeks × {len(df_out.columns)} cols) in SQLite & CSV.")
    return df_out


# ==============================================================================
# 3. CLI EXECUTION & VALIDATION
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" POLICE CRASH DATASET ADAPTER & 4-WEEK TEMPORAL INGESTION ".center(80, "="))
    print("=" * 80 + "\n")
    
    # 1. Generate realistic dataset with exact 18 factors
    raw_df = generate_benchmark_police_crash_dataset(num_records=4000)
    raw_csv = os.path.join(DATA_DIR, "chicago_police_crashes_raw.csv")
    raw_df.to_csv(raw_csv, index=False)
    print(f"[+] Generated benchmark crash dataset ({len(raw_df):,} rows × {len(raw_df.columns)} columns) matching exact schema:")
    print("    Columns:", list(raw_df.columns))

    # 2. Transform to 4-Week Panel
    panel_df = transform_crash_records_to_4week_panel(raw_df)
    print(f"\n[+] 4-Week Panel Output Sample (Columns: {len(panel_df.columns)}):")
    print(panel_df[["beat_id", "week", "crashes_rolling4w_avg", "crashes_trend4w_slope", "dominant_weather", "incident_occurred"]].head(10).to_string(index=False))
    print("\n" + "=" * 80 + "\n")
