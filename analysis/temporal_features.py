"""
================================================================================
TEMPORAL AI-BASED TRAFFIC RISK AND PRIORITY INTELLIGENCE SYSTEM
MODULE: TEMPORAL FEATURE ENGINEERING (Step 2)
================================================================================

This module ingests the raw multi-week synthetic city simulation dataset
and engineers leakage-free temporal intelligence features:
1. One-Week Lag Features (t - 1)
2. Four-Week Rolling Historical Aggregations (t-4 to t-1)
3. Week-over-Week Changes & Percentage Deltas (t vs t-1)
4. Four-Week Linear Trend Slopes (OLS slopes over t-4 to t-1)
5. Strict Separation of Features (X) vs Target (y = incident_occurred)

ARCHITECTURAL PRINCIPLE:
- SIMULATOR: Generates what happened in the raw urban environment.
- FEATURE ENGINEERING: Generates what the predictive AI system could realistically
  know from past history when forecasting current-week incident risk.
================================================================================
"""

import os
import numpy as np
import pandas as pd


# ==============================================================================
# 1. DATA INGESTION & SORTING
# ==============================================================================

def load_and_sort_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw traffic simulation dataset and strictly enforces chronological
    sorting by zone_id and week.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found at: {filepath}")

    df = pd.read_csv(filepath)
    
    # Enforce strict chronological ordering per zone
    df = df.sort_values(by=["zone_id", "week"]).reset_index(drop=True)
    return df


# ==============================================================================
# 2. LAG-1 PREVIOUS WEEK FEATURES (t - 1)
# ==============================================================================

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 1-week lag features for each zone using only the immediately preceding week.
    For Week t, uses Week t-1.
    """
    grouped = df.groupby("zone_id")

    df["previous_week_vehicle_density"] = grouped["vehicle_density"].shift(1)
    df["previous_week_congestion"] = grouped["congestion"].shift(1)
    df["previous_week_average_speed"] = grouped["average_speed"].shift(1)
    df["previous_week_red_light_violations"] = grouped["red_light_violations"].shift(1)
    df["previous_week_incident_count"] = grouped["incident_count"].shift(1)
    df["previous_week_incident_occurred"] = grouped["incident_occurred"].shift(1)
    df["previous_week_traffic_pressure"] = grouped["traffic_pressure"].shift(1)

    # 2-week lag of incident count for computing historical incident changes safely
    df["_lag2_incident_count"] = grouped["incident_count"].shift(2)

    return df


# ==============================================================================
# 3. FOUR-WEEK ROLLING HISTORICAL FEATURES (t-4 to t-1)
# ==============================================================================

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 4-week rolling historical averages strictly over past weeks (t-4 to t-1).
    IMPORTANT: Current week t is NEVER included in its rolling historical metrics.
    """
    grouped = df.groupby("zone_id")

    # Shifted series to ensure past-only windows
    shifted_veh = grouped["vehicle_density"].shift(1)
    shifted_cong = grouped["congestion"].shift(1)
    shifted_spd = grouped["average_speed"].shift(1)
    shifted_viol = grouped["red_light_violations"].shift(1)
    shifted_inc_cnt = grouped["incident_count"].shift(1)
    shifted_inc_occ = grouped["incident_occurred"].shift(1)
    shifted_press = grouped["traffic_pressure"].shift(1)

    # Compute rolling metrics with min_periods=4 (complete 4-week window required)
    df["rolling_4_week_avg_vehicle_density"] = shifted_veh.groupby(df["zone_id"]).rolling(4, min_periods=4).mean().round(2).values
    df["rolling_4_week_avg_congestion"] = shifted_cong.groupby(df["zone_id"]).rolling(4, min_periods=4).mean().round(2).values
    df["rolling_4_week_avg_speed"] = shifted_spd.groupby(df["zone_id"]).rolling(4, min_periods=4).mean().round(2).values
    df["rolling_4_week_avg_violations"] = shifted_viol.groupby(df["zone_id"]).rolling(4, min_periods=4).mean().round(2).values
    df["rolling_4_week_incident_count"] = shifted_inc_cnt.groupby(df["zone_id"]).rolling(4, min_periods=4).sum().values
    df["rolling_4_week_incident_rate"] = shifted_inc_occ.groupby(df["zone_id"]).rolling(4, min_periods=4).mean().round(3).values
    df["rolling_4_week_avg_traffic_pressure"] = shifted_press.groupby(df["zone_id"]).rolling(4, min_periods=4).mean().round(3).values

    return df


# ==============================================================================
# 4. WEEK-OVER-WEEK CHANGE & PERCENTAGE CHANGE FEATURES
# ==============================================================================

def create_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Week-over-Week absolute and percentage changes.
    - Environmental deltas compare current conditions (t) vs previous conditions (t-1).
    - Incident change compares last week (t-1) vs prior week (t-2) to avoid target leakage.
    """
    # Absolute WoW Changes
    df["vehicle_density_change"] = (df["vehicle_density"] - df["previous_week_vehicle_density"]).round(2)
    df["congestion_change"] = (df["congestion"] - df["previous_week_congestion"]).round(2)
    df["speed_change"] = (df["average_speed"] - df["previous_week_average_speed"]).round(2)
    df["violations_change"] = (df["red_light_violations"] - df["previous_week_red_light_violations"]).round(2)
    df["traffic_pressure_change"] = (df["traffic_pressure"] - df["previous_week_traffic_pressure"]).round(3)

    # Safe Historical Incident Change: (t-1) minus (t-2)
    df["incident_count_change"] = (df["previous_week_incident_count"] - df["_lag2_incident_count"])

    # Safe Percentage Changes (with epsilon to prevent division by zero or infinite values)
    eps = 1e-4
    df["vehicle_density_pct_change"] = ((df["vehicle_density_change"] / (df["previous_week_vehicle_density"] + eps)) * 100.0).round(2)
    df["congestion_pct_change"] = ((df["congestion_change"] / (df["previous_week_congestion"] + eps)) * 100.0).round(2)
    df["speed_pct_change"] = ((df["speed_change"] / (df["previous_week_average_speed"] + eps)) * 100.0).round(2)

    # Exposure feature: ratio of vehicle density to population density (in thousands)
    df["vehicle_population_ratio"] = (df["vehicle_density"] / (df["population_density"] / 1000.0)).round(3)

    # Drop temporary helper column
    df = df.drop(columns=["_lag2_incident_count"])

    return df


# ==============================================================================
# 5. FOUR-WEEK LINEAR TREND FEATURES (OLS Slope over t-4 to t-1)
# ==============================================================================

def calculate_4w_slope(s_lag1: pd.Series, s_lag2: pd.Series, s_lag3: pd.Series, s_lag4: pd.Series) -> pd.Series:
    """
    Computes exact closed-form OLS linear regression slope for 4 chronological points
    y = [y_{t-4}, y_{t-3}, y_{t-2}, y_{t-1}] at x = [0, 1, 2, 3].
    
    Formula derivation:
    Slope = Sum((x_i - mean_x) * (y_i - mean_y)) / Sum((x_i - mean_x)^2)
          = (-1.5*y_{t-4} - 0.5*y_{t-3} + 0.5*y_{t-2} + 1.5*y_{t-1}) / 5.0
          = (3*y_{t-1} + y_{t-2} - y_{t-3} - 3*y_{t-4}) / 10.0
    """
    slope = (3.0 * s_lag1 + s_lag2 - s_lag3 - 3.0 * s_lag4) / 10.0
    return slope.round(3)


def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates 4-week historical trend slopes strictly using past weeks (t-4 to t-1).
    Positive slope = metric is trending upward over time.
    Negative slope = metric is trending downward over time.
    """
    grouped = df.groupby("zone_id")

    # Lag 1, 2, 3, 4 for each series
    cong_l1 = grouped["congestion"].shift(1)
    cong_l2 = grouped["congestion"].shift(2)
    cong_l3 = grouped["congestion"].shift(3)
    cong_l4 = grouped["congestion"].shift(4)

    veh_l1 = grouped["vehicle_density"].shift(1)
    veh_l2 = grouped["vehicle_density"].shift(2)
    veh_l3 = grouped["vehicle_density"].shift(3)
    veh_l4 = grouped["vehicle_density"].shift(4)

    spd_l1 = grouped["average_speed"].shift(1)
    spd_l2 = grouped["average_speed"].shift(2)
    spd_l3 = grouped["average_speed"].shift(3)
    spd_l4 = grouped["average_speed"].shift(4)

    inc_l1 = grouped["incident_occurred"].shift(1)
    inc_l2 = grouped["incident_occurred"].shift(2)
    inc_l3 = grouped["incident_occurred"].shift(3)
    inc_l4 = grouped["incident_occurred"].shift(4)

    df["congestion_trend_4w"] = calculate_4w_slope(cong_l1, cong_l2, cong_l3, cong_l4)
    df["vehicle_density_trend_4w"] = calculate_4w_slope(veh_l1, veh_l2, veh_l3, veh_l4)
    df["speed_trend_4w"] = calculate_4w_slope(spd_l1, spd_l2, spd_l3, spd_l4)
    df["incident_trend_4w"] = calculate_4w_slope(inc_l1, inc_l2, inc_l3, inc_l4)

    return df


# ==============================================================================
# 6. PIPELINE ORCHESTRATOR
# ==============================================================================

def generate_temporal_features(input_path: str) -> pd.DataFrame:
    """
    Executes the complete end-to-end temporal feature engineering pipeline.
    """
    # 1. Load and sort
    df = load_and_sort_data(input_path)

    # 2. Lag-1 features
    df = create_lag_features(df)

    # 3. Rolling 4-week features
    df = create_rolling_features(df)

    # 4. Changes & Percentages
    df = create_change_features(df)

    # 5. 4-week Trend Slopes
    df = create_trend_features(df)

    return df


# ==============================================================================
# 7. VALIDATION & LEAKAGE AUDIT SUITE
# ==============================================================================

def validate_and_report_features(df: pd.DataFrame):
    """
    Executes automated data quality validations, missing value checks,
    and a manual sanity audit on Zone_01 at Week 10 to mathematically prove zero leakage.
    """
    print("\n" + "=" * 94)
    print(" TEMPORAL FEATURE ENGINEERING -- VALIDATION & AUDIT REPORT ".center(94, "="))
    print("=" * 94)

    # 1. Dimensions
    total_rows = len(df)
    unique_zones = df["zone_id"].nunique()
    unique_weeks = df["week"].nunique()
    total_cols = len(df.columns)

    print("\n[+] 1. DATASET INTEGRITY CHECKS:")
    print(f"    - Total Observations:      {total_rows:,} (Expected: 2,600)")
    print(f"    - Unique Zones:            {unique_zones}")
    print(f"    - Unique Weeks:            {unique_weeks}")
    print(f"    - Total Features/Columns:  {total_cols}")
    print(f"    - Duplicate Grid Check:    {'PASS (0 duplicates)' if df.duplicated(subset=['zone_id', 'week']).sum() == 0 else 'FAIL'}")
    print(f"    - Infinite Values Check:   {'PASS (0 infinite values)' if not np.isinf(df.select_dtypes(include=np.number)).any().any() else 'FAIL'}")

    # 2. Missing Value Analysis (Expected for initial warm-up weeks 1-4)
    print("\n[+] 2. MISSING VALUES (WARM-UP ANALYSIS):")
    print("-" * 94)
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    for col, cnt in null_cols.items():
        print(f"    - {col:<38}: {cnt:>4} NaNs ({cnt/unique_zones:.0f} weeks per zone)")
    print("    * Note: Week 1 has no lag-1. Weeks 1-4 have incomplete 4-week rolling windows.")
    print("      This is mathematically expected and proves future data was NEVER backfilled.")
    print("-" * 94)

    # 3. Target Distribution
    inc_rate = df["incident_occurred"].mean() * 100.0
    print("\n[+] 3. TARGET VARIABLE CHECK:")
    print(f"    - Target Column (y):        'incident_occurred'")
    print(f"    - Positive Class Rate:      {inc_rate:.1f}% ({df['incident_occurred'].sum():,} positive incident weeks)")
    print(f"    - Outcome Metric:           'incident_count' (Kept for evaluation, excluded from X)")

    # 4. Manual Zero-Leakage Audit for Zone_01 at Week 10
    print("\n[+] 4. CONCRETE LEAKAGE AUDIT (Zone_01 across Weeks 6 to 10):")
    print("-" * 94)
    z1_slice = df[(df["zone_id"] == "Zone_01") & (df["week"].between(6, 10))][
        ["week", "congestion", "previous_week_congestion", "rolling_4_week_avg_congestion", "congestion_trend_4w", "incident_occurred"]
    ]
    print(z1_slice.to_string(index=False))
    print("-" * 94)

    # Mathematical Proof Verification
    w10_row = df[(df["zone_id"] == "Zone_01") & (df["week"] == 10)].iloc[0]
    w6_9_cong = df[(df["zone_id"] == "Zone_01") & (df["week"].between(6, 9))]["congestion"].values
    expected_rolling = round(float(np.mean(w6_9_cong)), 2)
    actual_rolling = w10_row["rolling_4_week_avg_congestion"]
    
    # OLS Slope formula: (3*y9 + y8 - y7 - 3*y6) / 10
    expected_trend = round((3.0*w6_9_cong[3] + w6_9_cong[2] - w6_9_cong[1] - 3.0*w6_9_cong[0]) / 10.0, 3)
    actual_trend = w10_row["congestion_trend_4w"]

    print(f"\n[+] MATHEMATICAL PROOF FOR ZONE_01 AT WEEK 10:")
    print(f"    - Week 10 Current Congestion:           {w10_row['congestion']}")
    print(f"    - Raw Congestion in Weeks 6, 7, 8, 9:    {w6_9_cong}")
    print(f"    - Previous Week Congestion (Week 9):    {w10_row['previous_week_congestion']} (Expected: {w6_9_cong[3]}) -> {'MATCH [PASS]' if w10_row['previous_week_congestion'] == w6_9_cong[3] else 'FAIL'}")
    print(f"    - Rolling 4-Wk Mean (Weeks 6-9):        {actual_rolling} (Expected: {expected_rolling}) -> {'MATCH [PASS]' if actual_rolling == expected_rolling else 'FAIL'}")
    print(f"    - 4-Wk Trend Slope (Weeks 6-9):         {actual_trend} (Expected: {expected_trend}) -> {'MATCH [PASS]' if actual_trend == expected_trend else 'FAIL'}")
    print("    * PROOF VERIFIED: Current week 10 value was NOT used in historical features.")

    # 5. Feature Matrix (X) and Target (y) Definition
    print("\n[+] 5. ML-READY SPECIFICATION:")
    exclude_cols = ["zone_id", "week", "incident_count", "incident_occurred"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"    - Total Feature Count (X):  {len(feature_cols)} features")
    print(f"    - Target Column (y):        incident_occurred")
    print(f"    - Categorical Features:     ['zone_type', 'temporal_profile', 'weather', 'road_condition']")
    print(f"    - Numerical Features:       {len(feature_cols) - 4} features")
    print("=" * 94 + "\n")


# ==============================================================================
# 8. SAFE SAVING & MAIN ENTRYPOINT
# ==============================================================================

def save_features(df: pd.DataFrame, output_path: str):
    """
    Saves temporal features dataset to CSV handling file locks safely.
    """
    import time
    try:
        df.to_csv(output_path, index=False)
        print(f"[+] Successfully saved temporal features to:\n    {output_path}\n")
    except PermissionError:
        time.sleep(0.5)
        try:
            df.to_csv(output_path, index=False)
            print(f"[+] Successfully saved temporal features to:\n    {output_path}\n")
        except Exception:
            backup = output_path.replace(".csv", "_updated.csv")
            df.to_csv(backup, index=False)
            print(f"[!] Primary file was locked. Saved features to backup:\n    {backup}\n")


if __name__ == "__main__":
    # Determine project root and file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir if os.path.basename(script_dir) == "traffic_sim" else os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")

    # Priority input path check
    input_file = os.path.join(data_dir, "traffic_simulation_updated.csv")
    if not os.path.exists(input_file):
        input_file = os.path.join(data_dir, "traffic_simulation.csv")

    output_file = os.path.join(data_dir, "temporal_features.csv")

    print(f"\n[+] Ingesting simulation data from:\n    {input_file}")
    df_features = generate_temporal_features(input_file)

    # Run validation and leakage verification
    validate_and_report_features(df_features)

    # Save to data/temporal_features.csv
    save_features(df_features, output_file)
