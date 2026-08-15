import pandas as pd
import numpy as np


# ============================================================
# ROAD SENSE AI - PRIORITY ENGINE
# ============================================================

INPUT_FILE = "data/risk_predictions.csv"
LOCATION_FILE = "data/location_mapping.csv"

COMPONENT_OUTPUT = "data/priority_components.csv"
FINAL_OUTPUT = "data/final_priority_results.csv"
HEATMAP_OUTPUT = "data/heatmap_data.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    """Load ML prediction dataset."""

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("ROAD SENSE AI - PRIORITY ENGINE")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    return df


# ============================================================
# 2. INPUT VALIDATION
# ============================================================

def validate_input(df):

    required_columns = [
        "zone_id",
        "week",
        "zone_type",
        "population_density",
        "vehicle_density",
        "congestion",
        "average_speed",
        "congestion_trend_4w",
        "predicted_risk_probability"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("\nInput validation: PASS")

    print(
        f"Unique zones: {df['zone_id'].nunique()}"
    )

    print(
        f"Weeks: {sorted(df['week'].unique())}"
    )

    duplicates = df.duplicated(
        subset=["zone_id", "week"]
    ).sum()

    if duplicates > 0:
        raise ValueError(
            f"Found {duplicates} duplicate zone-week records."
        )

    print("Duplicate zone-week check: PASS")

    null_count = df[required_columns].isnull().sum().sum()

    if null_count > 0:
        raise ValueError(
            f"Found {null_count} missing values."
        )

    print("Null value check: PASS")


# ============================================================
# 3. NORMALIZATION
# ============================================================

def min_max_normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            50.0,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    ) * 100


# ============================================================
# 4. COMPONENT SCORES
# ============================================================

def calculate_component_scores(df):

    # ML risk
    df["risk_score"] = (
        df["predicted_risk_probability"] * 100
    )

    # Population exposure
    df["population_exposure_score"] = (
        min_max_normalize(
            df["population_density"]
        )
    )

    # Vehicle exposure
    df["vehicle_exposure_score"] = (
        min_max_normalize(
            df["vehicle_density"]
        )
    )

    return df


# ============================================================
# 5. TEMPORAL TREND SCORE
# ============================================================

def calculate_temporal_score(df):

    trend = df["congestion_trend_4w"]

    # Observed range in the current simulation
    trend_min = -13.791
    trend_max = 11.964

    df["temporal_trend_score"] = (
        (trend - trend_min)
        / (trend_max - trend_min)
    ) * 100

    df["temporal_trend_score"] = (
        df["temporal_trend_score"]
        .clip(0, 100)
    )

    return df


# ============================================================
# 6. FINAL PRIORITY SCORE
# ============================================================

def calculate_priority_score(df):

    # --------------------------------------------------------
    # WEIGHTS
    # --------------------------------------------------------

    RISK_WEIGHT = 0.40
    POPULATION_WEIGHT = 0.25
    VEHICLE_WEIGHT = 0.20
    TEMPORAL_WEIGHT = 0.15

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    df["priority_score"] = (
        (RISK_WEIGHT * df["risk_score"])
        +
        (POPULATION_WEIGHT *
         df["population_exposure_score"])
        +
        (VEHICLE_WEIGHT *
         df["vehicle_exposure_score"])
        +
        (TEMPORAL_WEIGHT *
         df["temporal_trend_score"])
    )

    # Safety boundary
    df["priority_score"] = (
        df["priority_score"]
        .clip(0, 100)
    )

    return df


# ============================================================
# 7. PRIORITY RANK
# ============================================================

def calculate_priority_rank(df):

    # Higher score = higher priority
    df["priority_rank"] = (
        df["priority_score"]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    return df


# ============================================================
# 8. PRIORITY LEVEL
# ============================================================

def assign_priority_level(df):

    def classify(score):

        if score >= 75:
            return "CRITICAL"

        elif score >= 55:
            return "HIGH"

        elif score >= 35:
            return "MODERATE"

        else:
            return "LOW"

    df["priority_level"] = (
        df["priority_score"]
        .apply(classify)
    )

    return df


# ============================================================
# 9. LOCATION MAPPING
# ============================================================

def attach_locations(df):

    locations = pd.read_csv(LOCATION_FILE)

    required_location_columns = [
        "zone_id",
        "location_name",
        "city",
        "latitude",
        "longitude"
    ]

    missing = [
        column
        for column in required_location_columns
        if column not in locations.columns
    ]

    if missing:
        raise ValueError(
            f"Missing location columns: {missing}"
        )

    # Prevent accidental duplicate mappings
    if locations["zone_id"].duplicated().any():
        raise ValueError(
            "Duplicate zone_id found in location_mapping.csv"
        )

    df = df.merge(
        locations[
            required_location_columns
        ],
        on="zone_id",
        how="left",
        validate="many_to_one"
    )

    # Make sure every zone received a location
    missing_locations = df["location_name"].isna().sum()

    if missing_locations > 0:
        raise ValueError(
            f"{missing_locations} records have no location mapping."
        )

    print(
        f"\nLocation mapping: PASS "
        f"({df['city'].nunique()} cities)"
    )

    return df


# ============================================================
# 10. LATEST WEEK HEATMAP DATA
# ============================================================

def create_heatmap_data(df):

    latest_week = df["week"].max()

    heatmap_df = (
        df[df["week"] == latest_week]
        .copy()
        .sort_values(
            "priority_score",
            ascending=False
        )
    )

    heatmap_columns = [
        "zone_id",
        "location_name",
        "city",
        "latitude",
        "longitude",
        "week",
        "priority_score",
        "priority_rank",
        "priority_level",
        "risk_score",
        "population_exposure_score",
        "vehicle_exposure_score",
        "temporal_trend_score"
    ]

    heatmap_df = heatmap_df[
        heatmap_columns
    ]

    heatmap_df.to_csv(
        HEATMAP_OUTPUT,
        index=False
    )

    print(
        f"\nHeatmap data saved to:"
        f"\n{HEATMAP_OUTPUT}"
    )

    print(
        f"Heatmap week: {latest_week}"
    )

    return heatmap_df


# ============================================================
# 11. VALIDATE FINAL SCORE
# ============================================================

def validate_priority_score(df):

    minimum = df["priority_score"].min()
    maximum = df["priority_score"].max()

    valid = (
        minimum >= 0
        and maximum <= 100
        and df["priority_score"].notna().all()
    )

    print("\n" + "=" * 70)
    print("FINAL PRIORITY SCORE VALIDATION")
    print("=" * 70)

    print(
        f"Minimum priority score: {minimum:.2f}"
    )

    print(
        f"Maximum priority score: {maximum:.2f}"
    )

    print(
        f"Status: {'PASS' if valid else 'FAIL'}"
    )

    if not valid:
        raise ValueError(
            "Priority score validation failed."
        )


# ============================================================
# 12. SHOW TOP PRIORITY LOCATIONS
# ============================================================

def show_top_priorities(df, number=15):

    print("\n" + "=" * 70)
    print("TOP PRIORITY LOCATIONS")
    print("=" * 70)

    latest_week = df["week"].max()

    latest = (
        df[df["week"] == latest_week]
        .sort_values(
            "priority_score",
            ascending=False
        )
        .head(number)
    )

    columns = [
        "priority_rank",
        "location_name",
        "city",
        "priority_score",
        "priority_level",
        "risk_score",
        "population_exposure_score",
        "vehicle_exposure_score",
        "temporal_trend_score"
    ]

    print(
        latest[columns]
        .round(2)
        .to_string(index=False)
    )


# ============================================================
# 13. PRIORITY DISTRIBUTION
# ============================================================

def show_priority_distribution(df):

    print("\n" + "=" * 70)
    print("PRIORITY DISTRIBUTION - LATEST WEEK")
    print("=" * 70)

    latest_week = df["week"].max()

    latest = df[
        df["week"] == latest_week
    ]

    distribution = (
        latest["priority_level"]
        .value_counts()
    )

    levels = [
        "CRITICAL",
        "HIGH",
        "MODERATE",
        "LOW"
    ]

    for level in levels:

        count = distribution.get(
            level,
            0
        )

        print(
            f"{level:<10}: {count}"
        )


# ============================================================
# 14. SAVE FINAL RESULTS
# ============================================================

def save_results(df):

    df.to_csv(
        FINAL_OUTPUT,
        index=False
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"Component scores:"
        f"\n  {COMPONENT_OUTPUT}"
    )

    print(
        f"Final priority results:"
        f"\n  {FINAL_OUTPUT}"
    )

    print(
        f"Heatmap data:"
        f"\n  {HEATMAP_OUTPUT}"
    )


# ============================================================
# 15. MAIN PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_input(df)

    # --------------------------------------------------------
    # COMPONENT SCORES
    # --------------------------------------------------------

    df = calculate_component_scores(df)

    df = calculate_temporal_score(df)

    # --------------------------------------------------------
    # FINAL PRIORITY
    # --------------------------------------------------------

    df = calculate_priority_score(df)

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    df = calculate_priority_rank(df)

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    df = assign_priority_level(df)

    # --------------------------------------------------------
    # LOCATION INFORMATION
    # --------------------------------------------------------

    df = attach_locations(df)

    # --------------------------------------------------------
    # VALIDATE FINAL SCORE
    # --------------------------------------------------------

    validate_priority_score(df)

    # --------------------------------------------------------
    # SAVE COMPLETE DATASET
    # --------------------------------------------------------

    df.to_csv(
        COMPONENT_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # CREATE HEATMAP DATA
    # --------------------------------------------------------

    create_heatmap_data(df)

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    show_top_priorities(df)

    show_priority_distribution(df)

    # --------------------------------------------------------
    # SAVE FINAL DATASET
    # --------------------------------------------------------

    save_results(df)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()