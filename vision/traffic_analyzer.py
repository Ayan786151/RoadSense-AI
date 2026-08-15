import argparse
import pandas as pd
from pathlib import Path


# ============================================================
# ROAD SENSE AI - TRAFFIC FEATURE ANALYZER
# ============================================================

DEFAULT_INPUT_CSV = "data/vision_traffic_metrics.csv"
DEFAULT_OUTPUT_CSV = "data/traffic_features.csv"


def resolve_paths(session_id: str = None):
    """
    Resolves input and output CSV paths based on session_id.
    """
    if session_id:
        session_dir = Path("data") / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        input_csv = str(session_dir / "vision_traffic_metrics.csv")
        output_csv = str(session_dir / "traffic_features.csv")
        return input_csv, output_csv, str(session_dir)
    return DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_CSV, "data"


def load_data(input_csv: str = DEFAULT_INPUT_CSV):
    """Load the raw computer-vision metrics."""

    if not Path(input_csv).exists():
        raise FileNotFoundError(
            f"Input file not found: {input_csv}"
        )

    df = pd.read_csv(input_csv)

    required_columns = [
        "timestamp_seconds",
        "vehicle_count",
        "cars",
        "motorcycles",
        "buses",
        "trucks"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# VEHICLE DENSITY
# ============================================================

def calculate_vehicle_density(df):
    """
    Calculate a normalized vehicle-density indicator.

    This is NOT vehicles per km².

    It represents how heavily occupied the observed
    camera scene is relative to the busiest observation
    in this video.
    """

    maximum_count = df["vehicle_count"].max()

    if maximum_count == 0:
        return 0.0

    density = (
        df["vehicle_count"] /
        maximum_count
    ) * 100

    return density.clip(0, 100)


# ============================================================
# CONGESTION INDICATOR
# ============================================================

def calculate_congestion(df):
    """
    Calculate a preliminary congestion indicator.

    For this first vision prototype, congestion is based
    on vehicle accumulation in the camera's observed road
    region.

    This is a prototype indicator, NOT a final traffic
    engineering congestion measurement.
    """

    maximum_count = df["vehicle_count"].max()

    if maximum_count == 0:
        return pd.Series(
            0.0,
            index=df.index
        )

    congestion = (
        df["vehicle_count"] /
        maximum_count
    ) * 100

    return congestion.clip(0, 100)


# ============================================================
# VEHICLE COMPOSITION
# ============================================================

def calculate_vehicle_percentages(df):

    total = df["vehicle_count"].replace(
        0,
        1
    )

    df["car_percentage"] = (
        df["cars"] / total
    ) * 100

    df["motorcycle_percentage"] = (
        df["motorcycles"] / total
    ) * 100

    df["bus_percentage"] = (
        df["buses"] / total
    ) * 100

    df["truck_percentage"] = (
        df["trucks"] / total
    ) * 100

    return df


# ============================================================
# TRAFFIC STATE
# ============================================================

def classify_traffic_state(congestion):

    if congestion < 25:
        return "LOW"

    elif congestion < 50:
        return "MODERATE"

    elif congestion < 75:
        return "HIGH"

    else:
        return "SEVERE"


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_traffic(session_id: str = None, video_path: str = None):

    input_csv, output_csv, session_dir = resolve_paths(session_id)

    # If video path is passed and input does not exist, run vehicle detection first
    if video_path and not Path(input_csv).exists():
        from vision.vehicle_detector import process_video
        print(f"[+] Input metrics missing. Automatically running vehicle detection on: {video_path}")
        process_video(video_path=video_path, session_id=session_id)

    print("=" * 70)
    print("ROAD SENSE AI - TRAFFIC FEATURE ANALYZER")
    print("=" * 70)
    if session_id:
        print(f"Active Session : {session_id}")
    print(f"Reading from   : {input_csv}")
    print(f"Writing to     : {output_csv}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data(input_csv)

    print(f"\nInput records: {len(df)}")

    # --------------------------------------------------------
    # Vehicle density
    # --------------------------------------------------------

    df["vehicle_density"] = (
        calculate_vehicle_density(df)
    )

    # --------------------------------------------------------
    # Congestion
    # --------------------------------------------------------

    df["congestion"] = (
        calculate_congestion(df)
    )

    # --------------------------------------------------------
    # Vehicle composition
    # --------------------------------------------------------

    df = calculate_vehicle_percentages(df)

    # --------------------------------------------------------
    # Traffic state
    # --------------------------------------------------------

    df["traffic_state"] = (
        df["congestion"]
        .apply(classify_traffic_state)
    )

    # --------------------------------------------------------
    # Preliminary feature quality checks
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE VALIDATION")
    print("=" * 70)

    print(
        f"\nVehicle density range: "
        f"{df['vehicle_density'].min():.2f} - "
        f"{df['vehicle_density'].max():.2f}"
    )

    print(
        f"Congestion range: "
        f"{df['congestion'].min():.2f} - "
        f"{df['congestion'].max():.2f}"
    )

    print(
        "\nNull values:"
    )

    print(
        df[
            [
                "vehicle_density",
                "congestion"
            ]
        ]
        .isnull()
        .sum()
    )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAFFIC FEATURE PREVIEW")
    print("=" * 70)

    preview_columns = [
        "timestamp_seconds",
        "vehicle_count",
        "cars",
        "motorcycles",
        "buses",
        "trucks",
        "vehicle_density",
        "congestion",
        "traffic_state"
    ]

    print(
        df[
            preview_columns
        ]
        .head(15)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Traffic-state distribution
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAFFIC STATE DISTRIBUTION")
    print("=" * 70)

    print(
        df["traffic_state"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    Path(
        output_csv
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nTraffic features saved to:"
        f"\n{output_csv}"
    )

    return df


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RoadSense AI - Traffic Feature Analyzer")
    parser.add_argument("--session", type=str, default=None, help="Session identifier (e.g. session_002)")
    parser.add_argument("--video", type=str, default=None, help="Optional video path if video processing is needed")

    args = parser.parse_args()

    analyze_traffic(session_id=args.session, video_path=args.video)


if __name__ == "__main__":
    main()