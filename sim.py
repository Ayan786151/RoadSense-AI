import os
import time
import numpy as np
import pandas as pd

NUM_ZONES = 50
NUM_WEEKS = 52
RANDOM_SEED = 42

POPULATION_MIN = 1200
POPULATION_MAX = 16000
ROAD_CAPACITY_MIN = 40
ROAD_CAPACITY_MAX = 100


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def create_city_zones(num_zones: int = NUM_ZONES, seed: int = RANDOM_SEED) -> list:
    np.random.seed(seed)
    zones = []

    archetypes = [
        "Residential",
        "Commercial_Downtown",
        "University_District",
        "Industrial_Corridor",
        "Highway_Junction",
        "Suburban_LowDensity"
    ]
    archetype_probs = [0.30, 0.25, 0.15, 0.10, 0.10, 0.10]

    temporal_profiles = [
        "STABLE",
        "GRADUAL_DETERIORATION",
        "GRADUAL_IMPROVEMENT",
        "PERIODIC_SPIKE",
        "EVENT_SENSITIVE",
        "RECOVERY_AFTER_DISRUPTION"
    ]
    profile_probs = [0.20, 0.20, 0.15, 0.15, 0.15, 0.15]

    for zone_num in range(1, num_zones + 1):
        zone_id = f"Zone_{zone_num:02d}"
        archetype = np.random.choice(archetypes, p=archetype_probs)
        temporal_profile = np.random.choice(temporal_profiles, p=profile_probs)

        if archetype == "Residential":
            pop = np.random.randint(8000, POPULATION_MAX)
            activity_pull = np.random.uniform(0.70, 0.85)
        elif archetype == "Commercial_Downtown":
            pop = np.random.randint(3500, 9500)
            activity_pull = np.random.uniform(1.20, 1.55)
        elif archetype == "University_District":
            pop = np.random.randint(6000, 12000)
            activity_pull = np.random.uniform(1.00, 1.30)
        elif archetype == "Industrial_Corridor":
            pop = np.random.randint(1800, 4500)
            activity_pull = np.random.uniform(1.10, 1.35)
        elif archetype == "Highway_Junction":
            pop = np.random.randint(POPULATION_MIN, 3000)
            activity_pull = np.random.uniform(1.40, 1.80)
        else:
            pop = np.random.randint(POPULATION_MIN, 4500)
            activity_pull = np.random.uniform(0.50, 0.70)

        base_capacity = int(np.random.randint(ROAD_CAPACITY_MIN, ROAD_CAPACITY_MAX))

        zones.append({
            "zone_id": zone_id,
            "zone_type": archetype,
            "temporal_profile": temporal_profile,
            "population_density": int(pop),
            "road_capacity": base_capacity,
            "activity_pull": round(activity_pull, 2)
        })

    return zones


def get_temporal_environment(week: int, zone: dict) -> dict:
    archetype = zone["zone_type"]
    profile = zone["temporal_profile"]

    weather = "Normal"
    road_condition = "Good"

    if 31 <= week <= 36:
        roll = np.random.rand()
        if roll < 0.38:
            weather = "Heavy Rain"
            road_condition = "Poor"
        elif roll < 0.78:
            weather = "Light Rain"
            road_condition = "Moderate"
    elif 48 <= week <= 52:
        if np.random.rand() < 0.40:
            weather = "Light Rain"
            road_condition = "Moderate"
    else:
        if np.random.rand() < 0.12:
            weather = "Light Rain"
            road_condition = "Moderate"

    special_event = 0
    event_surge_vehicles = 0.0
    event_capacity_penalty = 1.0

    if (18 <= week <= 19) and (archetype in ["Commercial_Downtown", "University_District"]):
        special_event = 1
        event_surge_vehicles = 18.0
        event_capacity_penalty = 0.85
    elif (38 <= week <= 40) and (archetype in ["Commercial_Downtown", "Highway_Junction"]):
        special_event = 1
        event_surge_vehicles = 22.0
        event_capacity_penalty = 0.80

    trend_drift = 0.0

    if profile == "GRADUAL_DETERIORATION":
        trend_drift = (week / 52.0) * 22.0

    elif profile == "GRADUAL_IMPROVEMENT":
        trend_drift = -(week / 52.0) * 16.0

    elif profile == "PERIODIC_SPIKE":
        if week % 5 == 0:
            trend_drift = 19.0
            special_event = 1
            event_capacity_penalty = 0.90
        else:
            trend_drift = 0.0

    elif profile == "EVENT_SENSITIVE":
        if special_event == 1:
            trend_drift = 12.0
            event_capacity_penalty *= 0.90
        elif week in [8, 24, 42]:
            special_event = 1
            trend_drift = 17.0
            event_capacity_penalty = 0.88

    elif profile == "RECOVERY_AFTER_DISRUPTION":
        if week <= 10:
            trend_drift = 0.0
        elif 11 <= week <= 20:
            trend_drift = ((week - 10) / 10.0) * 18.0
        elif 21 <= week <= 28:
            trend_drift = 20.0
            event_capacity_penalty = 0.82
        elif 29 <= week <= 38:
            progress = (week - 28) / 10.0
            trend_drift = 20.0 - (progress * 22.0)
        else:
            trend_drift = -2.5

    else:
        trend_drift = 0.0

    return {
        "weather": weather,
        "road_condition": road_condition,
        "special_event": special_event,
        "event_surge_vehicles": event_surge_vehicles,
        "event_capacity_penalty": event_capacity_penalty,
        "trend_drift": trend_drift
    }


def calculate_effective_capacity(
    base_capacity: int,
    weather: str,
    road_condition: str,
    event_capacity_penalty: float
) -> float:
    w_factor = {"Normal": 1.00, "Light Rain": 0.92, "Heavy Rain": 0.78}[weather]
    r_factor = {"Good": 1.00, "Moderate": 0.90, "Poor": 0.75}[road_condition]
    effective_cap = base_capacity * w_factor * r_factor * event_capacity_penalty
    return max(15.0, round(effective_cap, 2))


def calculate_vehicle_density(
    population_density: int,
    activity_pull: float,
    event_surge: float,
    trend_drift: float,
    weather: str,
    prev_density: float = None
) -> float:
    pop_score = ((population_density - POPULATION_MIN) / (POPULATION_MAX - POPULATION_MIN)) * 100.0
    base_target = pop_score * 0.65 * activity_pull + event_surge + trend_drift

    if weather == "Heavy Rain":
        base_target *= 0.92

    if prev_density is not None:
        target = 0.65 * prev_density + 0.35 * base_target
    else:
        target = base_target

    weekly_noise = np.random.normal(loc=0.0, scale=4.0)
    return float(np.clip(round(target + weekly_noise, 2), 0.0, 100.0))


def calculate_congestion(
    vehicle_density: float,
    effective_capacity: float,
    weather: str
) -> tuple:
    traffic_pressure = vehicle_density / max(1.0, effective_capacity)
    w_mult = {"Normal": 1.00, "Light Rain": 1.12, "Heavy Rain": 1.28}[weather]
    raw_congestion = 100.0 * sigmoid(3.4 * (traffic_pressure - 0.85)) * w_mult
    noise = np.random.normal(0.0, 2.5)

    congestion = float(np.clip(round(raw_congestion + noise, 2), 0.0, 100.0))
    return round(traffic_pressure, 3), congestion


def calculate_speed(congestion: float, weather: str, road_condition: str) -> float:
    w_penalty = {"Normal": 0.0, "Light Rain": 5.0, "Heavy Rain": 13.0}[weather]
    r_penalty = {"Good": 0.0, "Moderate": 3.0, "Poor": 8.0}[road_condition]

    speed_degradation = (congestion / 100.0) ** 1.35 * 38.0
    raw_speed = 65.0 - speed_degradation - w_penalty - r_penalty + np.random.normal(0.0, 1.8)
    return float(np.clip(round(raw_speed, 2), 10.0, 80.0))


def generate_violations(congestion: float, special_event: int, archetype: str) -> int:
    type_bias = 2.0 if archetype in ["Commercial_Downtown", "Highway_Junction"] else 0.0
    event_bias = 3.0 if special_event == 1 else 0.0
    expected_lambda = 1.0 + (congestion / 100.0) * 9.0 + type_bias + event_bias
    return int(np.random.poisson(lam=max(0.5, expected_lambda)))


def generate_incidents(
    congestion: float,
    violations: int,
    average_speed: float,
    traffic_pressure: float,
    weather: str,
    road_condition: str,
    recent_incident_memory: float
) -> tuple:
    w_risk = {"Normal": 0.0, "Light Rain": 0.30, "Heavy Rain": 0.80}[weather]
    r_risk = {"Good": 0.0, "Moderate": 0.25, "Poor": 0.60}[road_condition]

    risk_logit = (
        -3.40
        + 0.036 * congestion
        + 0.095 * violations
        + 0.040 * max(0.0, 42.0 - average_speed)
        + 0.350 * max(0.0, traffic_pressure - 1.0)
        + 0.250 * recent_incident_memory
        + w_risk
        + r_risk
        + np.random.normal(0.0, 0.18)
    )

    incident_prob = float(np.clip(sigmoid(risk_logit), 0.01, 0.96))
    incident_occurred = int(np.random.rand() < incident_prob)
    incident_count = int(1 + np.random.poisson(lam=incident_prob * 1.3)) if incident_occurred else 0
    return incident_count, incident_occurred


def run_traffic_simulator(
    num_zones: int = NUM_ZONES,
    num_weeks: int = NUM_WEEKS,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    zones = create_city_zones(num_zones=num_zones, seed=seed)
    records = []

    history_tracker = {
        z["zone_id"]: {
            "prev_density": None,
            "past_incidents": []
        } for z in zones
    }

    for week in range(1, num_weeks + 1):
        for zone in zones:
            z_id = zone["zone_id"]
            pop = zone["population_density"]
            base_cap = zone["road_capacity"]
            archetype = zone["zone_type"]
            profile = zone["temporal_profile"]
            activity_pull = zone["activity_pull"]

            env = get_temporal_environment(week, zone)
            weather = env["weather"]
            road_cond = env["road_condition"]
            event = env["special_event"]
            event_surge = env["event_surge_vehicles"]
            cap_penalty = env["event_capacity_penalty"]
            trend_drift = env["trend_drift"]

            eff_capacity = calculate_effective_capacity(
                base_capacity=base_cap,
                weather=weather,
                road_condition=road_cond,
                event_capacity_penalty=cap_penalty
            )

            prev_dens = history_tracker[z_id]["prev_density"]
            vehicle_density = calculate_vehicle_density(
                population_density=pop,
                activity_pull=activity_pull,
                event_surge=event_surge,
                trend_drift=trend_drift,
                weather=weather,
                prev_density=prev_dens
            )
            history_tracker[z_id]["prev_density"] = vehicle_density

            pressure, congestion = calculate_congestion(
                vehicle_density=vehicle_density,
                effective_capacity=eff_capacity,
                weather=weather
            )

            avg_speed = calculate_speed(
                congestion=congestion,
                weather=weather,
                road_condition=road_cond
            )

            violations = generate_violations(
                congestion=congestion,
                special_event=event,
                archetype=archetype
            )

            past_inc = history_tracker[z_id]["past_incidents"]
            recent_memory = float(np.mean(past_inc[-4:])) if len(past_inc) > 0 else 0.0

            inc_count, inc_occurred = generate_incidents(
                congestion=congestion,
                violations=violations,
                average_speed=avg_speed,
                traffic_pressure=pressure,
                weather=weather,
                road_condition=road_cond,
                recent_incident_memory=recent_memory
            )
            history_tracker[z_id]["past_incidents"].append(inc_occurred)

            records.append({
                "zone_id": z_id,
                "week": week,
                "zone_type": archetype,
                "temporal_profile": profile,
                "population_density": pop,
                "road_capacity": base_cap,
                "effective_road_capacity": eff_capacity,
                "vehicle_density": vehicle_density,
                "traffic_pressure": pressure,
                "congestion": congestion,
                "average_speed": avg_speed,
                "red_light_violations": violations,
                "weather": weather,
                "road_condition": road_cond,
                "special_event": event,
                "incident_count": inc_count,
                "incident_occurred": inc_occurred,
            })

    df = pd.DataFrame(records)
    return df


def validate_and_display_data(df: pd.DataFrame):
    print("\n" + "=" * 94)
    print(" TEMPORAL TRAFFIC SIMULATOR -- DEFINITIVE VALIDATION REPORT ".center(94, "="))
    print("=" * 94)

    total_rows = len(df)
    unique_zones = df["zone_id"].nunique()
    unique_weeks = df["week"].nunique()
    expected_rows = unique_zones * unique_weeks

    print("\n[+] 1. DATASET INTEGRITY CHECKS:")
    print(f"    - Total Observations:      {total_rows:,} (Expected: {expected_rows:,})")
    print(f"    - Unique Zones:            {unique_zones}")
    print(f"    - Unique Consecutive Weeks: {unique_weeks}")
    print(f"    - Missing Values Check:    {'PASS (0 nulls)' if df.isnull().sum().sum() == 0 else 'FAIL'}")
    print(f"    - Duplicate Grid Check:    {'PASS (0 duplicates)' if df.duplicated(subset=['zone_id', 'week']).sum() == 0 else 'FAIL'}")

    print("\n[+] 2. TEMPORAL BEHAVIOR PROFILE AUDIT:")
    print("-" * 94)
    profile_counts = df.groupby("zone_id")["temporal_profile"].first().value_counts()
    for prof, cnt in profile_counts.items():
        print(f"    - {prof:<28}: {cnt:>2} zones ({cnt/unique_zones*100:>4.1f}%)")
    print("-" * 94)

    print("\n[+] 3. DEMONSTRATION PROFILES -- TEMPORAL EVOLUTION SNAPSHOTS:")
    sample_weeks = [1, 10, 20, 30, 40, 52]
    disp_cols = ["week", "vehicle_density", "congestion", "average_speed", "red_light_violations", "weather", "special_event", "incident_count"]

    target_profiles = [
        "GRADUAL_DETERIORATION",
        "GRADUAL_IMPROVEMENT",
        "STABLE",
        "PERIODIC_SPIKE",
        "EVENT_SENSITIVE",
        "RECOVERY_AFTER_DISRUPTION"
    ]

    for prof in target_profiles:
        match_zone = df[df["temporal_profile"] == prof]["zone_id"].iloc[0]
        z_type = df[df["zone_id"] == match_zone]["zone_type"].iloc[0]
        print(f"\n>>> Profile: {prof} (Example: {match_zone} | Type: {z_type})")
        print("." * 94)
        subset = df[(df["zone_id"] == match_zone) & (df["week"].isin(sample_weeks))][disp_cols]
        print(subset.to_string(index=False))
        print("." * 94)

    print("\n[+] 4. TEMPORAL CORRELATION AUDIT (Week vs Metrics per Profile):")
    trend_audit = df.groupby("temporal_profile").apply(
        lambda g: pd.Series({
            "Corr(Week, Vehicles)": g[["week", "vehicle_density"]].corr().iloc[0, 1],
            "Corr(Week, Congestion)": g[["week", "congestion"]].corr().iloc[0, 1],
            "Avg Incidents / Wk": g["incident_occurred"].mean()
        }), include_groups=False
    ).round(3)
    print(trend_audit.to_string())

    print("\n[+] 5. CAUSAL CORRELATION MATRIX (Verifying Physics & Exposure Separation):")
    numeric_cols = [
        "population_density", "vehicle_density", "effective_road_capacity",
        "traffic_pressure", "congestion", "average_speed", "red_light_violations",
        "incident_occurred"
    ]
    print(df[numeric_cols].corr().round(3).to_string())

    print("\n[+] 6. DATA QUALITY & BOUNDS ASSERTIONS:")
    v_min, v_max = df["vehicle_density"].min(), df["vehicle_density"].max()
    c_min, c_max = df["congestion"].min(), df["congestion"].max()
    s_min, s_max = df["average_speed"].min(), df["average_speed"].max()
    inc_rate = df["incident_occurred"].mean() * 100.0

    print(f"    - Vehicle Density Range:   {v_min:.1f} to {v_max:.1f} (Valid [0, 100]: {'PASS' if 0 <= v_min and v_max <= 100 else 'FAIL'})")
    print(f"    - Congestion Range:        {c_min:.1f} to {c_max:.1f} (Valid [0, 100]: {'PASS' if 0 <= c_min and c_max <= 100 else 'FAIL'})")
    print(f"    - Speed Range:             {s_min:.1f} to {s_max:.1f} km/h (Valid [10, 80]: {'PASS' if 10 <= s_min and s_max <= 80 else 'FAIL'})")
    print(f"    - Overall Incident Rate:   {inc_rate:.1f}% (Healthy realistic signal for ML)")
    print("=" * 94 + "\n")


def save_simulation_data(df: pd.DataFrame, target_path: str):
    try:
        df.to_csv(target_path, index=False)
        print(f"[+] Successfully saved simulation dataset to:\n    {target_path}\n")
    except PermissionError:
        time.sleep(0.5)
        try:
            df.to_csv(target_path, index=False)
            print(f"[+] Successfully saved simulation dataset to:\n    {target_path}\n")
        except Exception:
            backup = target_path.replace(".csv", "_updated.csv")
            df.to_csv(backup, index=False)
            print(f"[!] Target file was locked. Saved simulation dataset to backup:\n    {backup}\n")


if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[+] Executing Definitive Simulation ({NUM_ZONES} Zones x {NUM_WEEKS} Weeks = {NUM_ZONES * NUM_WEEKS:,} records)...")
    df_sim = run_traffic_simulator(
        num_zones=NUM_ZONES,
        num_weeks=NUM_WEEKS,
        seed=RANDOM_SEED
    )

    validate_and_display_data(df_sim)

    output_path = str(data_dir / "traffic_simulation.csv")
    save_simulation_data(df_sim, output_path)
