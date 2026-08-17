"""
================================================================================
TEMPORAL AI-BASED TRAFFIC RISK AND PRIORITY INTELLIGENCE SYSTEM
MODULE: DEFINITIVE SYNTHETIC CITY SIMULATOR (Phases 1 to 5)
================================================================================

This module generates a rich, realistic 52-week synthetic panel dataset of
urban traffic conditions across 50 zones (2,600 zone-week observations).

CORE SYSTEM PRINCIPLE:
1. RISK (Probability)       : P(Incident | Congestion, Speed, Violations, Weather, History)
2. EXPOSURE (Volume)        : Population Density + Vehicle Density (Citizens at risk)
3. PRIORITY (Action Index)  : Resource allocation combining Risk x Exposure x Trend

TEMPORAL DYNAMICS PROFILES:
- STABLE                    : Predictable baseline with standard weekly stochastic variation
- GRADUAL_DETERIORATION     : Traffic and congestion progressively worsen over 52 weeks
- GRADUAL_IMPROVEMENT       : Traffic and congestion progressively improve over 52 weeks
- PERIODIC_SPIKE            : Regular recurring traffic surges (e.g., stadium/market activity)
- EVENT_SENSITIVE           : Normal baseline, but highly reactive to municipal events
- RECOVERY_AFTER_DISRUPTION : Baseline -> Deterioration -> Peak Strain -> Recovery -> Stable

================================================================================
"""

import os
import time
import numpy as np
import pandas as pd


# ==============================================================================
# 0. CONFIGURATION CONSTANTS
# ==============================================================================

NUM_ZONES = 50          # 50 distinct geographical urban zones
NUM_WEEKS = 52          # 52 weeks (1 full year of temporal data = 2,600 observations)
RANDOM_SEED = 42        # Ensures exact mathematical reproducibility

POPULATION_MIN = 1200   # Min residential population per sq km
POPULATION_MAX = 16000  # Max residential population per sq km
ROAD_CAPACITY_MIN = 40  # Min road infrastructure capacity score (0-100 scale)
ROAD_CAPACITY_MAX = 100 # Max road infrastructure capacity score (0-100 scale)


# ==============================================================================
# 1. MATHEMATICAL HELPER FUNCTIONS
# ==============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Standard Logistic Sigmoid Function:
    Smoothly maps any real-valued number (logit) into a bounded probability [0, 1].
    Formula: S(x) = 1 / (1 + e^(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))


# ==============================================================================
# 2. PHASE 1: ZONE GENERATOR & TEMPORAL PROFILES (Stable Infrastructure)
# ==============================================================================

def create_city_zones(num_zones: int = NUM_ZONES, seed: int = RANDOM_SEED) -> list:
    """
    Creates 50 persistent physical zones with:
    1. Urban Archetypes (Commercial, Residential, University, Industrial, Highway, Suburban)
    2. Distinct Temporal Profiles (Stable, Deterioration, Improvement, Spike, Event-Sensitive, Recovery)
    """
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

    # Explicit 6-Profile Temporal Distribution
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

        # Baseline population tailored to archetype
        if archetype == "Residential":
            pop = np.random.randint(8000, POPULATION_MAX)
            activity_pull = np.random.uniform(0.70, 0.85)
        elif archetype == "Commercial_Downtown":
            pop = np.random.randint(3500, 9500)
            activity_pull = np.random.uniform(1.20, 1.55)  # Heavy commuter pull
        elif archetype == "University_District":
            pop = np.random.randint(6000, 12000)
            activity_pull = np.random.uniform(1.00, 1.30)
        elif archetype == "Industrial_Corridor":
            pop = np.random.randint(1800, 4500)
            activity_pull = np.random.uniform(1.10, 1.35)  # Freight trucks
        elif archetype == "Highway_Junction":
            pop = np.random.randint(POPULATION_MIN, 3000)  # Very low population
            activity_pull = np.random.uniform(1.40, 1.80)  # Massive traffic corridor
        else: # Suburban_LowDensity
            pop = np.random.randint(POPULATION_MIN, 4500)
            activity_pull = np.random.uniform(0.50, 0.70)

        # Baseline road capacity (40 to 100)
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


# ==============================================================================
# 3. PHASE 4: ENVIRONMENTAL & TEMPORAL SCENARIO CONTROLLER
# ==============================================================================

def get_temporal_environment(week: int, zone: dict) -> dict:
    """
    Computes weekly environmental shocks and zone-specific temporal drift:
    
    1. Macro Weather Timeline:
       - Weeks 31-36: Monsoon / Severe Rain Season (capacity drop, speed drop)
       - Weeks 48-52: Winter Rain & Storms
       - Other Weeks: Normal with sporadic light rain
       
    2. Zone-Specific Temporal Evolution:
       - STABLE                    : Drift = 0.0
       - GRADUAL_DETERIORATION     : Progressive traffic drift (+22.0 over 52 weeks)
       - GRADUAL_IMPROVEMENT       : Progressive traffic drift (-16.0 over 52 weeks)
       - PERIODIC_SPIKE            : Bi-monthly surge (+20.0 every 5 weeks)
       - EVENT_SENSITIVE           : Sharp surge on scheduled events (+22.0, capacity -15%)
       - RECOVERY_AFTER_DISRUPTION : Rise -> Peak -> Recovery -> Stable
    """
    archetype = zone["zone_type"]
    profile = zone["temporal_profile"]

    # 1. Weather Modeling
    weather = "Normal"
    road_condition = "Good"

    # Monsoon Season (Weeks 31-36)
    if 31 <= week <= 36:
        roll = np.random.rand()
        if roll < 0.38:
            weather = "Heavy Rain"
            road_condition = "Poor"
        elif roll < 0.78:
            weather = "Light Rain"
            road_condition = "Moderate"
    # Winter Rain (Weeks 48-52)
    elif 48 <= week <= 52:
        if np.random.rand() < 0.40:
            weather = "Light Rain"
            road_condition = "Moderate"
    else:
        if np.random.rand() < 0.12:
            weather = "Light Rain"
            road_condition = "Moderate"

    # 2. Special Events Modeling
    special_event = 0
    event_surge_vehicles = 0.0
    event_capacity_penalty = 1.0

    # Spring Festival (Weeks 18-19 in Commercial & University districts)
    if (18 <= week <= 19) and (archetype in ["Commercial_Downtown", "University_District"]):
        special_event = 1
        event_surge_vehicles = 18.0
        event_capacity_penalty = 0.85

    # Autumn Expo & Conventions (Weeks 38-40 in Highway Junctions & Downtown)
    elif (38 <= week <= 40) and (archetype in ["Commercial_Downtown", "Highway_Junction"]):
        special_event = 1
        event_surge_vehicles = 22.0
        event_capacity_penalty = 0.80

    # 3. Zone-Specific Temporal Dynamics (The 6 Profiles)
    trend_drift = 0.0

    if profile == "GRADUAL_DETERIORATION":
        # Worsens smoothly over 52 weeks
        trend_drift = (week / 52.0) * 22.0

    elif profile == "GRADUAL_IMPROVEMENT":
        # Improves smoothly over 52 weeks
        trend_drift = -(week / 52.0) * 16.0

    elif profile == "PERIODIC_SPIKE":
        # Spikes periodically every 5 weeks (e.g., weeks 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
        if week % 5 == 0:
            trend_drift = 19.0
            special_event = 1
            event_capacity_penalty = 0.90
        else:
            trend_drift = 0.0

    elif profile == "EVENT_SENSITIVE":
        # Responds with extra sensitivity during scheduled municipal events
        if special_event == 1:
            trend_drift = 12.0
            event_capacity_penalty *= 0.90
        # Recurring local community gathering in weeks 8, 24, 42
        elif week in [8, 24, 42]:
            special_event = 1
            trend_drift = 17.0
            event_capacity_penalty = 0.88

    elif profile == "RECOVERY_AFTER_DISRUPTION":
        # Narrative: Normal -> Deterioration -> Peak Disruption -> Remediation -> Recovery
        if week <= 10:
            trend_drift = 0.0
        elif 11 <= week <= 20:
            # Deterioration starts
            trend_drift = ((week - 10) / 10.0) * 18.0
        elif 21 <= week <= 28:
            # Peak bottleneck period
            trend_drift = 20.0
            event_capacity_penalty = 0.82
        elif 29 <= week <= 38:
            # Remediation & recovery taking effect
            progress = (week - 28) / 10.0
            trend_drift = 20.0 - (progress * 22.0)  # Drops from +20 down to -2
        else: # Weeks 39 to 52
            # Post-remediation stable improvement
            trend_drift = -2.5

    else: # STABLE
        trend_drift = 0.0

    return {
        "weather": weather,
        "road_condition": road_condition,
        "special_event": special_event,
        "event_surge_vehicles": event_surge_vehicles,
        "event_capacity_penalty": event_capacity_penalty,
        "trend_drift": trend_drift
    }


# ==============================================================================
# 4. PHYSICAL DYNAMICS & SIMULATION FUNCTIONS
# ==============================================================================

def calculate_effective_capacity(
    base_capacity: int,
    weather: str,
    road_condition: str,
    event_capacity_penalty: float
) -> float:
    """
    Computes usable effective road throughput.
    Effective Capacity = Base Capacity * Weather Factor * Road Condition Factor * Event Disruption
    """
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
    """
    Computes vehicle density (0-100 scale).
    Combines:
    - Base Activity pull from population
    - Zone temporal profile drift (Deterioration, Improvement, Spikes, Recovery)
    - Special event surge
    - Autoregressive continuity (AR(1): 65% past state + 35% new target + noise)
    """
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
    """
    Computes Traffic Pressure and Congestion Index (0-100 scale).
    Traffic Pressure = Vehicle Density / Effective Road Capacity
    Congestion follows a non-linear Sigmoid response curve to pressure.
    """
    traffic_pressure = vehicle_density / max(1.0, effective_capacity)
    w_mult = {"Normal": 1.00, "Light Rain": 1.12, "Heavy Rain": 1.28}[weather]
    
    # Sigmoid curve centered around pressure threshold 0.85
    raw_congestion = 100.0 * sigmoid(3.4 * (traffic_pressure - 0.85)) * w_mult
    noise = np.random.normal(0.0, 2.5)

    congestion = float(np.clip(round(raw_congestion + noise, 2), 0.0, 100.0))
    return round(traffic_pressure, 3), congestion


def calculate_speed(congestion: float, weather: str, road_condition: str) -> float:
    """
    Computes average vehicle speed (km/h).
    Free-flow speed (65 km/h) reduced non-linearly by congestion and surface conditions.
    """
    w_penalty = {"Normal": 0.0, "Light Rain": 5.0, "Heavy Rain": 13.0}[weather]
    r_penalty = {"Good": 0.0, "Moderate": 3.0, "Poor": 8.0}[road_condition]

    speed_degradation = (congestion / 100.0) ** 1.35 * 38.0
    raw_speed = 65.0 - speed_degradation - w_penalty - r_penalty + np.random.normal(0.0, 1.8)
    return float(np.clip(round(raw_speed, 2), 10.0, 80.0))


def generate_violations(congestion: float, special_event: int, archetype: str) -> int:
    """
    Computes Red-Light / Traffic Violations count via Poisson distribution.
    """
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
    """
    Generates Incident Probability, Occurrence, and Count via Latent Log-Odds.
    """
    w_risk = {"Normal": 0.0, "Light Rain": 0.30, "Heavy Rain": 0.80}[weather]
    r_risk = {"Good": 0.0, "Moderate": 0.25, "Poor": 0.60}[road_condition]

    risk_logit = (
        -3.40                                    # Baseline intercept
        + 0.036 * congestion                     # Congestion impact
        + 0.095 * violations                     # Violations impact
        + 0.040 * max(0.0, 42.0 - average_speed) # Stop-and-go speed hazard
        + 0.350 * max(0.0, traffic_pressure - 1.0) # Capacity overflow penalty
        + 0.250 * recent_incident_memory         # Autoregressive historical memory
        + w_risk
        + r_risk
        + np.random.normal(0.0, 0.18)            # Natural variance
    )

    incident_prob = float(np.clip(sigmoid(risk_logit), 0.01, 0.96))
    incident_occurred = int(np.random.rand() < incident_prob)
    incident_count = int(1 + np.random.poisson(lam=incident_prob * 1.3)) if incident_occurred else 0
    return incident_count, incident_occurred


# ==============================================================================
# 5. FULL SIMULATION PIPELINE (Phases 1 to 5)
# ==============================================================================

def run_traffic_simulator(
    num_zones: int = NUM_ZONES,
    num_weeks: int = NUM_WEEKS,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Executes the definitive synthetic traffic simulation across all zones and weeks.
    """
    zones = create_city_zones(num_zones=num_zones, seed=seed)
    records = []

    history_tracker = {
        z["zone_id"]: {
            "prev_density": None,
            "past_incidents": []
        } for z in zones
    }

    # Chronological simulation week-by-week
    for week in range(1, num_weeks + 1):
        for zone in zones:
            z_id = zone["zone_id"]
            pop = zone["population_density"]
            base_cap = zone["road_capacity"]
            archetype = zone["zone_type"]
            profile = zone["temporal_profile"]
            activity_pull = zone["activity_pull"]

            # 1. Environmental & Scenario conditions
            env = get_temporal_environment(week, zone)
            weather = env["weather"]
            road_cond = env["road_condition"]
            event = env["special_event"]
            event_surge = env["event_surge_vehicles"]
            cap_penalty = env["event_capacity_penalty"]
            trend_drift = env["trend_drift"]

            # 2. Effective Road Capacity
            eff_capacity = calculate_effective_capacity(
                base_capacity=base_cap,
                weather=weather,
                road_condition=road_cond,
                event_capacity_penalty=cap_penalty
            )

            # 3. Vehicle Density
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

            # 4. Traffic Pressure & Congestion
            pressure, congestion = calculate_congestion(
                vehicle_density=vehicle_density,
                effective_capacity=eff_capacity,
                weather=weather
            )

            # 5. Average Speed
            avg_speed = calculate_speed(
                congestion=congestion,
                weather=weather,
                road_condition=road_cond
            )

            # 6. Red Light Violations
            violations = generate_violations(
                congestion=congestion,
                special_event=event,
                archetype=archetype
            )

            # 7. Historical Incident Memory (Strictly past weeks, zero data leakage)
            past_inc = history_tracker[z_id]["past_incidents"]
            recent_memory = float(np.mean(past_inc[-4:])) if len(past_inc) > 0 else 0.0

            # 8. Incident Generation
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


# ==============================================================================
# 6. COMPREHENSIVE DATA QUALITY & TEMPORAL VALIDATION SUITE
# ==============================================================================

def validate_and_display_data(df: pd.DataFrame):
    """
    Performs automated quality validations, temporal trajectory audits,
    and prints demonstration timelines for each profile.
    """
    print("\n" + "=" * 94)
    print(" TEMPORAL TRAFFIC SIMULATOR -- DEFINITIVE VALIDATION REPORT ".center(94, "="))
    print("=" * 94)

    # 1. Dataset Dimensions & Completeness
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

    # 2. Temporal Profile Distribution & Trajectory Audit
    print("\n[+] 2. TEMPORAL BEHAVIOR PROFILE AUDIT:")
    print("-" * 94)
    profile_counts = df.groupby("zone_id")["temporal_profile"].first().value_counts()
    for prof, cnt in profile_counts.items():
        print(f"    - {prof:<28}: {cnt:>2} zones ({cnt/unique_zones*100:>4.1f}%)")
    print("-" * 94)

    # 3. Demonstration Zones (Timeline snapshots across Weeks 1, 10, 20, 30, 40, 52)
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

    # 4. Temporal Trend Correlations
    print("\n[+] 4. TEMPORAL CORRELATION AUDIT (Week vs Metrics per Profile):")
    trend_audit = df.groupby("temporal_profile").apply(
        lambda g: pd.Series({
            "Corr(Week, Vehicles)": g[["week", "vehicle_density"]].corr().iloc[0, 1],
            "Corr(Week, Congestion)": g[["week", "congestion"]].corr().iloc[0, 1],
            "Avg Incidents / Wk": g["incident_occurred"].mean()
        }), include_groups=False
    ).round(3)
    print(trend_audit.to_string())

    # 5. Causal Correlation Matrix
    print("\n[+] 5. CAUSAL CORRELATION MATRIX (Verifying Physics & Exposure Separation):")
    numeric_cols = [
        "population_density", "vehicle_density", "effective_road_capacity",
        "traffic_pressure", "congestion", "average_speed", "red_light_violations",
        "incident_occurred"
    ]
    print(df[numeric_cols].corr().round(3).to_string())

    # 6. Quality Assertions & Bounds Checks
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


# ==============================================================================
# 7. MAIN EXECUTION ENTRYPOINT
# ==============================================================================

def save_simulation_data(df: pd.DataFrame, target_path: str):
    """
    Safely saves dataframe to CSV handling potential file locks gracefully.
    """
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

    # Validate output
    validate_and_display_data(df_sim)

    # Save dataset safely
    output_path = str(data_dir / "traffic_simulation.csv")
    save_simulation_data(df_sim, output_path)
