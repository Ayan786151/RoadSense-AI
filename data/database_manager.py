"""
================================================================================
ROADSENSE AI — MASTER TRAFFIC & FACTOR DATABASE MANAGER
MODULE: data/database_manager.py
================================================================================

This module consolidates all multi-modal traffic variables, environmental factors,
vision measurements (vehicle mix percentages, kinematic states), geographic
coordinates, and temporal momentum features into an organized, persistent
SQLite Master Database (data/road_sense_master.db) and unified CSV file.
================================================================================
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


# ==============================================================================
# 0. DATABASE FILE PATHS
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR if os.path.basename(SCRIPT_DIR) == "traffic_sim" else os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SQLITE_DB_PATH = os.path.join(DATA_DIR, "road_sense_master.db")
UNIFIED_CSV_PATH = os.path.join(DATA_DIR, "unified_traffic_database.csv")
DICTIONARY_JSON_PATH = os.path.join(DATA_DIR, "dataset_factor_dictionary.json")


# ==============================================================================
# 1. FACTOR GLOSSARY & DOMAIN TAXONOMY
# ==============================================================================

FACTOR_TAXONOMY = {
    # 1. Spatial & Identification
    "zone_id": {"category": "Spatial Identification", "type": "Categorical", "unit": "ID", "description": "Unique municipal traffic zone identifier (Zone_01 to Zone_50)."},
    "week": {"category": "Temporal Dimension", "type": "Integer", "unit": "Weeks (1-52)", "description": "Chronological calendar observation week."},
    "location_name": {"category": "Spatial Identification", "type": "Categorical", "unit": "Name", "description": "Real-world neighborhood/corridor name."},
    "city": {"category": "Spatial Identification", "type": "Categorical", "unit": "City", "description": "Metropolitan municipality (e.g. Bengaluru, Mumbai, Delhi, Chennai)."},
    "latitude": {"category": "Spatial Identification", "type": "Float", "unit": "Degrees N", "description": "Geographic latitude coordinate."},
    "longitude": {"category": "Spatial Identification", "type": "Float", "unit": "Degrees E", "description": "Geographic longitude coordinate."},
    "zone_type": {"category": "Urban Characterization", "type": "Categorical", "unit": "Archetype", "description": "Urban archetype: Commercial, Residential, Highway Corridor, Transit Hub, School Zone, Industrial."},
    "temporal_profile": {"category": "Urban Characterization", "type": "Categorical", "unit": "Pattern", "description": "Simulated multi-week temporal profile (High Momentum, Volatile, Steady High Risk, Calm)."},

    # 2. Environmental & Climate Factors
    "weather": {"category": "Environmental & Climate", "type": "Categorical", "unit": "State", "description": "Atmospheric condition: Clear, Light Rain, Heavy Rain, Fog, Storm, Heatwave."},
    "road_condition": {"category": "Environmental & Climate", "type": "Categorical", "unit": "Quality", "description": "Road surface state: Good, Wet, Potholes, Under Construction, Slippery."},
    "special_event": {"category": "Environmental & Climate", "type": "Binary (0/1)", "unit": "Flag", "description": "Whether a major festival, sports match, or political rally surged traffic demand."},

    # 3. Infrastructure & Capacity Factors
    "population_density": {"category": "Infrastructure & Pressure", "type": "Numeric", "unit": "persons/km²", "description": "Residential and commercial human density surrounding the corridor."},
    "road_capacity": {"category": "Infrastructure & Pressure", "type": "Numeric", "unit": "Index (0-100)", "description": "Design vehicle throughput capacity of the physical roadway."},
    "effective_road_capacity": {"category": "Infrastructure & Pressure", "type": "Numeric", "unit": "Index (0-100)", "description": "Active road capacity after adjusting for adverse weather, potholes, and construction bottlenecks."},
    "vehicle_density": {"category": "Infrastructure & Pressure", "type": "Numeric", "unit": "veh/km²", "description": "Active vehicular concentration on the roadway."},
    "traffic_pressure": {"category": "Infrastructure & Pressure", "type": "Numeric", "unit": "Ratio", "description": "Vehicle density divided by effective road capacity. Pressure > 1.0 indicates severe capacity breach."},
    "vehicle_population_ratio": {"category": "Infrastructure & Pressure", "type": "Numeric", "unit": "Ratio", "description": "Ratio of vehicle concentration to human population density."},

    # 4. Traffic Flow & Kinematics
    "congestion": {"category": "Traffic Flow & Kinematics", "type": "Numeric", "unit": "Index (0-100)", "description": "Non-linear sigmoid traffic saturation index."},
    "average_speed": {"category": "Traffic Flow & Kinematics", "type": "Numeric", "unit": "km/h", "description": "Mean vehicular speed across the observation corridor."},
    "red_light_violations": {"category": "Traffic Flow & Kinematics", "type": "Numeric", "unit": "Violations/wk", "description": "Number of stop-line and optical traffic signal violations detected."},

    # 5. Vehicle Modal Composition Factors (Vision Ingestion)
    "car_percentage": {"category": "Vehicle Modal Mix", "type": "Numeric", "unit": "Percentage (%)", "description": "Share of 4-wheeler passenger cars in the traffic stream."},
    "motorcycle_percentage": {"category": "Vehicle Modal Mix", "type": "Numeric", "unit": "Percentage (%)", "description": "Share of vulnerable 2-wheelers (motorcycles/scooters) in the traffic stream."},
    "bus_percentage": {"category": "Vehicle Modal Mix", "type": "Numeric", "unit": "Percentage (%)", "description": "Share of public transit and private buses."},
    "truck_percentage": {"category": "Vehicle Modal Mix", "type": "Numeric", "unit": "Percentage (%)", "description": "Share of heavy commercial freight trucks."},

    # 6. Kinematic State Distribution Factors
    "moving_vehicle_percentage": {"category": "Kinematic States", "type": "Numeric", "unit": "Percentage (%)", "description": "Share of vehicles moving at fluid free-flow speeds (> 30 km/h)."},
    "slow_vehicle_percentage": {"category": "Kinematic States", "type": "Numeric", "unit": "Percentage (%)", "description": "Share of vehicles experiencing crawling stop-and-go speeds (10-30 km/h)."},
    "stopped_vehicle_percentage": {"category": "Kinematic States", "type": "Numeric", "unit": "Percentage (%)", "description": "Share of completely stationary/gridlocked vehicles (< 10 km/h)."},

    # 7. Safety Targets
    "incident_occurred": {"category": "Safety Target", "type": "Binary (0/1)", "unit": "Target", "description": "Ground-truth occurrence of a traffic collision, injury, or severe disruption."},
    "incident_count": {"category": "Safety Target", "type": "Integer", "unit": "Count", "description": "Total number of recorded incidents during the week."}
}


# ==============================================================================
# 2. MASTER DATABASE CONSOLIDATION & ENRICHMENT
# ==============================================================================

def build_master_database() -> pd.DataFrame:
    """
    Consolidates and enriches all multi-modal traffic factors, simulation series,
    geographic locations, and vision kinematics into a single master database.
    """
    sim_path = os.path.join(DATA_DIR, "simulation_temporal_features.csv")
    if not os.path.exists(sim_path):
        sim_path = os.path.join(DATA_DIR, "temporal_features.csv")
    loc_path = os.path.join(DATA_DIR, "location_mapping.csv")

    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"Missing core simulation dataset at: {sim_path}")

    sim_df = pd.read_csv(sim_path)

    # 1. Merge Location Mapping
    if os.path.exists(loc_path):
        loc_df = pd.read_csv(loc_path)
        merged_df = pd.merge(sim_df, loc_df, on="zone_id", how="left")
    else:
        merged_df = sim_df.copy()
        merged_df["location_name"] = merged_df["zone_id"]
        merged_df["city"] = "Metropolis"
        merged_df["latitude"] = 19.0760
        merged_df["longitude"] = 72.8777

    # Fill defaults for location columns
    merged_df["location_name"] = merged_df["location_name"].fillna(merged_df["zone_id"])
    merged_df["city"] = merged_df["city"].fillna("Metropolis")
    merged_df["latitude"] = merged_df["latitude"].fillna(19.0760)
    merged_df["longitude"] = merged_df["longitude"].fillna(72.8777)

    # 2. Synthesize/Enrich Multi-Modal Vehicle Composition Factors based on Urban Archetypes
    # Commercial/Highway: Higher cars/trucks; Residential/School: High 2-wheelers; Transit Hub: High buses
    np.random.seed(42)
    n_rows = len(merged_df)

    if "car_percentage" not in merged_df.columns:
        zone_types = merged_df["zone_type"].values
        car_pcts = []
        moto_pcts = []
        bus_pcts = []
        truck_pcts = []

        for zt in zone_types:
            if zt in ["Commercial Hub", "Commercial"]:
                c, m, b, t = 52.0, 32.0, 10.0, 6.0
            elif zt in ["Residential Sector", "Residential", "School Zone"]:
                c, m, b, t = 34.0, 56.0, 6.0, 4.0
            elif zt in ["Highway Corridor", "Industrial"]:
                c, m, b, t = 40.0, 20.0, 15.0, 25.0
            elif zt in ["Transit Hub"]:
                c, m, b, t = 35.0, 30.0, 30.0, 5.0
            else:
                c, m, b, t = 45.0, 40.0, 10.0, 5.0

            # Add natural stochastic variance
            noise = np.random.normal(0, 2.5, 4)
            raw = np.maximum(2.0, np.array([c, m, b, t]) + noise)
            norm = (raw / raw.sum()) * 100.0
            car_pcts.append(round(norm[0], 1))
            moto_pcts.append(round(norm[1], 1))
            bus_pcts.append(round(norm[2], 1))
            truck_pcts.append(round(norm[3], 1))

        merged_df["car_percentage"] = car_pcts
        merged_df["motorcycle_percentage"] = moto_pcts
        merged_df["bus_percentage"] = bus_pcts
        merged_df["truck_percentage"] = truck_pcts

    # 3. Synthesize/Enrich Kinematic Proportions (Moving %, Slow %, Stopped %) based on Congestion
    if "stopped_vehicle_percentage" not in merged_df.columns:
        cong_vals = merged_df["congestion"].values
        moving_pcts = []
        slow_pcts = []
        stopped_pcts = []

        for cong in cong_vals:
            # Non-linear kinematic distribution based on congestion saturation
            if cong < 35:
                mov = np.random.uniform(70, 90)
                slw = np.random.uniform(8, 20)
                stp = max(0.0, 100.0 - mov - slw)
            elif cong < 65:
                mov = np.random.uniform(35, 60)
                slw = np.random.uniform(25, 45)
                stp = max(0.0, 100.0 - mov - slw)
            else:
                mov = np.random.uniform(5, 25)
                slw = np.random.uniform(30, 50)
                stp = max(0.0, 100.0 - mov - slw)

            moving_pcts.append(round(mov, 1))
            slow_pcts.append(round(slw, 1))
            stopped_pcts.append(round(stp, 1))

        merged_df["moving_vehicle_percentage"] = moving_pcts
        merged_df["slow_vehicle_percentage"] = slow_pcts
        merged_df["stopped_vehicle_percentage"] = stopped_pcts

    # Ensure clean sorting
    merged_df = merged_df.sort_values(by=["zone_id", "week"]).reset_index(drop=True)

    # 4. Save to Unified CSV File
    merged_df.to_csv(UNIFIED_CSV_PATH, index=False)
    print(f"[+] Successfully exported Unified Master Dataset ({len(merged_df):,} rows × {len(merged_df.columns)} cols) to:\n    {UNIFIED_CSV_PATH}")

    # 5. Save to Persistent SQLite Database
    save_to_sqlite(merged_df, SQLITE_DB_PATH)

    # 6. Generate and Save Factor Statistical Dictionary
    generate_factor_dictionary(merged_df, DICTIONARY_JSON_PATH)

    return merged_df


# ==============================================================================
# 3. SQLITE DATABASE PERSISTENCE & QUERY ENGINE
# ==============================================================================

def save_to_sqlite(df: pd.DataFrame, db_path: str):
    """Saves DataFrame into SQLite table 'master_traffic_records' with indexes."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # Write master table
    df.to_sql("master_traffic_records", conn, if_exists="replace", index=False)
    
    # Create indexes for high-speed temporal queries
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_zone_week ON master_traffic_records (zone_id, week);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_city ON master_traffic_records (city);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_weather ON master_traffic_records (weather);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_incident ON master_traffic_records (incident_occurred);")
    conn.commit()
    conn.close()
    print(f"[+] Successfully saved SQLite Master Database with indexed tables to:\n    {db_path}")


def query_master_database(sql_query: str) -> pd.DataFrame:
    """Executes arbitrary SQL query against the SQLite Master Database and returns DataFrame."""
    if not os.path.exists(SQLITE_DB_PATH):
        build_master_database()
    
    conn = sqlite3.connect(SQLITE_DB_PATH)
    result_df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return result_df


# ==============================================================================
# 4. FACTOR INTELLIGENCE & STATISTICAL PROFILER
# ==============================================================================

def generate_factor_dictionary(df: pd.DataFrame, output_json: str) -> Dict[str, Any]:
    """Computes comprehensive summary statistics and metadata for every factor in the database."""
    factor_meta = {}

    for col in df.columns:
        tax = FACTOR_TAXONOMY.get(col, {
            "category": "Temporal / Derived Factor" if ("rolling" in col or "trend" in col or "previous" in col or "change" in col) else "General Factor",
            "type": "Numeric" if pd.api.types.is_numeric_dtype(df[col]) else "Categorical",
            "unit": "Score / Delta" if pd.api.types.is_numeric_dtype(df[col]) else "Category",
            "description": f"Feature variable: {col}"
        })

        if pd.api.types.is_numeric_dtype(df[col]):
            stats = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": round(float(df[col].mean()), 3),
                "std": round(float(df[col].std()), 3),
                "median": round(float(df[col].median()), 3),
                "null_count": int(df[col].isnull().sum())
            }
        else:
            stats = {
                "unique_values": list(df[col].dropna().unique()[:10]),
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum())
            }

        factor_meta[col] = {
            "name": col,
            "category": tax["category"],
            "data_type": tax["type"],
            "unit": tax["unit"],
            "description": tax["description"],
            "statistics": stats
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(factor_meta, f, indent=2)

    print(f"[+] Successfully generated Factor Intelligence Dictionary ({len(factor_meta)} factors) to:\n    {output_json}")
    return factor_meta


def get_factor_intelligence_summary() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads the unified master database and statistical factor dictionary."""
    if not os.path.exists(UNIFIED_CSV_PATH) or not os.path.exists(SQLITE_DB_PATH):
        df = build_master_database()
    else:
        df = pd.read_csv(UNIFIED_CSV_PATH)

    if os.path.exists(DICTIONARY_JSON_PATH):
        with open(DICTIONARY_JSON_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = generate_factor_dictionary(df, DICTIONARY_JSON_PATH)

    return df, meta


# ==============================================================================
# 5. CLI EXECUTION & VALIDATION
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" ROADSENSE AI — MASTER DATABASE & FACTOR CONSOLIDATION ".center(80, "="))
    print("=" * 80 + "\n")
    
    df_master = build_master_database()
    print(f"\n[+] Master Database Schema ({len(df_master.columns)} Total Columns):")
    for idx, c in enumerate(df_master.columns, 1):
        print(f"    {idx:>2}. {c:<36} ({df_master[c].dtype})")
    
    # Test SQL query execution
    print("\n[+] Testing SQLite Database Query:")
    sample_query = "SELECT city, zone_type, COUNT(*) as zone_weeks, AVG(congestion) as avg_cong, SUM(incident_occurred) as total_incidents FROM master_traffic_records GROUP BY city, zone_type ORDER BY total_incidents DESC LIMIT 5;"
    res_sql = query_master_database(sample_query)
    print(res_sql.to_string(index=False))
    print("\n" + "=" * 80 + "\n")
