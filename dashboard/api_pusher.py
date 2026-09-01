import os
import json
import math
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Define project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Import intelligence and priority modules
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from intelligence.signal_co2 import compute_optimal_signal_timing, estimate_co2_impact
from intelligence.llm_briefing import generate_zone_briefing
from priority.intervention_engine import generate_interventions, generate_priority_explanation


def clean_for_json(obj: Any) -> Any:
    """Recursively sanitizes data structure converting NaN/Inf and numpy types to JSON-compliant primitives."""
    if obj is None:
        return None
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(round(obj, 4))
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [clean_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    return obj


def get_risk_badge(prob: float):
    """Returns risk label and hex color code based on predicted probability."""
    if prob is None or (isinstance(prob, float) and math.isnan(prob)):
        return "BASELINE WARMUP", "#71717a"
    elif prob >= 0.75:
        return "CRITICAL RISK", "#ef4444"
    elif prob >= 0.55:
        return "HIGH RISK", "#f97316"
    elif prob >= 0.35:
        return "MODERATE RISK", "#eab308"
    else:
        return "LOW RISK", "#22c55e"


def compute_priority_scores_and_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes component scores, composite priority score, rank (1-50), and priority level
    for every zone within each week.
    """
    df = df.copy()

    # Risk score (0-100)
    df["risk_score"] = pd.to_numeric(df["predicted_risk_probability"], errors="coerce").fillna(0.0) * 100.0

    # Grouped/Normalized exposure scores across zones per week
    def normalize_series(s):
        s_num = pd.to_numeric(s, errors="coerce").fillna(0.0)
        s_min, s_max = s_num.min(), s_num.max()
        if pd.isna(s_max) or pd.isna(s_min) or s_max == s_min:
            return pd.Series(50.0, index=s.index)
        return ((s_num - s_min) / (s_max - s_min)) * 100.0

    df["population_exposure_score"] = df.groupby("week")["population_density"].transform(normalize_series).fillna(50.0)
    df["vehicle_exposure_score"] = df.groupby("week")["vehicle_density"].transform(normalize_series).fillna(50.0)

    # Temporal trend score from 4-week congestion slope
    trend_col = "congestion_trend_4w" if "congestion_trend_4w" in df.columns else "congestion"
    df["temporal_trend_score"] = df.groupby("week")[trend_col].transform(normalize_series).fillna(50.0).clip(0, 100)

    # Weighted Composite Priority Score (40% Risk + 25% Pop + 20% Veh + 15% Trend)
    df["priority_score"] = (
        0.40 * df["risk_score"] +
        0.25 * df["population_exposure_score"] +
        0.20 * df["vehicle_exposure_score"] +
        0.15 * df["temporal_trend_score"]
    ).fillna(50.0).clip(0, 100)

    # Priority Rank (1 = highest priority in that week)
    df["priority_rank"] = df.groupby("week")["priority_score"].rank(method="min", ascending=False).fillna(50).astype(int)

    # Priority Level Classification
    def classify_priority(score):
        if pd.isna(score):
            return "MODERATE"
        if score >= 75:
            return "CRITICAL"
        elif score >= 55:
            return "HIGH"
        elif score >= 35:
            return "MODERATE"
        else:
            return "LOW"

    df["priority_level"] = df["priority_score"].apply(classify_priority)
    return df


def build_static_api():
    """Builds a comprehensive static JSON REST API for frontend integration."""
    api_dir = os.path.join(BASE_DIR, "api", "v1")
    os.makedirs(api_dir, exist_ok=True)
    os.makedirs(os.path.join(api_dir, "zones"), exist_ok=True)
    os.makedirs(os.path.join(api_dir, "weeks"), exist_ok=True)
    
    print("[1/5] Loading simulation data, location mappings, and ML model...")
    sim_path = os.path.join(BASE_DIR, "data", "simulation_temporal_features.csv")
    loc_path = os.path.join(BASE_DIR, "data", "location_mapping.csv")
    model_path = os.path.join(BASE_DIR, "models", "best_risk_model.pkl")

    df = pd.read_csv(sim_path)
    
    # Merge location metadata if it exists
    if os.path.exists(loc_path):
        loc_df = pd.read_csv(loc_path)
        # Drop duplicates if any in loc_df
        loc_df = loc_df.drop_duplicates(subset=["zone_id"])
        df = pd.merge(df, loc_df, on="zone_id", how="left")
    else:
        df["location_name"] = df["zone_id"]
        df["city"] = "Metropolis"
        df["latitude"] = 19.0760
        df["longitude"] = 72.8777
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None

    # Compute predicted risk probability for all rows
    print("[2/5] Running vectorized ML model inference for all zones & weeks...")
    df["predicted_risk_probability"] = None
    if model is not None:
        valid_mask = df["week"] >= 5
        if valid_mask.any():
            try:
                probs = model.predict_proba(df[valid_mask])[:, 1]
                df.loc[valid_mask, "predicted_risk_probability"] = probs
            except Exception as e:
                print(f"Batch prediction fallback: {e}")
                for idx in df[valid_mask].index:
                    try:
                        df.loc[idx, "predicted_risk_probability"] = float(model.predict_proba(df.loc[[idx]])[0, 1])
                    except Exception:
                        pass

    # Compute Priority scores, ranks, and levels across weeks
    print("[3/5] Computing composite priority scores and rankings...")
    df = compute_priority_scores_and_ranks(df)

    zones = sorted(df["zone_id"].unique().tolist())
    weeks = sorted(df["week"].unique().tolist())

    # Generate Zones Master List (/api/v1/zones.json)
    zone_catalog = []
    for zone in zones:
        z_sample = df[df["zone_id"] == zone].iloc[0]
        zone_catalog.append({
            "zone_id": zone,
            "location_name": str(z_sample.get("location_name", zone)),
            "city": str(z_sample.get("city", "")),
            "zone_type": str(z_sample.get("zone_type", "")),
            "latitude": float(z_sample.get("latitude", 0.0)),
            "longitude": float(z_sample.get("longitude", 0.0))
        })

    with open(f"{api_dir}/zones.json", "w") as f:
        json.dump({
            "total_zones": len(zones),
            "weeks_available": [int(w) for w in weeks],
            "zones": zone_catalog
        }, f, indent=2)

    print(f"[4/5] Generating rich zone & timeline endpoints for {len(zones)} zones...")

    # Dictionary to aggregate week-level leaderboards
    week_leaderboards: Dict[int, List[Dict[str, Any]]] = {int(w): [] for w in weeks}

    for zone in zones:
        zone_dir = f"{api_dir}/zones/{zone}"
        os.makedirs(zone_dir, exist_ok=True)
        
        zone_df = df[df["zone_id"] == zone].sort_values("week").reset_index(drop=True)
        all_weeks_data = []

        for i, row in zone_df.iterrows():
            week = int(row["week"])
            prev_row = zone_df.iloc[i - 1] if i > 0 else None

            risk_prob = row["predicted_risk_probability"]
            risk_badge, risk_color = get_risk_badge(risk_prob)

            # Compute week-over-week deltas
            prev_risk = prev_row["predicted_risk_probability"] if prev_row is not None else None
            risk_delta = ((risk_prob - prev_risk) * 100.0) if (risk_prob is not None and prev_risk is not None) else None
            cong_delta = float(row["congestion"] - prev_row["congestion"]) if prev_row is not None else None
            speed_delta = float(row["average_speed"] - prev_row["average_speed"]) if prev_row is not None else None
            dens_delta = float(row["vehicle_density"] - prev_row["vehicle_density"]) if prev_row is not None else None

            # Adaptive Signal Timing
            signal_info = compute_optimal_signal_timing(
                congestion=float(row["congestion"]),
                vehicle_density=float(row["vehicle_density"]),
                average_speed=float(row["average_speed"]),
                zone_type=str(row.get("zone_type", "Residential")),
                special_event=int(row.get("special_event", 0)),
                weather=str(row.get("weather", "Normal")),
            )

            # Sustainability & CO2 offsets
            co2_info = estimate_co2_impact(
                vehicle_density=float(row["vehicle_density"]),
                congestion=float(row["congestion"]),
                average_speed=float(row["average_speed"]),
                population_density=int(row.get("population_density", 5000)),
            )

            # Rule-based Municipal Interventions & Explanations
            row_dict = row.to_dict()
            row_dict["risk_prob"] = risk_prob
            interventions = generate_interventions(row_dict)
            explanation = generate_priority_explanation(row_dict)

            # Plain-English Executive Briefing
            briefing = generate_zone_briefing(row_dict, signal_info, co2_info)

            # Complete payload for this zone-week
            payload = {
                "zone_id": zone,
                "location_name": str(row.get("location_name", zone)),
                "city": str(row.get("city", "")),
                "zone_type": str(row.get("zone_type", "")),
                "week": week,
                "latitude": float(row.get("latitude", 0.0)),
                "longitude": float(row.get("longitude", 0.0)),
                
                # ML Risk Output
                "risk_analysis": {
                    "predicted_risk_probability": risk_prob,
                    "predicted_risk_percentage": round(risk_prob * 100.0, 2) if risk_prob is not None else None,
                    "risk_label": risk_badge,
                    "risk_color": risk_color,
                    "risk_delta_vs_prev_week": risk_delta,
                    "actual_incident": int(row.get("actual_incident", row.get("incident_occurred", 0)))
                },

                # Priority Ranking Scores
                "priority": {
                    "priority_score": float(row["priority_score"]),
                    "priority_rank": int(row["priority_rank"]),
                    "priority_level": str(row["priority_level"]),
                    "components": {
                        "risk_score": float(row["risk_score"]),
                        "population_exposure_score": float(row["population_exposure_score"]),
                        "vehicle_exposure_score": float(row["vehicle_exposure_score"]),
                        "temporal_trend_score": float(row["temporal_trend_score"])
                    },
                    "explanation": explanation
                },

                # Traffic Kinematics & KPIs
                "telemetry": {
                    "congestion": float(row["congestion"]),
                    "congestion_delta": cong_delta,
                    "average_speed_kmh": float(row["average_speed"]),
                    "speed_delta": speed_delta,
                    "vehicle_density": float(row["vehicle_density"]),
                    "density_delta": dens_delta,
                    "population_density": int(row.get("population_density", 0)),
                    "red_light_violations": int(row.get("red_light_violations", 0)),
                    "weather": str(row.get("weather", "Normal")),
                    "special_event": int(row.get("special_event", 0))
                },

                # Adaptive Signal Control
                "signal_optimization": signal_info,

                # CO2 & Economic Offsets
                "environmental_impact": co2_info,

                # Actionable Interventions & AI Briefing
                "interventions": interventions,
                "executive_briefing": briefing,

                # Civilian Road Safety Radar (Preserving all existing keys)
                "civilian_road_safety_radar": {
                    "current_week": {
                        "week": week,
                        "risk_score": round(risk_prob * 100.0, 1) if risk_prob is not None else None,
                        "risk_percentage": f"{round(risk_prob * 100.0):.0f}%" if risk_prob is not None else "N/A",
                        "risk_badge": risk_badge,
                        "risk_color": risk_color,
                        "description": "Elevated collision activity observed." if risk_prob and risk_prob >= 0.55 else "Corridor operating within normal baseline.",
                        "crashes_reported": int(row.get("actual_incident", 0))
                    },
                    "why_accidents_might_happen": {
                        "primary_collision_cause": {
                            "title": "Congestion Bottleneck & Merge Conflicts",
                            "explanation": "High vehicle density causes abrupt braking and sideswipe risks at merging points."
                        },
                        "behavior_factor": {
                            "title": "Tailgating & Speed Variance",
                            "explanation": "Drivers following too closely during peak hours increase rear-end impact frequency."
                        },
                        "street_environment": {
                            "zone_type": str(row.get("zone_type", "")),
                            "average_speed_kmh": float(row["average_speed"]),
                            "congestion_level": float(row["congestion"])
                        }
                    },
                    "civilian_safety_guide": {
                        "for_drivers": {
                            "action": "Increase Following Distance",
                            "details": "Maintain 3 car lengths buffer during peak rush hours."
                        },
                        "for_pedestrians_and_cyclists": {
                            "action": "Double-Check Turning Vehicles",
                            "details": "Make eye contact with turning drivers at intersections before crossing."
                        },
                        "peak_danger_windows": {
                            "peak_window": "Evening Commute (4:30 PM - 7:30 PM)",
                            "details": "Congestion peaks during evening travel windows."
                        }
                    }
                },

                # Full raw feature vector
                "raw_metrics": clean_for_json(row_dict)
            }

            clean_payload = clean_for_json(payload)

            # Save /api/v1/zones/{zone}/{week}.json
            with open(f"{zone_dir}/{week}.json", "w") as f:
                json.dump(clean_payload, f, indent=2)

            all_weeks_data.append(clean_payload)

            # Append to week leaderboard summary
            week_leaderboards[week].append({
                "zone_id": zone,
                "location_name": clean_payload["location_name"],
                "city": clean_payload["city"],
                "zone_type": clean_payload["zone_type"],
                "latitude": clean_payload["latitude"],
                "longitude": clean_payload["longitude"],
                "priority_rank": clean_payload["priority"]["priority_rank"],
                "priority_score": clean_payload["priority"]["priority_score"],
                "priority_level": clean_payload["priority"]["priority_level"],
                "predicted_risk_probability": clean_payload["risk_analysis"]["predicted_risk_probability"],
                "risk_label": clean_payload["risk_analysis"]["risk_label"],
                "congestion": clean_payload["telemetry"]["congestion"],
                "average_speed_kmh": clean_payload["telemetry"]["average_speed_kmh"],
                "vehicle_density": clean_payload["telemetry"]["vehicle_density"],
                "recommended_green_seconds": clean_payload["signal_optimization"]["recommended_green_seconds"],
                "co2_savings_kg": clean_payload["environmental_impact"]["potential_savings_kg_per_week"]
            })

        # Save /api/v1/zones/{zone}/all.json (complete 52-week time-series)
        with open(f"{zone_dir}/all.json", "w") as f:
            json.dump({
                "zone_id": zone,
                "location_name": str(zone_df.iloc[0].get("location_name", zone)),
                "city": str(zone_df.iloc[0].get("city", "")),
                "total_weeks": len(all_weeks_data),
                "timeline": all_weeks_data
            }, f, indent=2)

    # 5. Generate Week-wise City Leaderboard Endpoints (/api/v1/weeks/{week}.json)
    print("[5/5] Generating city-wide weekly leaderboard endpoints & summary...")
    for week, zones_list in week_leaderboards.items():
        # Sort by priority rank ascending
        sorted_zones = sorted(zones_list, key=lambda z: z["priority_rank"])
        with open(f"{api_dir}/weeks/{week}.json", "w") as f:
            json.dump({
                "week": week,
                "total_zones": len(sorted_zones),
                "rankings": sorted_zones
            }, f, indent=2)

    # Generate Overall Summary (/api/v1/summary.json)
    latest_week = max(weeks)
    latest_rankings = week_leaderboards[latest_week]
    critical_count = sum(1 for z in latest_rankings if z["priority_level"] == "CRITICAL")
    high_count = sum(1 for z in latest_rankings if z["priority_level"] == "HIGH")
    total_co2_savings = sum(z["co2_savings_kg"] for z in latest_rankings)

    with open(f"{api_dir}/summary.json", "w") as f:
        json.dump({
            "status": "OPERATIONAL",
            "total_zones": len(zones),
            "latest_week": latest_week,
            "weeks_range": [min(weeks), max(weeks)],
            "latest_week_stats": {
                "critical_zones": critical_count,
                "high_risk_zones": high_count,
                "city_total_co2_savings_kg_per_week": round(total_co2_savings, 2)
            }
        }, f, indent=2)

    # 6. Push Chicago Civilian Road Safety Radar Endpoints
    try:
        from backend.civilian_api_builder import push_civilian_safety_radar_to_api
        push_civilian_safety_radar_to_api()
    except Exception as e:
        print(f"[!] Warning: Civilian safety radar API generation: {e}")

    print(f"[+] Complete! Rich REST API generated at ./{api_dir}/")


if __name__ == "__main__":
    build_static_api()