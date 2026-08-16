"""
================================================================================
ROAD SENSE AI - INTERVENTION RECOMMENDATION ENGINE
MODULE: RULE-BASED MUNICIPAL INTERVENTION & PRIORITY EXPLANATION
================================================================================

This module transforms risk predictions and priority scores into actionable
municipal intervention recommendations and human-readable explanations.

ARCHITECTURAL PRINCIPLES:
1. TRANSPARENT RULES: Every recommendation is derived from explicit, auditable
   conditional logic — not opaque model predictions.
2. MULTI-SIGNAL REASONING: Recommendations consider risk, congestion, speed,
   violations, trends, population exposure, and road condition together.
3. PRIORITY EXPLANATIONS: Each zone receives a structured explanation of WHY
   it is ranked where it is, listing the top contributing factors.
4. HONEST LABELING: Recommendations are labeled as "recommended interventions"
   not "guaranteed optimal actions."
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


# ==============================================================================
# 1. INTERVENTION RULE DEFINITIONS
# ==============================================================================

INTERVENTION_RULES = [
    {
        "id": "SIGNAL_OPTIMIZATION",
        "name": "Traffic Signal Optimization",
        "icon": "🚦",
        "condition": lambda r: r.get("congestion", 0) >= 60 and r.get("average_speed", 99) < 35,
        "severity": "HIGH",
        "description": "High congestion with low throughput speed indicates signal timing inefficiency.",
        "action": "Review and optimize signal timing at major intersections in this zone."
    },
    {
        "id": "SPEED_CALMING",
        "name": "Speed Calming & Enforcement",
        "icon": "🏎️",
        "condition": lambda r: r.get("average_speed", 0) > 55 and r.get("risk_score", 0) > 50,
        "severity": "HIGH",
        "description": "High average speed combined with elevated risk score suggests speed-related hazards.",
        "action": "Deploy speed calming measures (speed bumps, radar signs) and increase enforcement patrols."
    },
    {
        "id": "VIOLATION_ENFORCEMENT",
        "name": "Red-Light Violation Enforcement",
        "icon": "🚨",
        "condition": lambda r: r.get("red_light_violations", 0) >= 8,
        "severity": "HIGH",
        "description": "Elevated red-light violation count indicates intersection compliance failure.",
        "action": "Install or activate red-light enforcement cameras. Review signal visibility and timing."
    },
    {
        "id": "CONGESTION_MANAGEMENT",
        "name": "Congestion Management & Diversion",
        "icon": "🚗",
        "condition": lambda r: r.get("congestion", 0) >= 70 and r.get("vehicle_density", 0) > 60,
        "severity": "CRITICAL",
        "description": "Severe congestion with high vehicle density requires active demand management.",
        "action": "Activate dynamic message signs for route diversion. Consider congestion pricing or access restrictions."
    },
    {
        "id": "PEDESTRIAN_SAFETY",
        "name": "Pedestrian Crossing Intervention",
        "icon": "🚶",
        "condition": lambda r: r.get("population_density", 0) > 10000 and r.get("congestion", 0) > 50,
        "severity": "MEDIUM",
        "description": "High population density zone with significant traffic congestion creates pedestrian exposure risk.",
        "action": "Install protected pedestrian crossings, improve sidewalk infrastructure, and add pedestrian countdown signals."
    },
    {
        "id": "ROAD_MAINTENANCE",
        "name": "Road Maintenance Inspection",
        "icon": "🛣️",
        "condition": lambda r: r.get("road_condition", "Good") in ["Moderate", "Poor"] and r.get("risk_score", 0) > 40,
        "severity": "MEDIUM",
        "description": "Deteriorated road surface combined with elevated risk indicates infrastructure maintenance need.",
        "action": "Schedule road surface inspection and prioritize pothole repair, marking refresh, and drainage check."
    },
    {
        "id": "TREND_URGENT_INSPECTION",
        "name": "Urgent Safety Inspection",
        "icon": "⚠️",
        "condition": lambda r: r.get("temporal_trend_score", 50) > 75 and r.get("risk_score", 0) > 60,
        "severity": "CRITICAL",
        "description": "Risk is actively worsening (high temporal trend) with already elevated risk score.",
        "action": "Deploy immediate field inspection team. Identify root cause of deteriorating safety conditions."
    },
    {
        "id": "EVENT_TRAFFIC_PLAN",
        "name": "Special Event Traffic Plan",
        "icon": "🎪",
        "condition": lambda r: r.get("special_event", 0) == 1,
        "severity": "MEDIUM",
        "description": "Active special event generating additional traffic demand.",
        "action": "Activate event traffic management plan. Deploy traffic marshals and temporary signage."
    },
    {
        "id": "MONITORING_UPGRADE",
        "name": "Enhanced Monitoring & CCTV Coverage",
        "icon": "📹",
        "condition": lambda r: r.get("risk_score", 0) > 70 and r.get("population_exposure_score", 0) > 60,
        "severity": "MEDIUM",
        "description": "High-risk zone with significant population exposure warrants improved surveillance.",
        "action": "Install additional CCTV cameras and integrate with traffic management center for real-time monitoring."
    },
    {
        "id": "BASELINE_MONITORING",
        "name": "Routine Monitoring",
        "icon": "📊",
        "condition": lambda r: r.get("risk_score", 0) < 30,
        "severity": "LOW",
        "description": "Zone currently within acceptable risk parameters.",
        "action": "Continue routine monitoring. No immediate intervention required."
    },
]


# ==============================================================================
# 2. GENERATE INTERVENTIONS FOR A ZONE
# ==============================================================================

def generate_interventions(zone_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Evaluates all intervention rules against a zone's current data
    and returns a list of triggered recommendations, sorted by severity.
    """
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    triggered = []

    for rule in INTERVENTION_RULES:
        try:
            if rule["condition"](zone_data):
                triggered.append({
                    "id": rule["id"],
                    "name": rule["name"],
                    "icon": rule["icon"],
                    "severity": rule["severity"],
                    "description": rule["description"],
                    "action": rule["action"]
                })
        except (KeyError, TypeError):
            continue

    # Sort by severity (CRITICAL first)
    triggered.sort(key=lambda x: severity_order.get(x["severity"], 99))

    # If nothing triggered, add baseline monitoring
    if not triggered:
        triggered.append({
            "id": "BASELINE_MONITORING",
            "name": "Routine Monitoring",
            "icon": "📊",
            "severity": "LOW",
            "description": "No specific risk indicators triggered.",
            "action": "Continue routine monitoring."
        })

    return triggered


# ==============================================================================
# 3. GENERATE PRIORITY EXPLANATION FOR A ZONE
# ==============================================================================

def generate_priority_explanation(zone_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produces a structured explanation of WHY a zone has its priority ranking.
    Returns contributing factors sorted by impact magnitude.
    """
    contributors = []

    # Risk score contribution
    risk = zone_data.get("risk_score", 0)
    if risk >= 70:
        contributors.append({"factor": "Risk Score", "icon": "🔴", "value": f"{risk:.0f}/100", "impact": "HIGH", "detail": "Model-estimated risk score is in the critical range", "weight": risk})
    elif risk >= 40:
        contributors.append({"factor": "Risk Score", "icon": "🟡", "value": f"{risk:.0f}/100", "impact": "MEDIUM", "detail": "Moderate model-estimated risk", "weight": risk})
    else:
        contributors.append({"factor": "Risk Score", "icon": "🟢", "value": f"{risk:.0f}/100", "impact": "LOW", "detail": "Risk score within acceptable range", "weight": risk})

    # Congestion contribution
    cong = zone_data.get("congestion", 0)
    if cong >= 65:
        contributors.append({"factor": "Congestion", "icon": "🚗", "value": f"{cong:.0f}/100", "impact": "HIGH", "detail": f"Congestion index at {cong:.0f} — severe traffic flow degradation", "weight": cong})
    elif cong >= 40:
        contributors.append({"factor": "Congestion", "icon": "🚗", "value": f"{cong:.0f}/100", "impact": "MEDIUM", "detail": f"Moderate congestion at {cong:.0f}", "weight": cong * 0.7})

    # Population exposure
    pop = zone_data.get("population_exposure_score", 0)
    if pop >= 60:
        contributors.append({"factor": "Population Exposure", "icon": "👥", "value": f"{pop:.0f}/100", "impact": "HIGH", "detail": f"High population density zone — {zone_data.get('population_density', 'N/A')} residents/km²", "weight": pop})
    elif pop >= 30:
        contributors.append({"factor": "Population Exposure", "icon": "👥", "value": f"{pop:.0f}/100", "impact": "MEDIUM", "detail": "Moderate population exposure", "weight": pop * 0.7})

    # Vehicle exposure
    veh = zone_data.get("vehicle_exposure_score", 0)
    if veh >= 60:
        contributors.append({"factor": "Vehicle Density", "icon": "🚙", "value": f"{veh:.0f}/100", "impact": "HIGH", "detail": "High vehicle density increases collision exposure", "weight": veh})

    # Temporal trend
    trend = zone_data.get("temporal_trend_score", 50)
    if trend >= 70:
        contributors.append({"factor": "Worsening Trend", "icon": "📈", "value": f"{trend:.0f}/100", "impact": "HIGH", "detail": "4-week congestion trend is actively worsening — conditions deteriorating", "weight": trend})
    elif trend >= 55:
        contributors.append({"factor": "Rising Trend", "icon": "📈", "value": f"{trend:.0f}/100", "impact": "MEDIUM", "detail": "Slight upward trend in congestion over past 4 weeks", "weight": trend * 0.6})
    elif trend <= 30:
        contributors.append({"factor": "Improving Trend", "icon": "📉", "value": f"{trend:.0f}/100", "impact": "POSITIVE", "detail": "Conditions are improving over past 4 weeks", "weight": -trend})

    # Speed anomaly
    speed = zone_data.get("average_speed", 50)
    if speed < 25:
        contributors.append({"factor": "Speed Anomaly", "icon": "🐌", "value": f"{speed:.0f} km/h", "impact": "HIGH", "detail": f"Average speed of {speed:.0f} km/h indicates near-gridlock conditions", "weight": 100 - speed})

    # Violations
    violations = zone_data.get("red_light_violations", 0)
    if violations >= 10:
        contributors.append({"factor": "Violation Count", "icon": "🚨", "value": f"{violations}", "impact": "HIGH", "detail": f"{violations} red-light violations recorded — intersection safety concern", "weight": violations * 5})

    # Sort by weight (impact magnitude)
    contributors.sort(key=lambda x: abs(x.get("weight", 0)), reverse=True)

    # Determine overall assessment
    priority_level = zone_data.get("priority_level", "MODERATE")
    priority_score = zone_data.get("priority_score", 50)

    return {
        "priority_level": priority_level,
        "priority_score": round(priority_score, 1),
        "top_contributors": contributors[:5],
        "total_factors": len(contributors),
        "assessment": _generate_assessment_text(priority_level, contributors)
    }


def _generate_assessment_text(level: str, contributors: List[Dict]) -> str:
    """Generates a one-paragraph human-readable assessment."""
    if not contributors:
        return "Insufficient data for assessment."

    high_factors = [c["factor"] for c in contributors if c.get("impact") == "HIGH"]
    medium_factors = [c["factor"] for c in contributors if c.get("impact") == "MEDIUM"]

    if level == "CRITICAL":
        drivers = ", ".join(high_factors[:3]) if high_factors else "multiple elevated indicators"
        return f"This zone requires URGENT attention. Primary risk drivers: {drivers}. Immediate field inspection and intervention planning recommended."
    elif level == "HIGH":
        drivers = ", ".join(high_factors[:2]) if high_factors else "elevated risk indicators"
        return f"This zone shows significant risk elevation driven by {drivers}. Proactive intervention should be scheduled within the current planning cycle."
    elif level == "MODERATE":
        all_factors = high_factors + medium_factors
        drivers = ", ".join(all_factors[:2]) if all_factors else "moderate indicators"
        return f"This zone has moderate risk levels with contributing factors: {drivers}. Continue monitoring and schedule review if conditions worsen."
    else:
        return "This zone is currently within acceptable safety parameters. Routine monitoring is sufficient."


# ==============================================================================
# 4. BATCH PROCESSING
# ==============================================================================

def generate_batch_interventions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame of zones and adds intervention recommendations
    and priority explanations as new columns.
    """
    interventions_list = []
    explanations_list = []

    for _, row in df.iterrows():
        zone_data = row.to_dict()
        interventions = generate_interventions(zone_data)
        explanation = generate_priority_explanation(zone_data)

        # Store primary intervention
        primary = interventions[0] if interventions else {}
        interventions_list.append({
            "primary_intervention": primary.get("name", "None"),
            "intervention_severity": primary.get("severity", "LOW"),
            "intervention_action": primary.get("action", "Continue monitoring"),
            "intervention_count": len(interventions)
        })

        explanations_list.append({
            "priority_assessment": explanation.get("assessment", ""),
            "top_factor_1": explanation["top_contributors"][0]["factor"] if len(explanation["top_contributors"]) > 0 else "",
            "top_factor_2": explanation["top_contributors"][1]["factor"] if len(explanation["top_contributors"]) > 1 else "",
            "top_factor_3": explanation["top_contributors"][2]["factor"] if len(explanation["top_contributors"]) > 2 else "",
        })

    interventions_df = pd.DataFrame(interventions_list)
    explanations_df = pd.DataFrame(explanations_list)

    result = pd.concat([df.reset_index(drop=True), interventions_df, explanations_df], axis=1)
    return result
