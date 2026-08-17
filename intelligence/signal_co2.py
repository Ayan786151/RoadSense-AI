"""Adaptive signal timing calculator and CO2 impact estimator."""

import numpy as np
from typing import Dict, Any


# ==============================================================================
# ADAPTIVE SIGNAL TIMING
# ==============================================================================

# Base green time by zone archetype (seconds)
BASE_GREEN_TIMES = {
    "Commercial_Downtown": 40,
    "Highway_Junction": 50,
    "University_District": 35,
    "Industrial_Corridor": 45,
    "Residential": 30,
    "Suburban_LowDensity": 25,
}

def compute_optimal_signal_timing(
    congestion: float,
    vehicle_density: float,
    average_speed: float,
    zone_type: str = "Residential",
    special_event: int = 0,
    weather: str = "Normal",
) -> Dict[str, Any]:
    """
    Computes recommended green-light duration based on current conditions.
    
    Logic:
    - Start with archetype-specific base green time
    - Scale up when congestion is high (vehicles need more clearance time)
    - Scale up when density is high (more vehicles queued)
    - Scale down when speed is high (vehicles clearing faster)
    - Boost for events and bad weather
    - Clamp to [15, 120] seconds (practical signal range)
    
    Returns dict with recommended timings and the reasoning.
    """
    base = BASE_GREEN_TIMES.get(zone_type, 35)
    
    # Congestion factor: 0-100 mapped to 1.0-1.8x multiplier
    congestion_factor = 1.0 + (congestion / 100.0) * 0.8
    
    # Density factor: high density = longer green needed
    density_factor = 1.0 + (min(vehicle_density, 100) / 100.0) * 0.4
    
    # Speed factor: if traffic is already flowing fast, less green needed
    speed_factor = max(0.7, 1.0 - (average_speed / 80.0) * 0.3)
    
    # Event and weather boosts
    event_boost = 1.15 if special_event else 1.0
    weather_boost = {"Normal": 1.0, "Light Rain": 1.08, "Heavy Rain": 1.18}.get(weather, 1.0)
    
    recommended = base * congestion_factor * density_factor * speed_factor * event_boost * weather_boost
    recommended = int(np.clip(recommended, 15, 120))
    
    # Compute change from default
    change = recommended - base
    change_pct = (change / base) * 100 if base > 0 else 0
    
    # Determine urgency
    if change_pct > 40:
        urgency = "CRITICAL"
        reason = "Severe congestion requires significantly extended green phase"
    elif change_pct > 20:
        urgency = "HIGH"
        reason = "Elevated traffic density needs extended clearance time"
    elif change_pct > 5:
        urgency = "MODERATE"
        reason = "Slight adjustment for current conditions"
    elif change_pct < -10:
        urgency = "OPTIMIZE"
        reason = "Low traffic — reduce green time to improve cross-traffic flow"
    else:
        urgency = "NOMINAL"
        reason = "Current default timing is adequate"
    
    return {
        "base_green_seconds": base,
        "recommended_green_seconds": recommended,
        "change_seconds": change,
        "change_percent": round(change_pct, 1),
        "urgency": urgency,
        "reason": reason,
        "factors": {
            "congestion_factor": round(congestion_factor, 2),
            "density_factor": round(density_factor, 2),
            "speed_factor": round(speed_factor, 2),
            "event_boost": event_boost,
            "weather_boost": weather_boost,
        }
    }


# ==============================================================================
# CO2 IMPACT ESTIMATION
# ==============================================================================

# Average CO2 emission rates (kg CO2 per vehicle per hour)
# Source: CPCB India urban traffic emission factors
IDLE_EMISSION_RATE = 2.3      # kg CO2/vehicle/hour while idling
MOVING_EMISSION_BASE = 0.12   # kg CO2/vehicle/km at optimal speed (40-60 km/h)
CONGESTED_EMISSION_MULT = 2.5 # Multiplier for stop-and-go traffic

def estimate_co2_impact(
    vehicle_density: float,
    congestion: float,
    average_speed: float,
    population_density: int = 5000,
    zone_area_km2: float = 2.0,
) -> Dict[str, Any]:
    """
    Estimates weekly CO2 emissions and potential savings from optimization.
    
    Assumptions:
    - Vehicle density is per-km, zone covers ~2 km² of road network
    - Vehicles spend ~2 hours in zone per weekday (commute pattern)
    - Optimization can reduce idle time by 15-30% based on signal improvements
    """
    # Estimated vehicles in zone
    estimated_vehicles = vehicle_density * zone_area_km2 * 10  # rough scaling
    
    # Hours spent in zone per week (5 weekdays × 2 peak hours + 2 weekend hours)
    weekly_hours = 12
    
    # Idle fraction based on congestion
    idle_fraction = min(congestion / 100.0, 0.95)
    moving_fraction = 1.0 - idle_fraction
    
    # Speed-based emission multiplier (stop-and-go is worse)
    if average_speed < 15:
        speed_mult = CONGESTED_EMISSION_MULT
    elif average_speed < 30:
        speed_mult = 1.8
    elif average_speed < 50:
        speed_mult = 1.0
    else:
        speed_mult = 0.9  # Efficient cruising
    
    # Weekly CO2 (kg)
    idle_co2 = estimated_vehicles * idle_fraction * IDLE_EMISSION_RATE * weekly_hours
    moving_co2 = estimated_vehicles * moving_fraction * MOVING_EMISSION_BASE * average_speed * weekly_hours * speed_mult
    total_co2_kg = idle_co2 + moving_co2
    
    # Potential savings from signal optimization (15-30% idle time reduction)
    optimization_factor = 0.20 + (congestion / 100.0) * 0.10  # 20-30% for high congestion
    potential_savings_kg = idle_co2 * optimization_factor
    
    # Social impact metrics
    trees_equivalent = potential_savings_kg / 21.0  # 1 tree absorbs ~21 kg CO2/year, weekly
    fuel_saved_liters = potential_savings_kg / 2.31  # 1 liter petrol = 2.31 kg CO2
    
    # Population affected
    citizens_impacted = int(population_density * zone_area_km2)
    
    return {
        "total_co2_kg_per_week": round(total_co2_kg, 1),
        "total_co2_tonnes_per_year": round(total_co2_kg * 52 / 1000, 2),
        "idle_co2_kg": round(idle_co2, 1),
        "moving_co2_kg": round(moving_co2, 1),
        "potential_savings_kg_per_week": round(potential_savings_kg, 1),
        "potential_savings_tonnes_per_year": round(potential_savings_kg * 52 / 1000, 2),
        "trees_equivalent_per_year": int(trees_equivalent * 52),
        "fuel_saved_liters_per_week": round(fuel_saved_liters, 1),
        "citizens_impacted": citizens_impacted,
        "optimization_factor_pct": round(optimization_factor * 100, 1),
    }
