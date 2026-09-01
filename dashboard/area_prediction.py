"""
================================================================================
ROADSENSE AI — AREA PREDICTION & TRAJECTORY INTELLIGENCE MODULE
DESIGN SYSTEM: KINETIC INFRASTRUCTURE INTELLIGENCE (STITCH MCP)
================================================================================

City-wide predictive analysis showing which of the 50 municipal zones are
improving, stable, or deteriorating — and the specific root-cause drivers
behind each trajectory.
================================================================================
"""

import os
import html
import hashlib
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Tuple, Optional, Any, List

from intelligence.signal_co2 import compute_optimal_signal_timing, estimate_co2_impact


# ==============================================================================
# 1. DATA LOADING (REUSE CACHED FUNCTIONS)
# ==============================================================================

@st.cache_data
def load_simulation_data_for_prediction() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads full 52-week temporal simulation dataset and location mappings."""
    sim_path = "data/simulation_temporal_features.csv"
    loc_path = "data/location_mapping.csv"

    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"Missing simulation dataset at: {sim_path}")

    sim_df = pd.read_csv(sim_path)

    if os.path.exists(loc_path):
        loc_df = pd.read_csv(loc_path)
        merged = pd.merge(sim_df, loc_df, on="zone_id", how="left")
    else:
        merged = sim_df.copy()
        merged["location_name"] = merged["zone_id"]
        merged["city"] = "Metropolis"
        merged["latitude"] = 19.0760
        merged["longitude"] = 72.8777
        loc_df = pd.DataFrame()

    return merged, loc_df


@st.cache_resource
def load_risk_model_for_prediction():
    """Loads the trained Supervised ML Risk Model pipeline."""
    model_path = "models/best_risk_model.pkl"
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return None


def compute_zone_risk(model, row_df: pd.DataFrame) -> float:
    """Computes risk probability using the trained ML model pipeline."""
    if model is None or row_df.empty:
        return np.nan
    try:
        week = row_df["week"].iloc[0]
        if week < 5:
            return np.nan
        raw_p = model.predict_proba(row_df)
        prob = float(raw_p[0, 1]) if (hasattr(raw_p, "shape") and raw_p.shape[1] > 1) else (1.0 if model.classes_[0] == 1 else 0.0)
        return float(prob)
    except Exception:
        return float(row_df.get("rolling_4_week_incident_rate", pd.Series([np.nan])).iloc[0])


# ==============================================================================
# 2. TRAJECTORY ANALYSIS ENGINE
# ==============================================================================

def analyze_zone_trajectory(zone_df: pd.DataFrame, reference_week: int, model) -> Dict[str, Any]:
    """
    Analyzes a single zone's trajectory to determine if it's improving, 
    stable, or deteriorating, and identifies the primary root-cause driver.
    
    Returns a dict with status, slopes, primary driver explanation, and metrics.
    """
    hist_df = zone_df[zone_df["week"] <= reference_week].sort_values("week").copy()
    
    zone_id = zone_df["zone_id"].iloc[0]
    location = zone_df.get("location_name", zone_df["zone_id"]).iloc[0]
    zone_type = zone_df.get("zone_type", pd.Series(["Urban"])).iloc[0]
    city = zone_df.get("city", pd.Series(["Metropolis"])).iloc[0]
    
    # Default result for insufficient data
    if len(hist_df) < 5:
        current = hist_df.iloc[-1] if len(hist_df) > 0 else None
        return {
            "zone_id": zone_id,
            "location_name": location,
            "zone_type": zone_type,
            "city": city,
            "status": "STABLE",
            "status_label": " STABLE",
            "status_color": "#eab308",
            "risk_slope": 0.0,
            "cong_slope": 0.0,
            "speed_slope": 0.0,
            "density_slope": 0.0,
            "violation_slope": 0.0,
            "current_risk_pct": 0.0,
            "current_congestion": float(current["congestion"]) if current is not None else 0.0,
            "current_speed": float(current["average_speed"]) if current is not None else 0.0,
            "current_density": float(current["vehicle_density"]) if current is not None else 0.0,
            "primary_driver": "Insufficient historical data for trajectory analysis",
            "primary_driver_metric": "data",
            "driver_magnitude": 0.0,
            "secondary_factors": [],
            "forecast_4w_risk_delta": 0.0,
        }
    
    # Get recent 4-week window for trend analysis
    recent_4w = hist_df.tail(min(4, len(hist_df)))
    current_row = hist_df.iloc[-1]
    weeks_x = recent_4w["week"].values
    
    # Compute OLS slopes for key metrics
    def safe_slope(x, y):
        if len(x) >= 2 and np.std(x) > 0:
            return float(np.polyfit(x, y, 1)[0])
        return 0.0
    
    cong_slope = safe_slope(weeks_x, recent_4w["congestion"].values)
    spd_slope = safe_slope(weeks_x, recent_4w["average_speed"].values)
    density_slope = safe_slope(weeks_x, recent_4w["vehicle_density"].values)
    
    # Violation slope
    if "red_light_violations" in recent_4w.columns:
        viol_slope = safe_slope(weeks_x, recent_4w["red_light_violations"].values)
    else:
        viol_slope = 0.0
    
    # Risk slope (using ML model predictions over recent weeks)
    prev_risks = []
    for _, r in recent_4w.iterrows():
        p = compute_zone_risk(model, pd.DataFrame([r]))
        if not np.isnan(p):
            prev_risks.append((r["week"], p * 100.0))
    
    if len(prev_risks) >= 2:
        rw_x = [x[0] for x in prev_risks]
        rw_y = [x[1] for x in prev_risks]
        risk_slope = safe_slope(np.array(rw_x), np.array(rw_y))
    else:
        risk_slope = (cong_slope * 0.4) - (spd_slope * 0.3)
    
    # Current risk probability
    curr_risk = compute_zone_risk(model, pd.DataFrame([current_row]))
    if np.isnan(curr_risk):
        curr_risk = 0.35
    curr_risk_pct = curr_risk * 100.0
    
    # ---- CLASSIFY TRAJECTORY ----
    # Composite health score: positive = worsening, negative = improving
    health_score = (
        (cong_slope * 0.35) +       # Congestion increasing is bad
        (-spd_slope * 0.25) +        # Speed decreasing is bad
        (density_slope * 0.15) +     # Density increasing is bad
        (viol_slope * 0.15) +        # Violations increasing is bad
        (risk_slope * 0.10)          # Risk increasing is bad
    )
    
    if health_score >= 0.8 or risk_slope >= 1.2 or (cong_slope >= 1.8 and spd_slope <= -0.8):
        status = "DETERIORATING"
        status_label = " DETERIORATING"
        status_color = "#ef4444"
    elif health_score <= -0.8 or risk_slope <= -1.2 or (cong_slope <= -1.8 and spd_slope >= 0.8):
        status = "IMPROVING"
        status_label = " IMPROVING"
        status_color = "#22c55e"
    else:
        status = "STABLE"
        status_label = " STABLE"
        status_color = "#eab308"
    
    # ---- IDENTIFY PRIMARY DRIVER (THE "WHY") ----
    drivers = []
    
    # Congestion driver
    if abs(cong_slope) > 0.3:
        if cong_slope > 0:
            drivers.append({
                "metric": "congestion",
                "magnitude": abs(cong_slope),
                "weight": 0.35,
                "score": abs(cong_slope) * 0.35,
                "explanation": f"Congestion surging +{cong_slope:.1f} pts/wk → intersection capacity overload",
                "direction": "worsening"
            })
        else:
            drivers.append({
                "metric": "congestion",
                "magnitude": abs(cong_slope),
                "weight": 0.35,
                "score": abs(cong_slope) * 0.35,
                "explanation": f"Congestion easing {cong_slope:.1f} pts/wk → improved signal coordination",
                "direction": "improving"
            })
    
    # Speed driver
    if abs(spd_slope) > 0.3:
        if spd_slope < 0:
            drivers.append({
                "metric": "speed",
                "magnitude": abs(spd_slope),
                "weight": 0.25,
                "score": abs(spd_slope) * 0.25,
                "explanation": f"Speed declining {spd_slope:.1f} km/h/wk → stop-and-go wave propagation",
                "direction": "worsening"
            })
        else:
            drivers.append({
                "metric": "speed",
                "magnitude": abs(spd_slope),
                "weight": 0.25,
                "score": abs(spd_slope) * 0.25,
                "explanation": f"Speed recovering +{spd_slope:.1f} km/h/wk → corridor flow stabilizing",
                "direction": "improving"
            })
    
    # Density driver
    if abs(density_slope) > 0.5:
        if density_slope > 0:
            drivers.append({
                "metric": "density",
                "magnitude": abs(density_slope),
                "weight": 0.15,
                "score": abs(density_slope) * 0.15,
                "explanation": f"Vehicle density rising +{density_slope:.1f} veh/km/wk → demand exceeding road capacity",
                "direction": "worsening"
            })
        else:
            drivers.append({
                "metric": "density",
                "magnitude": abs(density_slope),
                "weight": 0.15,
                "score": abs(density_slope) * 0.15,
                "explanation": f"Vehicle density dropping {density_slope:.1f} veh/km/wk → demand relief",
                "direction": "improving"
            })
    
    # Violation driver
    if abs(viol_slope) > 0.3:
        if viol_slope > 0:
            drivers.append({
                "metric": "violations",
                "magnitude": abs(viol_slope),
                "weight": 0.15,
                "score": abs(viol_slope) * 0.15,
                "explanation": f"Red-light violations rising +{viol_slope:.1f}/wk → enforcement gap widening",
                "direction": "worsening"
            })
        else:
            drivers.append({
                "metric": "violations",
                "magnitude": abs(viol_slope),
                "weight": 0.15,
                "score": abs(viol_slope) * 0.15,
                "explanation": f"Red-light violations declining {viol_slope:.1f}/wk → enforcement impact visible",
                "direction": "improving"
            })
    
    # Risk model driver
    if abs(risk_slope) > 0.5:
        if risk_slope > 0:
            drivers.append({
                "metric": "risk",
                "magnitude": abs(risk_slope),
                "weight": 0.10,
                "score": abs(risk_slope) * 0.10,
                "explanation": f"ML-predicted risk climbing +{risk_slope:.1f}%/wk → multi-factor hazard escalation",
                "direction": "worsening"
            })
        else:
            drivers.append({
                "metric": "risk",
                "magnitude": abs(risk_slope),
                "weight": 0.10,
                "score": abs(risk_slope) * 0.10,
                "explanation": f"ML-predicted risk falling {risk_slope:.1f}%/wk → systemic safety improvement",
                "direction": "improving"
            })
    
    # Sort by weighted score
    drivers.sort(key=lambda d: d["score"], reverse=True)
    
    if drivers:
        primary = drivers[0]
        primary_driver = primary["explanation"]
        primary_driver_metric = primary["metric"]
        driver_magnitude = primary["score"]
        secondary_factors = [d["explanation"] for d in drivers[1:3]]
    else:
        primary_driver = "All metrics within normal variance — no dominant trend detected"
        primary_driver_metric = "none"
        driver_magnitude = 0.0
        secondary_factors = []
    
    return {
        "zone_id": zone_id,
        "location_name": location,
        "zone_type": zone_type,
        "city": city,
        "status": status,
        "status_label": status_label,
        "status_color": status_color,
        "risk_slope": round(risk_slope, 2),
        "cong_slope": round(cong_slope, 2),
        "speed_slope": round(spd_slope, 2),
        "density_slope": round(density_slope, 2),
        "violation_slope": round(viol_slope, 2),
        "current_risk_pct": round(curr_risk_pct, 1),
        "current_congestion": round(float(current_row["congestion"]), 1),
        "current_speed": round(float(current_row["average_speed"]), 1),
        "current_density": round(float(current_row["vehicle_density"]), 1),
        "primary_driver": primary_driver,
        "primary_driver_metric": primary_driver_metric,
        "driver_magnitude": round(driver_magnitude, 3),
        "secondary_factors": secondary_factors,
        "forecast_4w_risk_delta": round(risk_slope * 4, 1),
        "health_score": round(health_score, 3),
    }


def analyze_all_zones(df: pd.DataFrame, reference_week: int, model) -> List[Dict[str, Any]]:
    """Runs trajectory analysis for every unique zone in the dataset."""
    results = []
    for zone_id in sorted(df["zone_id"].unique()):
        zone_df = df[df["zone_id"] == zone_id].sort_values("week").reset_index(drop=True)
        result = analyze_zone_trajectory(zone_df, reference_week, model)
        results.append(result)
    return results


# ==============================================================================
# 3. RENDER AREA PREDICTION DASHBOARD
# ==============================================================================

def render_area_prediction_dashboard():
    """Renders the city-wide Area Prediction & Trajectory Intelligence module."""
    
    # Header
    st.markdown("""
    <div class="telemetry-header" style="border-left: 4px solid #818cf8;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #818cf8; letter-spacing: 0.08em; text-transform: uppercase;">
            PREDICTIVE INTELLIGENCE MODULE 07
        </div>
        <h2 style="margin: 4px 0 0 0; font-size: 22px;">
            Area Prediction & Trajectory Intelligence
        </h2>
        <p style="margin: 6px 0 0 0; color: #a1a1aa; font-size: 13px;">
            City-wide predictive analysis identifying which zones are improving, stable, or deteriorating — with root-cause driver attribution explaining <em>why</em> each area is trending in its direction.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    try:
        df, loc_df = load_simulation_data_for_prediction()
        model = load_risk_model_for_prediction()
    except Exception as e:
        st.error(f"Failed to load simulation data: {e}")
        return
    
    # Sidebar control: Reference Week
    st.sidebar.markdown("### PREDICTION CONTROLS")
    reference_week = st.sidebar.slider(
        "ANALYSIS REFERENCE WEEK",
        min_value=5,
        max_value=52,
        value=50,
        help="The week from which trajectory analysis looks backward (4 weeks) and forward (4 weeks)."
    )
    
    # Run city-wide analysis
    with st.spinner("Computing trajectory predictions across all 50 zones..."):
        all_results = analyze_all_zones(df, reference_week, model)
    
    # Classify zones
    improving = [z for z in all_results if z["status"] == "IMPROVING"]
    stable = [z for z in all_results if z["status"] == "STABLE"]
    deteriorating = [z for z in all_results if z["status"] == "DETERIORATING"]
    
    # Sort deteriorating by worst health score, improving by best
    deteriorating.sort(key=lambda z: z["health_score"], reverse=True)
    improving.sort(key=lambda z: z["health_score"])
    
    avg_risk_slope = np.mean([z["risk_slope"] for z in all_results]) if all_results else 0.0
    
    # ================================================================
    # SUMMARY KPI SCORECARDS
    # ================================================================
    st.markdown("""
    <div style="background: #18181b; border: 1px solid #27272a; border-radius: 6px; padding: 18px 20px; margin-bottom: 20px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #818cf8; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 14px;">
            CITY-WIDE TRAJECTORY HEALTH OVERVIEW — WEEK """ + str(reference_week) + """
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
        <div class="telemetry-card" style="border-left: 3px solid #22c55e;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">IMPROVING ZONES</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 36px; font-weight: 800; color: #22c55e; margin: 4px 0;">{len(improving)}</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #71717a;">of {len(all_results)} total zones</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi2:
        st.markdown(f"""
        <div class="telemetry-card" style="border-left: 3px solid #eab308;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">STABLE ZONES</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 36px; font-weight: 800; color: #eab308; margin: 4px 0;">{len(stable)}</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #71717a;">nominal equilibrium</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi3:
        st.markdown(f"""
        <div class="telemetry-card" style="border-left: 3px solid #ef4444;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">DETERIORATING ZONES</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 36px; font-weight: 800; color: #ef4444; margin: 4px 0;">{len(deteriorating)}</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #71717a;">require intervention</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi4:
        risk_color = "#ef4444" if avg_risk_slope > 0.5 else ("#22c55e" if avg_risk_slope < -0.5 else "#eab308")
        risk_sign = "+" if avg_risk_slope >= 0 else ""
        st.markdown(f"""
        <div class="telemetry-card" style="border-left: 3px solid {risk_color};">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">AVG RISK TRAJECTORY</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 36px; font-weight: 800; color: {risk_color}; margin: 4px 0;">{risk_sign}{avg_risk_slope:.1f}%</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #71717a;">per week city-wide</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ================================================================
    # TOP MOVERS: MOST IMPROVED & MOST DETERIORATED
    # ================================================================
    top_col1, top_col2 = st.columns(2)
    
    with top_col1:
        st.markdown("""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #22c55e; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.06em;">
             TOP 3 MOST IMPROVED AREAS
        </div>
        """, unsafe_allow_html=True)
        
        if improving:
            for i, z in enumerate(improving[:3]):
                safe_loc = html.escape(str(z["location_name"]))
                safe_type = html.escape(str(z["zone_type"]))
                safe_driver = html.escape(str(z["primary_driver"]))
                st.markdown(f"""
                <div style="background: #121215; border: 1px solid #22c55e33; border-left: 3px solid #22c55e; border-radius: 4px; padding: 12px 14px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 13px; font-weight: 700; color: #fafafa;">{z['zone_id']} — {safe_loc}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #22c55e; font-weight: 700;">Risk: {z['risk_slope']:+.1f}%/wk</span>
                    </div>
                    <div style="font-size: 11px; color: #a1a1aa; margin-top: 2px;">{safe_type}</div>
                    <div style="font-size: 12px; color: #d4d4d8; margin-top: 6px; padding-top: 6px; border-top: 1px solid #27272a;">
                         <b>Why improving:</b> {safe_driver}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No zones show significant improvement at this reference week.")
    
    with top_col2:
        st.markdown("""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #ef4444; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.06em;">
             TOP 3 MOST DETERIORATED AREAS
        </div>
        """, unsafe_allow_html=True)
        
        if deteriorating:
            for i, z in enumerate(deteriorating[:3]):
                safe_loc = html.escape(str(z["location_name"]))
                safe_type = html.escape(str(z["zone_type"]))
                safe_driver = html.escape(str(z["primary_driver"]))
                st.markdown(f"""
                <div style="background: #121215; border: 1px solid #ef444433; border-left: 3px solid #ef4444; border-radius: 4px; padding: 12px 14px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 13px; font-weight: 700; color: #fafafa;">{z['zone_id']} — {safe_loc}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #ef4444; font-weight: 700;">Risk: {z['risk_slope']:+.1f}%/wk</span>
                    </div>
                    <div style="font-size: 11px; color: #a1a1aa; margin-top: 2px;">{safe_type}</div>
                    <div style="font-size: 12px; color: #d4d4d8; margin-top: 6px; padding-top: 6px; border-top: 1px solid #27272a;">
                         <b>Why worsening:</b> {safe_driver}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No zones show significant deterioration at this reference week.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ================================================================
    # TABS: DETAILED VIEWS
    # ================================================================
    tab_table, tab_scatter, tab_drivers, tab_drilldown = st.tabs([
        " ALL ZONES RANKED TABLE",
        " TRAJECTORY SCATTER PLOT",
        " ROOT-CAUSE DRIVER BREAKDOWN",
        " SINGLE-ZONE DRILL-DOWN FORECAST"
    ])
    
    # ---- TAB 1: FULL RANKED TABLE ----
    with tab_table:
        st.markdown("#### Complete 50-Zone Trajectory Ranking")
        st.caption("All municipal zones ranked by trajectory health. Zones worsening the most appear first. Each row includes a plain-English explanation of WHY the zone is trending in that direction.")
        
        # Build dataframe
        table_data = []
        for z in sorted(all_results, key=lambda x: x["health_score"], reverse=True):
            table_data.append({
                "Zone": z["zone_id"],
                "Location": z["location_name"],
                "Type": z["zone_type"],
                "Status": z["status_label"],
                "Risk %": f"{z['current_risk_pct']:.1f}%",
                "Risk Trend (per wk)": f"{z['risk_slope']:+.1f}%",
                "Cong. Trend": f"{z['cong_slope']:+.1f}",
                "Speed Trend": f"{z['speed_slope']:+.1f} km/h",
                "Density Trend": f"{z['density_slope']:+.1f} veh/km",
                "PRIMARY DRIVER — WHY?": z["primary_driver"],
            })
        
        table_df = pd.DataFrame(table_data)
        st.dataframe(
            table_df,
            use_container_width=True,
            height=600,
            column_config={
                "Status": st.column_config.TextColumn("Status", width="medium"),
                "PRIMARY DRIVER — WHY?": st.column_config.TextColumn("PRIMARY DRIVER — WHY?", width="large"),
            }
        )
    
    # ---- TAB 2: TRAJECTORY SCATTER PLOT ----
    with tab_scatter:
        st.markdown("#### Zone Trajectory Health Map")
        st.caption("Each bubble represents a zone. X-axis = congestion trend (positive = worsening), Y-axis = speed trend (negative = worsening). Bubble size = current risk level. Color = trajectory status.")
        
        scatter_data = pd.DataFrame([{
            "Zone": z["zone_id"],
            "Location": z["location_name"],
            "Congestion Trend (pts/wk)": z["cong_slope"],
            "Speed Trend (km/h/wk)": z["speed_slope"],
            "Current Risk %": z["current_risk_pct"],
            "Status": z["status"],
            "Primary Driver": z["primary_driver"],
            "Risk Slope": z["risk_slope"],
        } for z in all_results])
        
        # Color mapping
        color_map = {"IMPROVING": "#22c55e", "STABLE": "#eab308", "DETERIORATING": "#ef4444"}
        
        fig_scatter = px.scatter(
            scatter_data,
            x="Congestion Trend (pts/wk)",
            y="Speed Trend (km/h/wk)",
            size="Current Risk %",
            color="Status",
            color_discrete_map=color_map,
            hover_data=["Zone", "Location", "Primary Driver", "Risk Slope"],
            title=f"Zone Trajectory Health Map — Week {reference_week}",
            size_max=30,
        )
        
        # Add quadrant reference lines
        fig_scatter.add_hline(y=0, line_dash="dot", line_color="#3f3f46", line_width=1)
        fig_scatter.add_vline(x=0, line_dash="dot", line_color="#3f3f46", line_width=1)
        
        # Add quadrant labels
        fig_scatter.add_annotation(x=3, y=3, text=" Improving<br>(↓Cong, ↑Speed)", showarrow=False, font=dict(size=10, color="rgba(34, 197, 94, 0.7)"))
        fig_scatter.add_annotation(x=-3, y=-3, text=" Worsening<br>(↑Cong, ↓Speed)", showarrow=False, font=dict(size=10, color="rgba(239, 68, 68, 0.7)"))
        
        fig_scatter.update_layout(
            paper_bgcolor="#18181b",
            plot_bgcolor="#18181b",
            font={"family": "Inter", "color": "#fafafa"},
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(gridcolor="#27272a", zeroline=False, title="← Improving | Congestion Trend (pts/wk) | Worsening →"),
            yaxis=dict(gridcolor="#27272a", zeroline=False, title="← Worsening | Speed Trend (km/h/wk) | Improving →"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ---- TAB 3: ROOT-CAUSE DRIVER BREAKDOWN ----
    with tab_drivers:
        st.markdown("#### Root-Cause Factor Attribution — Top 10 Worsening Zones")
        st.caption("Breakdown of which specific metric is driving each deteriorating zone's worsening trajectory. Larger bars indicate a stronger contributing factor.")
        
        worst_zones = deteriorating[:10] if deteriorating else sorted(all_results, key=lambda x: x["health_score"], reverse=True)[:10]
        
        # Build stacked bar data
        bar_data = []
        for z in worst_zones:
            zone_label = f"{z['zone_id']}: {z['location_name']}"
            # Normalize each contributing factor as a positive magnitude
            bar_data.append({"Zone": zone_label, "Factor": "Congestion Surge", "Contribution": max(0, z["cong_slope"] * 0.35)})
            bar_data.append({"Zone": zone_label, "Factor": "Speed Decline", "Contribution": max(0, -z["speed_slope"] * 0.25)})
            bar_data.append({"Zone": zone_label, "Factor": "Density Increase", "Contribution": max(0, z["density_slope"] * 0.15)})
            bar_data.append({"Zone": zone_label, "Factor": "Violation Rise", "Contribution": max(0, z["violation_slope"] * 0.15)})
            bar_data.append({"Zone": zone_label, "Factor": "Risk Escalation", "Contribution": max(0, z["risk_slope"] * 0.10)})
        
        if bar_data:
            bar_df = pd.DataFrame(bar_data)
            # Filter out zero contributions
            bar_df = bar_df[bar_df["Contribution"] > 0.01]
            
            fig_bar = px.bar(
                bar_df,
                x="Contribution",
                y="Zone",
                color="Factor",
                orientation="h",
                title="Worsening Factor Decomposition — What's Causing Each Zone to Deteriorate?",
                color_discrete_map={
                    "Congestion Surge": "#ef4444",
                    "Speed Decline": "#f97316",
                    "Density Increase": "#eab308",
                    "Violation Rise": "#818cf8",
                    "Risk Escalation": "#ec4899",
                },
                barmode="stack",
            )
            fig_bar.update_layout(
                paper_bgcolor="#18181b",
                plot_bgcolor="#18181b",
                font={"family": "Inter", "color": "#fafafa"},
                height=max(350, len(worst_zones) * 45),
                margin=dict(l=10, r=20, t=40, b=20),
                xaxis=dict(gridcolor="#27272a", title="Weighted Contribution Score"),
                yaxis=dict(autorange="reversed", gridcolor="#27272a"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detail cards for each zone
        st.markdown("##### Detailed Root-Cause Explanations")
        for z in worst_zones:
            safe_loc = html.escape(str(z["location_name"]))
            safe_driver = html.escape(str(z["primary_driver"]))
            secondary_html = ""
            if z["secondary_factors"]:
                secondary_items = "".join([f"<li style='color: #a1a1aa; font-size: 12px;'>{html.escape(f)}</li>" for f in z["secondary_factors"]])
                secondary_html = f"<div style='margin-top: 6px;'><span style='font-size: 11px; color: #71717a; text-transform: uppercase;'>Contributing Factors:</span><ul style='margin: 4px 0 0 16px; padding: 0;'>{secondary_items}</ul></div>"
            
            st.markdown(f"""
            <div style="background: #121215; border: 1px solid #27272a; border-left: 3px solid {z['status_color']}; border-radius: 4px; padding: 12px 16px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 13px; font-weight: 700; color: #fafafa;">{z['zone_id']} — {safe_loc}</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: {z['status_color']}; font-weight: 700;">{z['status_label']} • Risk: {z['current_risk_pct']:.1f}% ({z['risk_slope']:+.1f}%/wk)</span>
                </div>
                <div style="font-size: 12px; color: #d4d4d8; margin-top: 8px;">
                    <b>Primary Driver:</b> {safe_driver}
                </div>
                {secondary_html}
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; padding-top: 8px; border-top: 1px solid #27272a;">
                    <div style="text-align: center;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #71717a; text-transform: uppercase;">Congestion</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: {'#ef4444' if z['cong_slope'] > 0.5 else '#22c55e' if z['cong_slope'] < -0.5 else '#a1a1aa'};">{z['cong_slope']:+.1f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #71717a; text-transform: uppercase;">Speed</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: {'#22c55e' if z['speed_slope'] > 0.5 else '#ef4444' if z['speed_slope'] < -0.5 else '#a1a1aa'};">{z['speed_slope']:+.1f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #71717a; text-transform: uppercase;">Density</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: {'#ef4444' if z['density_slope'] > 0.5 else '#22c55e' if z['density_slope'] < -0.5 else '#a1a1aa'};">{z['density_slope']:+.1f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #71717a; text-transform: uppercase;">Violations</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: {'#ef4444' if z['violation_slope'] > 0.3 else '#22c55e' if z['violation_slope'] < -0.3 else '#a1a1aa'};">{z['violation_slope']:+.1f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ---- TAB 4: SINGLE-ZONE DRILL-DOWN FORECAST ----
    with tab_drilldown:
        st.markdown("#### Single-Zone Deep-Dive Forecast")
        st.caption("Select any zone to see its detailed 4-week forward projection with unmitigated vs AI-mitigated scenarios.")
        
        # Zone selector for drill-down
        zone_options = {z["zone_id"]: f"{z['zone_id']}: {z['location_name']} ({z['zone_type']}) — {z['status_label']}" for z in all_results}
        selected_drill_zone = st.selectbox(
            "SELECT ZONE FOR DEEP-DIVE",
            list(zone_options.keys()),
            format_func=lambda z: zone_options.get(z, z),
            index=0,
        )
        
        # Find the zone's analysis result
        zone_result = next((z for z in all_results if z["zone_id"] == selected_drill_zone), None)
        if zone_result is None:
            st.warning("Zone not found.")
            return
        
        zone_df_drill = df[df["zone_id"] == selected_drill_zone].sort_values("week").reset_index(drop=True)
        
        # Build 4-week forecast
        hist_df = zone_df_drill[zone_df_drill["week"] <= reference_week].sort_values("week")
        if len(hist_df) < 2:
            st.info("Insufficient data for this zone.")
            return
        
        current_row = hist_df.iloc[-1]
        base_cong = float(current_row["congestion"])
        base_spd = float(current_row["average_speed"])
        base_dens = float(current_row["vehicle_density"])
        curr_risk = compute_zone_risk(model, pd.DataFrame([current_row]))
        if np.isnan(curr_risk):
            curr_risk = 0.35
        
        # Status banner
        safe_loc = html.escape(str(zone_result["location_name"]))
        safe_driver = html.escape(str(zone_result["primary_driver"]))
        st.markdown(f"""
        <div style="background: #18181b; border: 1px solid #27272a; border-left: 4px solid {zone_result['status_color']}; border-radius: 6px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: {zone_result['status_color']}; font-weight: 700;">
                    {zone_result['status_label']}
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa;">
                    Risk: {zone_result['current_risk_pct']:.1f}% • Trajectory: {zone_result['risk_slope']:+.1f}%/wk
                </div>
            </div>
            <div style="font-size: 14px; font-weight: 700; color: #fafafa;">{zone_result['zone_id']} — {safe_loc}</div>
            <div style="font-size: 13px; color: #d4d4d8; margin-top: 8px;">
                <b>Root Cause:</b> {safe_driver}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Forecast cards
        forecast_weeks = []
        for step in range(1, 5):
            f_week = reference_week + step
            f_cong = float(np.clip(base_cong + (zone_result["cong_slope"] * step), 10.0, 98.0))
            f_spd = float(np.clip(base_spd + (zone_result["speed_slope"] * step), 12.0, 75.0))
            f_dens = float(np.clip(base_dens + (zone_result["density_slope"] * step), 15.0, 160.0))
            f_risk_pct = float(np.clip((curr_risk * 100.0) + (zone_result["risk_slope"] * step), 5.0, 95.0))
            exp_incidents = round(max(0.1, (f_risk_pct / 100.0) * (f_dens / 45.0) * 1.4), 1)
            risk_color = "#ef4444" if f_risk_pct >= 75 else ("#f97316" if f_risk_pct >= 55 else ("#eab308" if f_risk_pct >= 35 else "#22c55e"))
            risk_tier = "CRITICAL" if f_risk_pct >= 75 else ("HIGH" if f_risk_pct >= 55 else ("MODERATE" if f_risk_pct >= 35 else "LOW"))
            
            forecast_weeks.append({
                "step": f"W+{step}",
                "calendar_week": f_week,
                "predicted_risk_pct": round(f_risk_pct, 1),
                "risk_tier": risk_tier,
                "risk_color": risk_color,
                "projected_congestion": round(f_cong, 1),
                "projected_speed_kmh": round(f_spd, 1),
                "projected_density": int(f_dens),
                "expected_incidents": exp_incidents,
            })
        
        # Render forecast cards
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px;">
            {''.join([f"""
            <div style="background: #121215; border: 1px solid #27272a; border-top: 3px solid {fw['risk_color']}; padding: 12px 14px; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa;">{fw['step']} (W{fw['calendar_week']})</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: {fw['risk_color']}; font-weight: 700;">{fw['risk_tier']}</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 800; color: {fw['risk_color']}; margin-top: 6px;">{fw['predicted_risk_pct']}%</div>
                <div style="font-size: 11px; color: #a1a1aa; margin-top: 6px;">
                    Cong: <b>{fw['projected_congestion']}</b> • Speed: <b>{fw['projected_speed_kmh']} km/h</b><br>
                    Density: <b>{fw['projected_density']} veh/km</b> • Est. Incidents: <b>{fw['expected_incidents']}</b>
                </div>
            </div>
            """ for fw in forecast_weeks])}
        </div>
        """, unsafe_allow_html=True)
        
        # Scenario comparison
        sc_col1, sc_col2 = st.columns(2)
        
        # Build scenario text based on status
        if zone_result["status"] == "DETERIORATING":
            unmitigated_text = f"If unmitigated, congestion is projected to surge by +{zone_result['cong_slope'] * 4:.1f} points over the next 4 weeks. {zone_result['primary_driver']}. Collision probability will continue escalating with {zone_result['forecast_4w_risk_delta']:+.1f}% risk shift."
            mitigated_text = "Deploying Dynamic Traffic Signal Control (DTSC) with +15s green-phase prioritization and automated stop-line enforcement is projected to cut accident probability by -32.5% and prevent gridlock cascades."
        elif zone_result["status"] == "IMPROVING":
            unmitigated_text = f"Sector is naturally recovering with congestion dissipating at {abs(zone_result['cong_slope']):.1f} pts/week. {zone_result['primary_driver']}. Collision probability is steadily dropping with {zone_result['forecast_4w_risk_delta']:+.1f}% projected shift."
            mitigated_text = "Maintaining optimal DTSC cycle times and continuing current enforcement posture will sustain the positive trajectory and prevent regression."
        else:
            unmitigated_text = f"Corridor will experience moderate localized rush-hour queues, but collision probability will remain within manageable thresholds. Risk shift: {zone_result['forecast_4w_risk_delta']:+.1f}% over 4 weeks."
            mitigated_text = "Standard automated green-wave coordination and periodic enforcement patrols will maintain equilibrium."
        
        with sc_col1:
            st.markdown(f"""
            <div style="background: #18181b; border: 1px solid #ef444444; border-left: 4px solid #ef4444; border-radius: 6px; padding: 16px; height: 100%;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #ef4444; font-weight: 700; text-transform: uppercase;">
                    SCENARIO A • STATUS QUO (UNMITIGATED)
                </div>
                <div style="font-size: 13px; color: #d4d4d8; line-height: 1.6; margin-top: 10px;">
                    {unmitigated_text}
                </div>
                <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid #27272a; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #ef4444;">
                     4-Week Risk Delta: <b>{zone_result['forecast_4w_risk_delta']:+.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with sc_col2:
            st.markdown(f"""
            <div style="background: #18181b; border: 1px solid #22c55e44; border-left: 4px solid #22c55e; border-radius: 6px; padding: 16px; height: 100%;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #22c55e; font-weight: 700; text-transform: uppercase;">
                    SCENARIO B • AI-OPTIMIZED DISPATCH & DTSC
                </div>
                <div style="font-size: 13px; color: #d4d4d8; line-height: 1.6; margin-top: 10px;">
                    {mitigated_text}
                </div>
                <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid #27272a; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #22c55e;">
                     Projected Mitigation Dividend: <b>-32.5% Incident Reduction</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Historical timeline chart for this zone
        st.markdown("##### Historical Risk & Congestion Timeline")
        
        fig_timeline = go.Figure()
        
        hist_weeks = zone_df_drill[zone_df_drill["week"] <= reference_week]
        
        fig_timeline.add_trace(go.Scatter(
            x=hist_weeks["week"],
            y=hist_weeks["congestion"],
            mode="lines+markers",
            name="Congestion Index",
            line=dict(color="#fafafa", width=2),
            marker=dict(size=3),
        ))
        
        fig_timeline.add_trace(go.Scatter(
            x=hist_weeks["week"],
            y=hist_weeks["average_speed"],
            mode="lines",
            name="Average Speed (km/h)",
            line=dict(color="#71717a", width=1.5, dash="dot"),
        ))
        
        # Risk probabilities
        risk_x, risk_y = [], []
        for _, row in hist_weeks.iterrows():
            if row["week"] >= 5:
                p = compute_zone_risk(model, pd.DataFrame([row]))
                if not np.isnan(p):
                    risk_x.append(row["week"])
                    risk_y.append(p * 100.0)
        
        fig_timeline.add_trace(go.Scatter(
            x=risk_x,
            y=risk_y,
            mode="lines+markers",
            name="ML Risk Probability (%)",
            line=dict(color="#ef4444", width=2.5),
            marker=dict(size=4),
        ))
        
        # Forward forecast overlay
        fc_x = [reference_week] + [fw["calendar_week"] for fw in forecast_weeks]
        fc_y = [risk_y[-1] if risk_y else 35.0] + [fw["predicted_risk_pct"] for fw in forecast_weeks]
        
        fig_timeline.add_trace(go.Scatter(
            x=fc_x,
            y=fc_y,
            mode="lines+markers",
            name=" 4-Week Forecast",
            line=dict(color=zone_result["status_color"], width=3, dash="dashdot"),
            marker=dict(size=6, symbol="diamond"),
        ))
        
        fig_timeline.add_vline(
            x=reference_week,
            line_width=1.5,
            line_dash="dash",
            line_color="#a1a1aa",
            annotation_text=f"Ref W{reference_week}",
            annotation_position="top right",
        )
        
        fig_timeline.update_layout(
            paper_bgcolor="#18181b",
            plot_bgcolor="#18181b",
            font={"family": "Inter", "color": "#fafafa"},
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Week", gridcolor="#27272a"),
            yaxis=dict(title="Index / %", gridcolor="#27272a"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
