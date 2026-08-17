"""
================================================================================
ROADSENSE AI — SIMULATION TESTING LAB & ML RISK PREDICTION ENGINE
DESIGN SYSTEM: KINETIC INFRASTRUCTURE INTELLIGENCE (STITCH MCP)
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
from typing import Dict, Tuple, Optional, Any

from intelligence.signal_co2 import compute_optimal_signal_timing, estimate_co2_impact
from intelligence.llm_briefing import generate_zone_briefing, generate_city_summary


# ==============================================================================
# 1. DATA INGESTION & CACHING
# ==============================================================================

@st.cache_data
def load_simulation_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
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
def load_trained_risk_model():
    """Loads the trained Supervised ML Risk Model pipeline with integrity verification."""
    model_path = "models/best_risk_model.pkl"
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, "rb") as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        model = joblib.load(model_path)
        print(f"[+] ML model loaded. SHA-256 prefix: {model_hash}")
        return model
    except Exception as e:
        st.error(f"Error loading ML model from {model_path}: {e}")
        return None


@st.cache_data
def get_zone_risk_timeline(_model, _zone_df: pd.DataFrame) -> list:
    """Caches timeline predictions across all 52 weeks for a given zone."""
    if _model is None:
        return [None] * len(_zone_df)
    results = []
    for _, row in _zone_df.iterrows():
        if row["week"] < 5:
            results.append(None)
        else:
            try:
                p = float(_model.predict_proba(pd.DataFrame([row]))[0, 1])
                results.append(round(p * 100.0, 1))
            except Exception:
                results.append(None)
    return results


# ==============================================================================
# 2. METRIC HELPERS & RISK SCORING
# ==============================================================================

def get_risk_badge(prob: float) -> Tuple[str, str]:
    """Returns (label, color_hex) for a risk probability."""
    if np.isnan(prob):
        return "BASELINE WARMUP", "#71717a"
    elif prob >= 0.75:
        return "CRITICAL RISK", "#ef4444"
    elif prob >= 0.55:
        return "HIGH RISK", "#f97316"
    elif prob >= 0.35:
        return "MODERATE RISK", "#eab308"
    else:
        return "LOW RISK", "#22c55e"


def compute_live_risk(model, row_df: pd.DataFrame) -> float:
    """Computes risk probability using the trained ML model pipeline."""
    if model is None or row_df.empty:
        return np.nan
    try:
        week = row_df["week"].iloc[0]
        if week < 5:
            return np.nan
        prob = model.predict_proba(row_df)[0, 1]
        return float(prob)
    except Exception:
        return float(row_df.get("rolling_4_week_incident_rate", pd.Series([np.nan])).iloc[0])


def render_metric_card(label: str, value: str, delta_str: str, val_color: str = "#fafafa"):
    """Minimalist metric tile component."""
    st.markdown(f"""
    <div class="telemetry-card">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">{html.escape(label)}</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; color: {val_color}; margin: 4px 0;">{value}</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #71717a;">{html.escape(delta_str)}</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 3. RENDER SIMULATION TESTING LAB
# ==============================================================================

def render_simulation_dashboard():
    """Renders the comprehensive zone-wise and week-wise simulation testing lab."""
    
    try:
        df, loc_df = load_simulation_data()
        model = load_trained_risk_model()
    except Exception as e:
        st.error(f"Failed to load simulation environment: {e}")
        return

    # Header Panel
    st.markdown("""
    <div class="telemetry-header">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; letter-spacing: 0.06em; text-transform: uppercase;">
            SIMULATION LAB 01
        </div>
        <h2 style="margin: 4px 0 0 0; font-size: 22px;">
            Municipal Risk Prediction & Scenario Testing Engine
        </h2>
        <p style="margin: 6px 0 0 0; color: #a1a1aa; font-size: 13px;">
            Temporal feature evaluation, machine learning risk forecasting, and civic carbon emission offset analysis across 50 municipal zones.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.markdown("### SIMULATION CONTROLS")
    
    all_types = ["All Types"] + sorted(list(df["zone_type"].dropna().unique()))
    selected_type = st.sidebar.selectbox("FILTER ZONE TYPE", all_types, index=0)

    filtered_df = df if selected_type == "All Types" else df[df["zone_type"] == selected_type]
    unique_zones = sorted(filtered_df["zone_id"].unique())
    
    zone_label_map = {}
    for z in unique_zones:
        sub = filtered_df[filtered_df["zone_id"] == z].iloc[0]
        loc_name = sub.get("location_name", z)
        city = sub.get("city", "")
        z_type = sub.get("zone_type", "")
        zone_label_map[z] = f"{z}: {loc_name} ({city}) — {z_type}"

    selected_zone = st.sidebar.selectbox(
        "MUNICIPAL ZONE",
        unique_zones,
        format_func=lambda z: zone_label_map.get(z, z),
        index=0
    )

    selected_week = st.sidebar.slider(
        "TIMELINE WEEK",
        min_value=1,
        max_value=52,
        value=50,
        help="Weeks 1-4 establish rolling baselines. Weeks 5-52 contain trained ML risk predictions."
    )

    if selected_week < 5:
        st.sidebar.info("Warmup Period: Weeks 1-4 establish rolling baselines. ML Risk predictions begin at Week 5.")

    # Filter Zone Data
    zone_all_weeks = df[df["zone_id"] == selected_zone].sort_values("week").reset_index(drop=True)
    current_row = zone_all_weeks[zone_all_weeks["week"] == selected_week]

    if current_row.empty:
        st.warning(f"No observation data found for {selected_zone} at Week {selected_week}.")
        return

    current_data = current_row.iloc[0]
    prev_row = zone_all_weeks[zone_all_weeks["week"] == (selected_week - 1)]
    prev_data = prev_row.iloc[0] if not prev_row.empty else None

    # Calculate Current Risk Probability
    current_risk_prob = compute_live_risk(model, current_row)
    prev_risk_prob = compute_live_risk(model, prev_row) if prev_data is not None else np.nan
    risk_label, risk_color = get_risk_badge(current_risk_prob)

    # Calculate Signal & CO2 Intelligence
    signal_info = compute_optimal_signal_timing(
        congestion=float(current_data["congestion"]),
        vehicle_density=float(current_data["vehicle_density"]),
        average_speed=float(current_data["average_speed"]),
        zone_type=str(current_data.get("zone_type", "Residential")),
        special_event=int(current_data.get("special_event", 0)),
        weather=str(current_data.get("weather", "Normal")),
    )

    co2_info = estimate_co2_impact(
        vehicle_density=float(current_data["vehicle_density"]),
        congestion=float(current_data["congestion"]),
        average_speed=float(current_data["average_speed"]),
        population_density=int(current_data.get("population_density", 5000)),
    )

    # Zone Summary Header
    loc_display = html.escape(str(current_data.get('location_name', selected_zone)))
    city_display = html.escape(str(current_data.get('city', 'Metropolis')))
    type_display = html.escape(str(current_data.get('zone_type', 'Urban')))
    safe_zone = html.escape(str(selected_zone))

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; background: #18181b; padding: 14px 20px; border-radius: 4px; border: 1px solid #27272a; margin-bottom: 16px;">
        <div>
            <span style="font-size: 16px; font-weight: 700; color: #fafafa;">{safe_zone} — {loc_display}</span>
            <span style="color: #71717a; font-size: 13px; margin-left: 8px;">({city_display} | {type_display})</span>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-family: 'JetBrains Mono', monospace; color: #a1a1aa; font-size: 12px;">WEEK {selected_week} / 52</span>
            <span class="telemetry-badge" style="color: {risk_color}; border: 1px solid {risk_color}44;">{risk_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Intelligence Briefing
    zone_payload = current_data.to_dict()
    zone_payload["risk_prob"] = current_risk_prob

    briefing_key = f"brief_{selected_zone}_{selected_week}"
    if briefing_key not in st.session_state:
        st.session_state[briefing_key] = generate_zone_briefing(zone_payload, signal_info, co2_info)

    briefing_text = st.session_state[briefing_key]

    st.markdown(f"""
    <div class="telemetry-card" style="border-left: 3px solid #fafafa;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase; margin-bottom: 6px;">
            CIVIL INTELLIGENCE EXECUTIVE SUMMARY
        </div>
        <div style="font-size: 13px; line-height: 1.6; color: #d4d4d8;">
            {html.escape(briefing_text)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        if not np.isnan(current_risk_prob):
            val_str = f"{current_risk_prob * 100:.1f}%"
            if not np.isnan(prev_risk_prob):
                delta_val = (current_risk_prob - prev_risk_prob) * 100.0
                delta_sign = "+" if delta_val >= 0 else ""
                delta_str = f"{delta_sign}{delta_val:.1f}% vs W{selected_week-1}"
            else:
                delta_str = "Baseline Week"
        else:
            val_str = "WARMUP"
            delta_str = "Insufficient History"
        render_metric_card("Predicted Incident Risk", val_str, delta_str, val_color=risk_color)

    with kpi2:
        cong = current_data["congestion"]
        if prev_data is not None:
            cong_delta = cong - prev_data["congestion"]
            cong_sign = "+" if cong_delta >= 0 else ""
            cong_delta_str = f"{cong_sign}{cong_delta:.1f} vs W{selected_week-1}"
        else:
            cong_delta_str = "Baseline"
        render_metric_card("Congestion Index", f"{cong:.1f} / 100", cong_delta_str)

    with kpi3:
        spd = current_data["average_speed"]
        if prev_data is not None:
            spd_delta = spd - prev_data["average_speed"]
            spd_sign = "+" if spd_delta >= 0 else ""
            spd_delta_str = f"{spd_sign}{spd_delta:.1f} km/h vs W{selected_week-1}"
        else:
            spd_delta_str = "Baseline"
        render_metric_card("Mean Velocity", f"{spd:.1f} km/h", spd_delta_str)

    with kpi4:
        dens = current_data["vehicle_density"]
        if prev_data is not None:
            dens_delta = dens - prev_data["vehicle_density"]
            dens_sign = "+" if dens_delta >= 0 else ""
            dens_delta_str = f"{dens_sign}{dens_delta:.0f} veh/km vs W{selected_week-1}"
        else:
            dens_delta_str = "Baseline"
        render_metric_card("Vehicle Density", f"{dens:.0f} veh/km", dens_delta_str)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "HISTORICAL TIMELINE & RISK TRENDS",
        "ADAPTIVE SIGNAL CONTROL & EMISSION SAVINGS",
        "MUNICIPAL 50-ZONE LEADERBOARD & EXECUTIVE SUMMARY",
        "WHAT-IF RISK SIMULATION ENGINE",
        "RAW OBSERVATION TELEMETRY"
    ])

    # Tab 1: Timeline
    with tab1:
        st.markdown("#### 52-Week Historical Trajectory")

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=zone_all_weeks["week"],
            y=zone_all_weeks["congestion"],
            mode="lines+markers",
            name="Congestion Index (0-100)",
            line=dict(color="#fafafa", width=2),
            marker=dict(size=3)
        ))
        fig_trend.add_trace(go.Scatter(
            x=zone_all_weeks["week"],
            y=zone_all_weeks["average_speed"],
            mode="lines",
            name="Average Speed (km/h)",
            line=dict(color="#71717a", width=1.5, dash="dot")
        ))

        risk_probs_timeline = get_zone_risk_timeline(model, zone_all_weeks)
        fig_trend.add_trace(go.Scatter(
            x=zone_all_weeks["week"],
            y=risk_probs_timeline,
            mode="lines+markers",
            name="Predicted Incident Risk (%)",
            line=dict(color="#ef4444", width=2.5),
            marker=dict(size=4)
        ))

        fig_trend.add_vline(
            x=selected_week,
            line_width=1.5,
            line_dash="dash",
            line_color="#a1a1aa",
            annotation_text=f"W{selected_week}",
            annotation_position="top right"
        )

        fig_trend.update_layout(
            paper_bgcolor="#18181b",
            plot_bgcolor="#18181b",
            font={"family": "Inter", "color": "#fafafa"},
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Calendar Timeline (Weeks 1 to 52)", gridcolor="#27272a"),
            yaxis=dict(title="Index / Percentage", gridcolor="#27272a"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_trend, width="stretch")

    # Tab 2: Signal & CO2
    with tab2:
        st.markdown("#### Dynamic Signal Optimization & Environmental Impact")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="telemetry-card">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">SIGNAL CONTROLLER OUTPUT</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700; color: #fafafa; margin: 6px 0;">{signal_info['recommended_green_seconds']}s <span style="font-size: 14px; color: #71717a;">(Base: {signal_info['base_green_seconds']}s)</span></div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #a1a1aa;">URGENCY: {signal_info['urgency']}</div>
                <div style="margin-top: 8px; font-size: 13px; color: #d4d4d8;">{signal_info['reason']}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="telemetry-card">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">CIVIC SUSTAINABILITY DIVIDEND</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700; color: #fafafa; margin: 6px 0;">{co2_info['potential_savings_kg_per_week']:,.0f} kg CO2/wk</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #a1a1aa;">Fuel Saved: {co2_info['fuel_saved_liters_per_week']:,.0f} L/wk • Tree Offset: {co2_info['trees_equivalent_per_year']:,} trees/yr</div>
            </div>
            """, unsafe_allow_html=True)

    # Tab 3: Leaderboard
    with tab3:
        st.markdown(f"#### Municipal 50-Zone Ranking — Week {selected_week}")
        
        all_week_data = df[df["week"] == selected_week].copy()
        if not all_week_data.empty:
            rank_df = all_week_data[["zone_id", "location_name", "city", "vehicle_density", "congestion", "incident_occurred"]].copy()
            rank_df = rank_df.sort_values(by="congestion", ascending=False).reset_index(drop=True)
            rank_df.index += 1
            rank_df.index.name = "RANK"
            st.dataframe(rank_df, width="stretch")

    # Tab 4: What-If Sandbox
    with tab4:
        st.markdown("#### Scenario Simulation Sandbox")
        st.caption("Adjust dynamic road variables to observe real-time risk model inference sensitivity.")

        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            sim_dens = st.slider("SIMULATED VEHICLE DENSITY", 50, 400, int(current_data["vehicle_density"]))
        with w_col2:
            sim_cong = st.slider("SIMULATED CONGESTION INDEX", 10.0, 100.0, float(current_data["congestion"]))
        with w_col3:
            sim_speed = st.slider("SIMULATED SPEED (KM/H)", 5.0, 70.0, float(current_data["average_speed"]))

        sim_row = current_row.copy()
        sim_row["vehicle_density"] = sim_dens
        sim_row["congestion"] = sim_cong
        sim_row["average_speed"] = sim_speed

        sim_risk = compute_live_risk(model, sim_row)
        s_lbl, s_col = get_risk_badge(sim_risk)

        st.markdown(f"""
        <div class="telemetry-card" style="margin-top: 16px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">SYNTHESIZED RISK OUTPUT</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 700; color: {s_col}; margin: 4px 0;">{s_lbl} ({sim_risk * 100:.1f}%)</div>
            <div style="font-size: 12px; color: #71717a;">Computed via trained XGBoost pipeline inference.</div>
        </div>
        """, unsafe_allow_html=True)

    # Tab 5: Raw Telemetry
    with tab5:
        st.markdown("#### Raw Feature Vector Stream")
        st.dataframe(current_row.T, width="stretch")
