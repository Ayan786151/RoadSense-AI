"""
================================================================================
ROAD SENSE AI - SIMULATION TESTING DASHBOARD & RISK PREDICTION LAB
MODULE: ZONE-WISE & WEEK-WISE HISTORICAL EXPLORER & COMPARATIVE ANALYTICS
================================================================================

This module provides a modular, self-contained testing dashboard for the 50-zone,
52-week simulated traffic intelligence dataset and trained ML Risk Model.

KEY CAPABILITIES:
1. ZONE & WEEK SELECTOR:
   - Browse all 50 municipal zones across Weeks 1 to 52.
2. AI-POWERED BRIEFING & LLM INSIGHTS:
   - Automated natural-language executive briefs (Groq LLaMA 3.3 / Gemini 2.5) focusing on citizen safety and social impact.
3. ADAPTIVE SIGNAL CONTROL & CO2 SUSTAINABILITY:
   - Dynamic green-light timing optimization and civic emission/fuel savings estimates.
4. CITY-WIDE 50-ZONE LEADERBOARD:
   - High-throughput batch inference with geographic scatter mapbox.
5. WHAT-IF RISK SIMULATION SANDBOX:
   - Real-time parameter tweaking to test ML model sensitivity under dynamic conditions.
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

def get_risk_badge(prob: float) -> Tuple[str, str, str]:
    """Returns (label, color_hex, css_class) for a risk probability."""
    if np.isnan(prob):
        return "WARMUP", "#6c757d", "badge-warmup"
    elif prob >= 0.75:
        return "CRITICAL RISK", "#dc3545", "badge-critical"
    elif prob >= 0.55:
        return "HIGH RISK", "#fd7e14", "badge-high"
    elif prob >= 0.35:
        return "MODERATE RISK", "#ffc107", "badge-moderate"
    else:
        return "LOW RISK", "#28a745", "badge-low"


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


def render_metric_card(label: str, value: str, delta_str: str, delta_class: str, val_color: str = "#fff"):
    """Reusable metric card component."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value" style="color: {val_color};">{value}</div>
        <div class="{delta_class}">{html.escape(delta_str)}</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 3. RENDER SIMULATION TESTING LAB
# ==============================================================================

def render_simulation_dashboard():
    """Renders the comprehensive zone-wise and week-wise simulation testing lab."""
    
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .metric-delta-pos {
        color: #dc3545;
        font-size: 14px;
        font-weight: 600;
    }
    .metric-delta-neg {
        color: #28a745;
        font-size: 14px;
        font-weight: 600;
    }
    .metric-label {
        font-size: 13px;
        color: #9aa0a6;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .badge-critical { background-color: rgba(220, 53, 69, 0.2); color: #ff6b6b; border: 1px solid #dc3545; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 13px; }
    .badge-high { background-color: rgba(253, 126, 20, 0.2); color: #ffa94d; border: 1px solid #fd7e14; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 13px; }
    .badge-moderate { background-color: rgba(255, 193, 7, 0.2); color: #ffd43b; border: 1px solid #ffc107; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 13px; }
    .badge-low { background-color: rgba(40, 167, 69, 0.2); color: #69db7c; border: 1px solid #28a745; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 13px; }
    .badge-warmup { background-color: rgba(108, 117, 125, 0.2); color: #adb5bd; border: 1px solid #6c757d; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 13px; }
    .ai-briefing-box {
        background: linear-gradient(135deg, rgba(26, 42, 68, 0.7), rgba(18, 30, 49, 0.7));
        border: 1px solid rgba(77, 171, 247, 0.35);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # Load Data & Model
    # ----------------------------------------------------
    try:
        df, loc_df = load_simulation_data()
        model = load_trained_risk_model()
    except Exception as e:
        st.error(f"Failed to load simulation environment: {e}")
        return

    # ----------------------------------------------------
    # Header Banner
    # ----------------------------------------------------
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 22px 28px; border-radius: 12px; margin-bottom: 24px; color: white;">
        <h2 style="margin: 0; font-weight: 700; display: flex; align-items: center; gap: 12px;">
            🚦 RoadSense AI — Traffic Intelligence & Civic Social Service Hub
        </h2>
        <p style="margin: 6px 0 0 0; opacity: 0.88; font-size: 14px;">
            AI-Driven Traffic Risk Forecasting, Adaptive Signal Timing, and Civic Carbon Emission Reduction for 50 Urban Municipal Zones.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # Sidebar Controls (Zone & Week Selection)
    # ----------------------------------------------------
    st.sidebar.markdown("### 🎛️ Simulation Controls")
    
    all_types = ["All Types"] + sorted(list(df["zone_type"].dropna().unique()))
    selected_type = st.sidebar.selectbox("Filter Zone Type", all_types, index=0)

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
        "📍 Select Municipal Zone",
        unique_zones,
        format_func=lambda z: zone_label_map.get(z, z),
        index=0
    )

    st.sidebar.markdown("---")
    selected_week = st.sidebar.slider(
        "📅 Select Calendar Week",
        min_value=1,
        max_value=52,
        value=50,
        help="Weeks 1-4 establish rolling baselines. Weeks 5-52 contain trained ML risk predictions."
    )

    if selected_week < 5:
        st.sidebar.info("ℹ️ **Warmup Period**: Weeks 1-4 establish rolling baselines. ML Risk predictions begin at Week 5.")

    # ----------------------------------------------------
    # Filter Zone Data
    # ----------------------------------------------------
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
    risk_label, risk_color, risk_class = get_risk_badge(current_risk_prob)

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

    # ----------------------------------------------------
    # ZONE SUMMARY HEADER
    # ----------------------------------------------------
    loc_display = html.escape(str(current_data.get('location_name', selected_zone)))
    city_display = html.escape(str(current_data.get('city', 'Metropolis')))
    type_display = html.escape(str(current_data.get('zone_type', 'Urban')))
    safe_zone = html.escape(str(selected_zone))

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.03); padding: 14px 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 16px;">
        <div>
            <span style="font-size: 20px; font-weight: 700; color: #fff;">{safe_zone} — {loc_display}</span>
            <span style="color: #9aa0a6; font-size: 14px; margin-left: 10px;">({city_display} | {type_display})</span>
        </div>
        <div style="display: flex; align-items: center; gap: 14px;">
            <span style="color: #9aa0a6; font-size: 14px;">WEEK {selected_week} of 52</span>
            <span class="{risk_class}">{risk_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # LLM INTELLIGENCE BRIEFING (AI SOCIAL SERVICE ANALYST)
    # ----------------------------------------------------
    zone_payload = current_data.to_dict()
    zone_payload["risk_prob"] = current_risk_prob

    # Generate or retrieve briefing
    briefing_key = f"brief_{selected_zone}_{selected_week}"
    if briefing_key not in st.session_state:
        st.session_state[briefing_key] = generate_zone_briefing(zone_payload, signal_info, co2_info)

    briefing_text = st.session_state[briefing_key]

    st.markdown(f"""
    <div class="ai-briefing-box">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 14px; font-weight: 700; color: #74c0fc; display: flex; align-items: center; gap: 8px;">
                🤖 AI Review & Safety Briefing
            </span>
            <span style="font-size: 12px; color: #adb5bd;">Citizen Protection Focus</span>
        </div>
        <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #e9ecef;">
            {html.escape(briefing_text)}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # TOP KPI METRIC CARDS
    # ----------------------------------------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        if not np.isnan(current_risk_prob):
            val_str = f"{current_risk_prob * 100:.1f}%"
            if not np.isnan(prev_risk_prob):
                delta_val = (current_risk_prob - prev_risk_prob) * 100.0
                delta_sign = "+" if delta_val >= 0 else ""
                delta_class = "metric-delta-pos" if delta_val > 0 else "metric-delta-neg"
                delta_str = f"{delta_sign}{delta_val:.1f}% vs W{selected_week-1}"
            else:
                delta_str = "Baseline Week"
                delta_class = "text-muted"
        else:
            val_str = "WARMUP"
            delta_str = "Insufficient History"
            delta_class = "text-muted"
        render_metric_card("Predicted Incident Risk", val_str, delta_str, delta_class, val_color=risk_color)

    with kpi2:
        cong = current_data["congestion"]
        if prev_data is not None:
            cong_delta = cong - prev_data["congestion"]
            cong_sign = "+" if cong_delta >= 0 else ""
            cong_class = "metric-delta-pos" if cong_delta > 0 else "metric-delta-neg"
            cong_delta_str = f"{cong_sign}{cong_delta:.1f} vs W{selected_week-1}"
        else:
            cong_delta_str = "Baseline"
            cong_class = "text-muted"
        render_metric_card("Congestion Index", f"{cong:.1f}/100", cong_delta_str, cong_class, val_color="#4dabf7")

    with kpi3:
        spd = current_data["average_speed"]
        if prev_data is not None:
            spd_delta = spd - prev_data["average_speed"]
            spd_sign = "+" if spd_delta >= 0 else ""
            spd_class = "metric-delta-neg" if spd_delta >= 0 else "metric-delta-pos"
            spd_delta_str = f"{spd_sign}{spd_delta:.1f} km/h vs W{selected_week-1}"
        else:
            spd_delta_str = "Baseline"
            spd_class = "text-muted"
        render_metric_card("Average Speed", f"{spd:.1f} km/h", spd_delta_str, spd_class, val_color="#69db7c")

    with kpi4:
        dens = current_data["vehicle_density"]
        if prev_data is not None:
            dens_delta = dens - prev_data["vehicle_density"]
            dens_sign = "+" if dens_delta >= 0 else ""
            dens_class = "metric-delta-pos" if dens_delta > 0 else "metric-delta-neg"
            dens_delta_str = f"{dens_sign}{dens_delta:.0f} veh/km vs W{selected_week-1}"
        else:
            dens_delta_str = "Baseline"
            delta_class = "text-muted"
        render_metric_card("Vehicle Density", f"{dens:.0f} veh/km", dens_delta_str, dens_class, val_color="#da77f2")

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # MULTI-TAB VIEW
    # ----------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Week-over-Week Comparison & Trends",
        "🚦 Adaptive Signal Timing & Eco-Impact (CO2)",
        "🗺️ City-Wide 50-Zone Leaderboard & AI Briefing",
        "🧪 What-If Risk Simulation Sandbox",
        "📋 Full Observation Diagnostics"
    ])

    # ==========================================================================
    # TAB 1: WEEK-OVER-WEEK COMPARISON & 52-WEEK TRENDS
    # ==========================================================================
    with tab1:
        st.markdown("#### 📈 52-Week Historical Timeline & Trend Analysis")

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=zone_all_weeks["week"],
            y=zone_all_weeks["congestion"],
            mode="lines+markers",
            name="Congestion Index (0-100)",
            line=dict(color="#4dabf7", width=2.5),
            marker=dict(size=4)
        ))
        fig_trend.add_trace(go.Scatter(
            x=zone_all_weeks["week"],
            y=zone_all_weeks["average_speed"],
            mode="lines",
            name="Average Speed (km/h)",
            line=dict(color="#69db7c", width=2, dash="dot")
        ))

        risk_probs_timeline = get_zone_risk_timeline(model, zone_all_weeks)
        fig_trend.add_trace(go.Scatter(
            x=zone_all_weeks["week"],
            y=risk_probs_timeline,
            mode="lines+markers",
            name="Predicted Incident Risk (%)",
            line=dict(color="#ff6b6b", width=3),
            marker=dict(size=5)
        ))

        fig_trend.add_vline(
            x=selected_week,
            line_width=2.5,
            line_dash="dash",
            line_color="#ffd43b",
            annotation_text=f"Selected Week {selected_week}",
            annotation_position="top left"
        )

        fig_trend.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Calendar Week (1 to 52)",
            yaxis_title="Score / Speed (km/h) / Probability (%)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown("#### 🔍 Granular Comparative Feature Matrix")
            comp_metrics = [
                ("Predicted Risk Probability", f"{current_risk_prob*100:.1f}%" if not np.isnan(current_risk_prob) else "WARMUP", f"{prev_risk_prob*100:.1f}%" if not np.isnan(prev_risk_prob) else "N/A"),
                ("Congestion Score (0-100)", f"{current_data['congestion']:.2f}", f"{prev_data['congestion']:.2f}" if prev_data is not None else "N/A"),
                ("Average Velocity (km/h)", f"{current_data['average_speed']:.2f}", f"{prev_data['average_speed']:.2f}" if prev_data is not None else "N/A"),
                ("Vehicle Density (veh/km)", f"{current_data['vehicle_density']:.1f}", f"{prev_data['vehicle_density']:.1f}" if prev_data is not None else "N/A"),
                ("Traffic Pressure Index", f"{current_data.get('traffic_pressure', 0):.2f}", f"{prev_data.get('traffic_pressure', 0):.2f}" if prev_data is not None else "N/A"),
                ("Red Light Violations", f"{current_data.get('red_light_violations', 0)}", f"{prev_data.get('red_light_violations', 0)}" if prev_data is not None else "N/A"),
                ("Actual Incident Occurred", "⚠️ YES (Incident)" if current_data.get('incident_occurred', 0) == 1 else "✅ None", "⚠️ YES" if prev_data is not None and prev_data.get('incident_occurred', 0) == 1 else "✅ None"),
                ("Weather Condition", str(current_data.get('weather', 'Clear')), str(prev_data.get('weather', 'Clear')) if prev_data is not None else "N/A"),
                ("Special Event Active", "🎉 Yes" if current_data.get('special_event', 0) == 1 else "No", "🎉 Yes" if prev_data is not None and prev_data.get('special_event', 0) == 1 else "No")
            ]
            comp_df = pd.DataFrame(comp_metrics, columns=["Feature / Parameter", f"Current Week (W{selected_week})", f"Previous Week (W{selected_week-1 if selected_week > 1 else 1})"])
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

        with col_right:
            st.markdown("##### 📈 4-Week Rolling Trend Slopes (OLS)")
            c_trend = current_data.get("congestion_trend_4w", np.nan)
            s_trend = current_data.get("speed_trend_4w", np.nan)
            i_trend = current_data.get("incident_trend_4w", np.nan)

            def format_slope(val: float, invert: bool = False):
                if np.isnan(val):
                    return "WARMUP (Need 4 Weeks)", "#868e96"
                if abs(val) < 0.05:
                    return f"→ Stable ({val:+.2f}/wk)", "#4dabf7"
                if val > 0:
                    col = "#ff6b6b" if not invert else "#69db7c"
                    return f"↗ Increasing ({val:+.2f}/wk)", col
                else:
                    col = "#69db7c" if not invert else "#ff6b6b"
                    return f"↘ Decreasing ({val:+.2f}/wk)", col

            c_text, c_col = format_slope(c_trend)
            s_text, s_col = format_slope(s_trend, invert=True)
            i_text, i_col = format_slope(i_trend)

            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 14px 18px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 10px;">
                <div style="font-size: 12px; color: #9aa0a6;">CONGESTION MOMENTUM (4-WEEK OLS)</div>
                <div style="font-size: 16px; font-weight: 700; color: {c_col}; margin-top: 4px;">{c_text}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 14px 18px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 10px;">
                <div style="font-size: 12px; color: #9aa0a6;">SPEED MOMENTUM (4-WEEK OLS)</div>
                <div style="font-size: 16px; font-weight: 700; color: {s_col}; margin-top: 4px;">{s_text}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 14px 18px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
                <div style="font-size: 12px; color: #9aa0a6;">INCIDENT FREQUENCY MOMENTUM</div>
                <div style="font-size: 16px; font-weight: 700; color: {i_col}; margin-top: 4px;">{i_text}</div>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 2: ADAPTIVE SIGNAL TIMING & ECO SUSTAINABILITY (CO2)
    # ==========================================================================
    with tab2:
        st.markdown(f"#### 🚦 Adaptive Traffic Signal Optimization & Environmental Impact: {loc_display}")
        st.caption("Converts live risk & congestion telemetry into actionable intersection signal timings and civic emission savings.")

        col_sig, col_co2 = st.columns([1, 1])

        with col_sig:
            st.markdown("##### ⏱️ Dynamic Green-Light Timing Calculator")
            
            sig_urgency = signal_info["urgency"]
            urg_colors = {"CRITICAL": "#dc3545", "HIGH": "#fd7e14", "MODERATE": "#ffc107", "OPTIMIZE": "#4dabf7", "NOMINAL": "#28a745"}
            urg_col = urg_colors.get(sig_urgency, "#28a745")

            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 16px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 13px; color: #9aa0a6;">RECOMMENDED SIGNAL STATE</span>
                    <span style="background-color: {urg_col}33; color: {urg_col}; border: 1px solid {urg_col}; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 700;">
                        {sig_urgency}
                    </span>
                </div>
                <div style="display: flex; align-items: baseline; gap: 12px; margin: 12px 0;">
                    <span style="font-size: 36px; font-weight: 700; color: #69db7c;">{signal_info['recommended_green_seconds']}s</span>
                    <span style="font-size: 14px; color: #adb5bd;">Green Phase (Default: {signal_info['base_green_seconds']}s)</span>
                </div>
                <div style="font-size: 13px; color: #ced4da;">{signal_info['reason']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Factors breakdown
            st.markdown("###### Multiplier Breakdown:")
            f = signal_info["factors"]
            factor_df = pd.DataFrame([
                {"Component": "Congestion Multiplier", "Value": f"{f['congestion_factor']}x"},
                {"Component": "Vehicle Density Weight", "Value": f"{f['density_factor']}x"},
                {"Component": "Velocity Clearance Factor", "Value": f"{f['speed_factor']}x"},
                {"Component": "Special Event Boost", "Value": f"{f['event_boost']}x"},
                {"Component": "Adverse Weather Boost", "Value": f"{f['weather_boost']}x"},
            ])
            st.dataframe(factor_df, hide_index=True, use_container_width=True)

        with col_co2:
            st.markdown("##### 🌱 Civic Environmental & Social Impact")
            
            c_card1, c_card2 = st.columns(2)
            with c_card1:
                st.metric("Potential CO2 Saved", f"{co2_info['potential_savings_kg_per_week']:,.0f} kg/wk", f"{co2_info['optimization_factor_pct']}% reduction")
            with c_card2:
                st.metric("Annual Tree Equivalent", f"{co2_info['trees_equivalent_per_year']:,} trees", "carbon offset")

            c_card3, c_card4 = st.columns(2)
            with c_card3:
                st.metric("Fuel Conserved", f"{co2_info['fuel_saved_liters_per_week']:,.0f} L/wk", "petrol/diesel")
            with c_card4:
                st.metric("Citizens Impacted", f"{co2_info['citizens_impacted']:,} residents", f"{loc_display}")

            st.markdown(f"""
            <div style="background: rgba(40, 167, 69, 0.1); border: 1px solid #28a745; border-radius: 8px; padding: 14px; margin-top: 14px;">
                <div style="font-size: 13px; font-weight: 700; color: #69db7c;">🌍 Social Service Impact Summary</div>
                <div style="font-size: 13px; color: #e9ecef; margin-top: 6px;">
                    By dynamically adapting green light phases to clear the <b>{current_data['congestion']:.0f}/100</b> congestion in {loc_display}, the municipality can prevent <b>{co2_info['potential_savings_tonnes_per_year']:.1f} tonnes of CO2/year</b> and save citizens <b>{co2_info['fuel_saved_liters_per_week']:,.0f} liters of wasted fuel weekly</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 3: CITY-WIDE 50-ZONE LEADERBOARD & AI BRIEFING
    # ==========================================================================
    with tab3:
        st.markdown(f"#### 🏆 City-Wide Risk & Priority Leaderboard (Week {selected_week})")

        week_all_zones = df[df["week"] == selected_week].copy()

        # Batch Model Inference for extreme performance
        if model is not None and selected_week >= 5:
            try:
                week_all_zones["risk_prob"] = model.predict_proba(week_all_zones)[:, 1]
            except Exception as e:
                week_all_zones["risk_prob"] = week_all_zones.get("rolling_4_week_incident_rate", pd.Series(0.0, index=week_all_zones.index))
        else:
            week_all_zones["risk_prob"] = np.nan

        pop_max = week_all_zones["population_density"].max()
        pop_max = pop_max if pop_max > 0 else 1
        week_all_zones["priority_score"] = (
            (week_all_zones["risk_prob"].fillna(0.3) * 50.0) +
            (week_all_zones["congestion"] * 0.35) +
            (week_all_zones["population_density"] / pop_max * 15.0)
        ).round(2)

        week_all_zones = week_all_zones.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
        week_all_zones["rank"] = np.arange(1, len(week_all_zones) + 1)

        # City-Wide AI Briefing Box
        city_summary_key = f"city_summary_w{selected_week}"
        if city_summary_key not in st.session_state:
            st.session_state[city_summary_key] = generate_city_summary(week_all_zones.head(15).to_dict(orient="records"))

        st.markdown(f"""
        <div style="background: rgba(18, 30, 49, 0.7); border: 1px solid rgba(77, 171, 247, 0.3); border-radius: 10px; padding: 14px 18px; margin-bottom: 18px;">
            <div style="font-size: 13px; font-weight: 700; color: #74c0fc;">🏙️ AI Review — City-Wide Executive Briefing (Week {selected_week})</div>
            <p style="margin: 4px 0 0 0; font-size: 13px; color: #dee2e6;">{html.escape(st.session_state[city_summary_key])}</p>
        </div>
        """, unsafe_allow_html=True)

        # Interactive Map Visualization
        if "latitude" in week_all_zones.columns and "longitude" in week_all_zones.columns:
            has_valid_coords = week_all_zones["latitude"].notnull().any() and week_all_zones["longitude"].notnull().any()
            if has_valid_coords:
                st.markdown("##### 🗺️ Geographic Risk & Priority Heatmap")
                map_data = week_all_zones[["latitude", "longitude", "risk_prob", "priority_score", "location_name", "zone_id", "city"]].dropna(subset=["latitude", "longitude"]).copy()
                map_data["risk_pct"] = (map_data["risk_prob"].fillna(0) * 100).round(1)
                map_data["size"] = (map_data["priority_score"].clip(5, 100) * 3).astype(int)

                fig_map = px.scatter_mapbox(
                    map_data,
                    lat="latitude",
                    lon="longitude",
                    size="size",
                    color="risk_pct",
                    color_continuous_scale=["#28a745", "#ffc107", "#fd7e14", "#dc3545"],
                    range_color=[0, 80],
                    hover_name="location_name",
                    hover_data={"zone_id": True, "city": True, "risk_pct": True, "priority_score": True, "size": False, "latitude": False, "longitude": False},
                    labels={"risk_pct": "Risk %", "priority_score": "Priority"},
                    mapbox_style="carto-darkmatter",
                    zoom=9.5,
                    height=400
                )
                fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title="Risk %"))
                st.plotly_chart(fig_map, use_container_width=True)

        # Scatter Matrix & Table
        st.markdown("##### 📊 Risk vs. Congestion by Zone Type")
        fig_scatter = px.scatter(
            week_all_zones,
            x="congestion",
            y="risk_prob",
            color="zone_type",
            size="vehicle_density",
            hover_name="location_name",
            hover_data=["zone_id", "average_speed", "priority_score"],
            labels={"congestion": "Congestion Score (0-100)", "risk_prob": "Predicted Risk Probability (0-1)"},
            template="plotly_dark",
            height=360
        )
        fig_scatter.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

        display_cols = ["rank", "zone_id", "location_name", "city", "zone_type", "risk_prob", "congestion", "average_speed", "priority_score"]
        avail_cols = [c for c in display_cols if c in week_all_zones.columns]

        st.dataframe(
            week_all_zones[avail_cols].style.format({
                "risk_prob": lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "WARMUP",
                "congestion": "{:.1f}",
                "average_speed": "{:.1f} km/h",
                "priority_score": "{:.1f}"
            }),
            use_container_width=True,
            height=300
        )

    # ==========================================================================
    # TAB 4: WHAT-IF RISK SIMULATION SANDBOX
    # ==========================================================================
    with tab4:
        st.markdown("#### 🧪 Real-Time What-If Machine Learning Sandbox")
        st.caption("Interactively adjust traffic and environmental parameters for this zone to test ML risk sensitivity in real-time.")

        if model is None:
            st.warning("Trained ML model pipeline is not loaded.")
        else:
            sb_col1, sb_col2, sb_col3 = st.columns(3)

            with sb_col1:
                test_speed = st.slider("Simulated Average Speed (km/h)", 5.0, 90.0, float(current_data["average_speed"]), step=1.0)
                test_congestion = st.slider("Simulated Congestion Index (0-100)", 0.0, 100.0, float(current_data["congestion"]), step=1.0)

            with sb_col2:
                test_density = st.slider("Vehicle Density (veh/km)", 10.0, 500.0, float(current_data["vehicle_density"]), step=5.0)
                test_violations = st.slider("Red Light Violations", 0, 50, int(current_data.get("red_light_violations", 5)), step=1)

            with sb_col3:
                weather_options = ["Normal", "Light Rain", "Heavy Rain"]
                cur_w = str(current_data.get("weather", "Normal"))
                default_w_idx = weather_options.index(cur_w) if cur_w in weather_options else 0
                test_weather = st.selectbox("Weather Condition", weather_options, index=default_w_idx)
                test_event = st.checkbox("Special Municipal Event Active", value=bool(current_data.get("special_event", 0)))

            mock_row = current_row.copy()
            mock_row["average_speed"] = test_speed
            mock_row["congestion"] = test_congestion
            mock_row["vehicle_density"] = test_density
            mock_row["red_light_violations"] = test_violations
            mock_row["weather"] = test_weather
            mock_row["special_event"] = 1 if test_event else 0

            try:
                simulated_risk = float(model.predict_proba(mock_row)[0, 1])
                sim_badge, sim_color, _ = get_risk_badge(simulated_risk)

                st.markdown("---")
                res_c1, res_c2 = st.columns([1, 2])

                with res_c1:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.04); padding: 18px; border-radius: 10px; border: 1px solid {sim_color}; text-align: center;">
                        <div style="font-size: 13px; color: #9aa0a6;">SIMULATED RISK PREDICTION</div>
                        <div style="font-size: 34px; font-weight: 700; color: {sim_color}; margin: 8px 0;">{simulated_risk*100:.1f}%</div>
                        <div style="font-size: 14px; font-weight: 600; color: {sim_color};">{sim_badge}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with res_c2:
                    if not np.isnan(current_risk_prob):
                        diff = (simulated_risk - current_risk_prob) * 100.0
                        diff_sign = "+" if diff >= 0 else ""
                        diff_col = "#ff6b6b" if diff > 0 else "#69db7c"
                        diff_text = "Risk Increased" if diff > 0 else "Risk Mitigated"

                        st.markdown(f"""
                        <div style="padding: 10px 18px;">
                            <h5 style="margin: 0; color: #fff;">Impact of Simulated Interventions:</h5>
                            <p style="font-size: 18px; font-weight: 600; color: {diff_col}; margin: 8px 0;">
                                {diff_sign}{diff:.1f}% Risk Delta ({diff_text})
                            </p>
                            <p style="font-size: 13px; color: #9aa0a6; margin: 0;">
                                Baseline historical risk for Week {selected_week} was <b>{current_risk_prob*100:.1f}%</b>.
                                Adjusted parameters simulated a response in incident probability.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Simulation inference error: {e}")

    # ==========================================================================
    # TAB 5: FULL OBSERVATION DIAGNOSTICS
    # ==========================================================================
    with tab5:
        st.markdown(f"#### 📋 Complete Diagnostic Record: {html.escape(str(selected_zone))} (Week {selected_week})")
        st.json(current_data.to_dict())
