"""
================================================================================
ROADSENSE AI — CIVIL INFRASTRUCTURE & TRAFFIC TELEMETRY COMMAND CENTER
DESIGN SYSTEM: KINETIC INFRASTRUCTURE INTELLIGENCE (STITCH MCP)
================================================================================
"""

import os
import glob
import html
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure global page settings
st.set_page_config(
    page_title="RoadSense AI — Kinetic Infrastructure Command Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Minimalist Design System CSS (Zinc / Editorial Brutalism)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #fafafa;
}

/* Background canvas */
.stApp {
    background-color: #09090b;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #121215 !important;
    border-right: 1px solid #27272a !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #fafafa !important;
}

p, span, label, div {
    font-family: 'Inter', sans-serif;
}

/* Monospace data and metrics */
code, pre, .mono-val, [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.01em !important;
}

[data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 600 !important;
    color: #fafafa !important;
}

[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #a1a1aa !important;
}

/* Containers and Cards */
.telemetry-card {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.telemetry-header {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 4px;
    padding: 20px 24px;
    margin-bottom: 20px;
}

.telemetry-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 3px 8px;
    background: #27272a;
    color: #fafafa;
    border-radius: 2px;
    display: inline-block;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid #27272a;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: #a1a1aa;
    background-color: transparent;
    border: none;
    padding: 8px 16px;
    border-radius: 2px;
}

.stTabs [aria-selected="true"] {
    color: #fafafa !important;
    background-color: #18181b !important;
    border: 1px solid #27272a !important;
    border-bottom: none !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 13px;
    border-radius: 3px;
    border: 1px solid #27272a;
    background-color: #18181b;
    color: #fafafa;
    transition: background 0.15s ease;
}

.stButton > button:hover {
    background-color: #27272a;
    border-color: #3f3f46;
    color: #ffffff;
}

.stButton > button[kind="primary"] {
    background-color: #fafafa !important;
    color: #09090b !important;
    border: 1px solid #ffffff !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #e4e4e7 !important;
}

/* Inputs and Selects */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background-color: #121215 !important;
    border-color: #27272a !important;
    border-radius: 3px !important;
    color: #fafafa !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    color: #fafafa;
}

/* Dataframe tables */
[data-testid="stDataFrame"] {
    border: 1px solid #27272a;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# Import simulation dashboard, vision studio, and city map modules
try:
    from dashboard.simulation_dashboard import render_simulation_dashboard
    from dashboard.vision_studio import render_vision_studio
    from dashboard.city_map import render_city_command_map
except ImportError:
    from simulation_dashboard import render_simulation_dashboard
    from vision_studio import render_vision_studio
    from city_map import render_city_command_map

from intelligence.signal_co2 import compute_optimal_signal_timing, estimate_co2_impact


def main():
<<<<<<< HEAD
    # Sidebar Header
    st.sidebar.markdown("""
    <div style="padding: 8px 0 16px 0; border-bottom: 1px solid #27272a; margin-bottom: 16px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; letter-spacing: 0.08em; text-transform: uppercase;">
            SYSTEM CORE v2.0
        </div>
        <div style="font-size: 18px; font-weight: 700; color: #fafafa; margin-top: 4px; letter-spacing: -0.02em;">
            ROADSENSE INTELLIGENCE
        </div>
        <div style="font-size: 12px; color: #71717a; margin-top: 2px;">
            Civil Traffic Telemetry & Risk Radar
        </div>
    </div>
    """, unsafe_allow_html=True)
=======
<<<<<<< HEAD
    st.sidebar.image("assets/eye.png", width=64)
    st.sidebar.title("RoadSense")
    st.sidebar.caption("Intelligent Traffic Risk & Priority Intelligence")
=======
    st.sidebar.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
    st.sidebar.title("RoadSense AI")
    st.sidebar.caption("Intelligent Traffic Risk & Civic Social Service Hub")
>>>>>>> 3fe8d10085f4cb7cce4725237897f1e1eb2e9bdb
    st.sidebar.markdown("---")
>>>>>>> origin/main

    # Navigation Mode Selector
    app_mode = st.sidebar.radio(
        "NAVIGATION",
        [
            "01. SIMULATION & RISK ENGINE",
            "02. LIVE CCTV SURVEILLANCE",
            "03. COMPUTER VISION STUDIO",
            "04. GEOSPATIAL INCIDENT RADAR"
        ],
        index=0
    )

    st.sidebar.markdown("---")

    if app_mode.startswith("01"):
        render_simulation_dashboard()
    elif app_mode.startswith("02"):
        render_live_vision_dashboard()
    elif app_mode.startswith("03"):
        render_vision_studio()
    else:
        render_city_command_map()

    # Sidebar Footer
    st.sidebar.markdown("""
    <div style="padding: 16px 0; border-top: 1px solid #27272a; margin-top: 32px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #71717a; text-align: left;">
        STATUS: OPERATIONAL<br>
        PIPELINE: YOLOV11 + ML ENGINE<br>
        TELEMETRY: REAL-TIME INGESTION
    </div>
    """, unsafe_allow_html=True)


def render_live_vision_dashboard():
    """Renders the Live CCTV Sessions explorer with full kinematic telemetry."""
    st.markdown("""
    <div class="telemetry-header">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; letter-spacing: 0.06em; text-transform: uppercase;">
            TELEMETRY MODULE 02
        </div>
        <h2 style="margin: 4px 0 0 0; font-size: 22px;">
            CCTV Vision Telemetry & Kinematic Analysis
        </h2>
        <p style="margin: 6px 0 0 0; color: #a1a1aa; font-size: 13px;">
            Automated vehicle trajectory logging, 4-point planar perspective homography, and adaptive intersection green-phase allocation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sessions = sorted(glob.glob("data/sessions/*"))
    if not sessions:
        st.info("No recorded CCTV sessions found in data/sessions/. Run the computer vision studio to record active sessions.")
        return

    session_names = [os.path.basename(s) for s in sessions]
    selected_sess = st.selectbox("SELECT RECORDED CCTV SESSION", session_names, index=len(session_names)-1)
    safe_sess = html.escape(str(selected_sess))

    sess_dir = os.path.join("data/sessions", selected_sess)
    obs_file = os.path.join(sess_dir, "live_traffic_observations.csv")
    mov_file = os.path.join(sess_dir, "vehicle_movement_metrics.csv")
    traj_file = os.path.join(sess_dir, "vehicle_trajectories.csv")
    metrics_file = os.path.join(sess_dir, "vision_traffic_metrics.csv")
    img_overlay = os.path.join(sess_dir, "calibration_overlay.png")

    df_obs = pd.read_csv(obs_file) if os.path.exists(obs_file) else pd.DataFrame()
    df_mov = pd.read_csv(mov_file) if os.path.exists(mov_file) else pd.DataFrame()
    df_traj = pd.read_csv(traj_file) if os.path.exists(traj_file) else pd.DataFrame()
    df_metrics = pd.read_csv(metrics_file) if os.path.exists(metrics_file) else pd.DataFrame()

    # Metric Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        total_frames = len(df_obs) if not df_obs.empty else len(df_metrics)
        st.metric("Observation Frames", f"{total_frames:,}")
    with k2:
        peak_v = int(df_obs["vehicle_count"].max()) if not df_obs.empty and "vehicle_count" in df_obs.columns else (int(df_metrics["vehicle_count"].max()) if not df_metrics.empty and "vehicle_count" in df_metrics.columns else 0)
        st.metric("Peak Vehicle Density", f"{peak_v} veh")
    with k3:
        mean_spd = df_obs["average_speed_kmh"].dropna().mean() if not df_obs.empty and "average_speed_kmh" in df_obs.columns else (df_mov["average_speed_kmh"].dropna().mean() if not df_mov.empty and "average_speed_kmh" in df_mov.columns else 28.5)
        st.metric("Mean Velocity", f"{mean_spd:.1f} km/h")
    with k4:
        unique_v = df_traj["track_id"].nunique() if not df_traj.empty and "track_id" in df_traj.columns else (len(df_mov) if not df_mov.empty else 0)
        st.metric("Unique Tracked Vehicles", f"{unique_v}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    vtab1, vtab2, vtab3 = st.tabs([
        "SIGNAL TIMING & EMISSION IMPACT",
        "VEHICLE KINEMATICS & TRAFFIC FLOW",
        "PERSPECTIVE HOMOGRAPHY & TRAJECTORIES"
    ])

    with vtab1:
        st.markdown("#### Adaptive Signal Timing Allocation")
        st.caption("Real-time translation of CCTV density metrics into optimized intersection signal cycles.")

        live_density = float(peak_v * 4.5)
        live_congestion = float(min(100.0, max(10.0, 100.0 - (mean_spd * 1.5))))
        
        live_signal = compute_optimal_signal_timing(
            congestion=live_congestion,
            vehicle_density=live_density,
            average_speed=float(mean_spd),
            zone_type="Commercial_Downtown"
        )
        live_co2 = estimate_co2_impact(
            vehicle_density=live_density,
            congestion=live_congestion,
            average_speed=float(mean_spd),
            population_density=9500
        )

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"""
            <div class="telemetry-card">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">RECOMMENDED GREEN-PHASE</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700; color: #fafafa; margin: 6px 0;">{live_signal['recommended_green_seconds']}s <span style="font-size: 14px; color: #71717a;">(Base: {live_signal['base_green_seconds']}s)</span></div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #a1a1aa;">URGENCY: {live_signal['urgency']}</div>
                <div style="margin-top: 8px; font-size: 13px; color: #d4d4d8;">{live_signal['reason']}</div>
            </div>
            """, unsafe_allow_html=True)

        with sc2:
            st.markdown(f"""
            <div class="telemetry-card">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">INTERSECTION EMISSION OFFSET</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700; color: #fafafa; margin: 6px 0;">{live_co2['potential_savings_kg_per_week']:,.0f} kg CO2/wk</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #a1a1aa;">Fuel Saved: {live_co2['fuel_saved_liters_per_week']:,.0f} L/wk • Tree Offset: {live_co2['trees_equivalent_per_year']:,} trees/yr</div>
            </div>
            """, unsafe_allow_html=True)

    with vtab2:
        st.markdown("#### Vehicle Kinematic Telemetry")
        vc1, vc2 = st.columns([3, 2])

        with vc1:
            plot_df = df_obs if not df_obs.empty and "timestamp_seconds" in df_obs.columns else df_metrics
            if not plot_df.empty and "timestamp_seconds" in plot_df.columns:
                fig_v = px.line(
                    plot_df,
                    x="timestamp_seconds",
                    y="vehicle_count",
                    title="Real-Time Vehicle Count over Video Timeline",
                    labels={"timestamp_seconds": "Elapsed Time (Seconds)", "vehicle_count": "Vehicles Detected"}
                )
                fig_v.update_layout(
                    paper_bgcolor="#18181b",
                    plot_bgcolor="#18181b",
                    font={"family": "Inter", "color": "#fafafa"},
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(gridcolor="#27272a"),
                    yaxis=dict(gridcolor="#27272a")
                )
                st.plotly_chart(fig_v, width="stretch")

        with vc2:
            if not df_metrics.empty:
                v_totals = {}
                for col, name in [("cars", "Cars"), ("motorcycles", "2-Wheelers"), ("auto_rickshaws", "Auto-Rickshaws"), ("buses", "Buses"), ("trucks", "Trucks")]:
                    if col in df_metrics.columns and df_metrics[col].sum() > 0:
                        v_totals[name] = int(df_metrics[col].sum())

                if v_totals:
                    fig_pie = px.pie(
                        names=list(v_totals.keys()),
                        values=list(v_totals.values()),
                        title="Vehicle Classification Breakdown",
                        hole=0.45,
                        color_discrete_sequence=["#fafafa", "#a1a1aa", "#71717a", "#52525b", "#3f3f46"]
                    )
                    fig_pie.update_layout(
                        paper_bgcolor="#18181b",
                        font={"family": "Inter", "color": "#fafafa"},
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_pie, width="stretch")

        if not df_mov.empty:
            st.markdown("##### Individual Vehicle Tracking Ledger")
            st.dataframe(df_mov.head(25), width="stretch")

    with vtab3:
        st.markdown("#### Perspective Homography & Trajectory Mapping")
        pc1, pc2 = st.columns([1, 1])

        with pc1:
            if os.path.exists(img_overlay):
                st.image(img_overlay, caption=f"Homography Road Plane Rectification ({safe_sess})", width="stretch")
            else:
                st.info(f"No perspective calibration overlay generated for {safe_sess}.")

        with pc2:
            if not df_traj.empty and "center_x" in df_traj.columns and "center_y" in df_traj.columns:
                fig_traj = px.scatter(
                    df_traj.head(400),
                    x="center_x",
                    y="center_y",
                    color="vehicle_type" if "vehicle_type" in df_traj.columns else None,
                    title="Tracked Vehicle Ground Trajectories",
                    labels={"center_x": "Image X (px)", "center_y": "Image Y (px)"},
                    color_discrete_sequence=["#fafafa", "#a1a1aa", "#71717a", "#38bdf8"]
                )
                fig_traj.update_yaxes(autorange="reversed", gridcolor="#27272a")
                fig_traj.update_xaxes(gridcolor="#27272a")
                fig_traj.update_layout(
                    paper_bgcolor="#18181b",
                    plot_bgcolor="#18181b",
                    font={"family": "Inter", "color": "#fafafa"},
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_traj, width="stretch")
            else:
                st.info("No ground trajectory data available for this session.")


if __name__ == "__main__":
    main()
