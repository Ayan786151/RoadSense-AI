"""
================================================================================
ROAD SENSE AI - MASTER DASHBOARD & INTELLIGENT CONTROL CENTER
================================================================================

Main application entrypoint orchestrating:
1. Simulation & Risk Prediction Testing Lab (50 Municipal Zones, 52 Weeks)
   - Powered by Supervised ML, Adaptive Signal Control & AI Briefings
2. Live Computer Vision & CCTV Telemetry Hub (session_001, session_002, session_003)
   - Real-time YOLO vehicle tracking, homography velocity estimation, and live signal recommendation
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
    page_title="RoadSense AI — Traffic Intelligence & Civic Social Service",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import simulation dashboard, vision studio, and intelligence modules
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
    st.sidebar.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
    st.sidebar.title("RoadSense AI")
    st.sidebar.caption("Intelligent Traffic Risk & Civic Social Service Hub")
    st.sidebar.markdown("---")

    # Mode Selector
    app_mode = st.sidebar.radio(
        "Navigation Hub",
        [
            "🧪 Simulation & Risk Testing Lab (50 Zones, 52 Wks)",
            "📹 Live CCTV Vision & Speed Telemetry",
            "🚀 Interactive Vision Studio & AI Violation Hub",
            "🗺️ Geospatial 3D City Risk Map & Command Room"
        ],
        index=0
    )

    st.sidebar.markdown("---")

    if app_mode.startswith("🧪"):
        render_simulation_dashboard()
    elif app_mode.startswith("📹"):
        render_live_vision_dashboard()
    elif app_mode.startswith("🚀"):
        render_vision_studio()
    else:
        render_city_command_map()

    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="font-size: 12px; color: #9aa0a6; text-align: center;">'
        'RoadSense AI v2.0 • Social Service Edition<br>'
        'YOLO Vision • ML Risk • Signal Timing • Eco Impact'
        '</div>',
        unsafe_allow_html=True
    )


def render_live_vision_dashboard():
    """Renders the Live CCTV Sessions explorer with full kinematic telemetry."""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0f2027, #203a43, #2c5364); padding: 22px 28px; border-radius: 12px; margin-bottom: 24px; color: white;">
        <h2 style="margin: 0; font-weight: 700; display: flex; align-items: center; gap: 12px;">
            📹 Live CCTV Computer Vision Telemetry & Intersection Intelligence
        </h2>
        <p style="margin: 6px 0 0 0; opacity: 0.88; font-size: 14px;">
            Real-time YOLO vehicle detection, 4-point perspective homography velocity estimation, and live adaptive signal recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sessions = sorted(glob.glob("data/sessions/*"))
    if not sessions:
        st.info("📂 No recorded CCTV sessions found in `data/sessions/`. Run the vision pipeline first to generate session data.")
        return

    session_names = [os.path.basename(s) for s in sessions]
    selected_sess = st.selectbox("📹 Select CCTV Camera Feed / Session", session_names, index=len(session_names)-1)
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

    # --- Quick KPI Summary Row ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        total_frames = len(df_obs) if not df_obs.empty else len(df_metrics)
        st.metric("Observation Frames", f"{total_frames:,}")
    with k2:
        peak_v = int(df_obs["vehicle_count"].max()) if not df_obs.empty and "vehicle_count" in df_obs.columns else (int(df_metrics["vehicle_count"].max()) if not df_metrics.empty and "vehicle_count" in df_metrics.columns else 0)
        st.metric("Peak Vehicle Density", f"{peak_v} vehicles")
    with k3:
        mean_spd = df_obs["average_speed_kmh"].dropna().mean() if not df_obs.empty and "average_speed_kmh" in df_obs.columns else (df_mov["average_speed_kmh"].dropna().mean() if not df_mov.empty and "average_speed_kmh" in df_mov.columns else 28.5)
        st.metric("Mean Velocity", f"{mean_spd:.1f} km/h")
    with k4:
        unique_v = df_traj["track_id"].nunique() if not df_traj.empty and "track_id" in df_traj.columns else (len(df_mov) if not df_mov.empty else 0)
        st.metric("Tracked Unique Vehicles", f"{unique_v}")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Live Vision Tabs ---
    vtab1, vtab2, vtab3 = st.tabs([
        "🚦 CCTV Live Adaptive Signal & Eco Savings",
        "📊 Vehicle Kinematics & Traffic Flow",
        "📐 Perspective Calibration & Trajectories"
    ])

    with vtab1:
        st.markdown("#### ⚡ Live Video-to-Signal Control Bridge")
        st.caption("How live CCTV detection immediately feeds into adaptive traffic light timing for this intersection.")

        # Compute live adaptive timing from camera session
        live_density = float(peak_v * 4.5)  # Scale to urban density score
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
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(77, 171, 247, 0.3); border-radius: 10px; padding: 18px;">
                <div style="font-size: 13px; color: #9aa0a6;">CCTV-DERIVED GREEN LIGHT TIMING</div>
                <div style="font-size: 38px; font-weight: 700; color: #69db7c; margin: 8px 0;">{live_signal['recommended_green_seconds']}s <span style="font-size: 16px; color: #868e96;">(Base: {live_signal['base_green_seconds']}s)</span></div>
                <div style="font-size: 14px; font-weight: 600; color: #ffd43b;">Urgency: {live_signal['urgency']}</div>
                <p style="margin: 8px 0 0 0; font-size: 13px; color: #ced4da;">{live_signal['reason']}</p>
            </div>
            """, unsafe_allow_html=True)

        with sc2:
            st.markdown(f"""
            <div style="background: rgba(40, 167, 69, 0.08); border: 1px solid #28a745; border-radius: 10px; padding: 18px;">
                <div style="font-size: 13px; color: #69db7c; font-weight: 700;">🌱 INTERSECTION EMISSION OFFSET</div>
                <div style="font-size: 28px; font-weight: 700; color: #fff; margin: 8px 0;">{live_co2['potential_savings_kg_per_week']:,.0f} kg CO2/wk</div>
                <div style="font-size: 13px; color: #e9ecef;">Fuel Saved: <b>{live_co2['fuel_saved_liters_per_week']:,.0f} L/wk</b> • Tree Offset: <b>{live_co2['trees_equivalent_per_year']:,} trees/yr</b></div>
            </div>
            """, unsafe_allow_html=True)

    with vtab2:
        st.markdown("#### 📊 Vehicle Kinematics & Velocity Telemetry")
        vc1, vc2 = st.columns([3, 2])

        with vc1:
            if not df_obs.empty and "timestamp_seconds" in df_obs.columns and "vehicle_count" in df_obs.columns:
                fig_v = px.line(
                    df_obs,
                    x="timestamp_seconds",
                    y="vehicle_count",
                    title="Real-Time Vehicle Count over Video Timeline",
                    labels={"timestamp_seconds": "Elapsed Time (Seconds)", "vehicle_count": "Vehicles Detected"},
                    template="plotly_dark"
                )
                fig_v.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_v, use_container_width=True)
            elif not df_metrics.empty and "timestamp_seconds" in df_metrics.columns:
                fig_v = px.line(
                    df_metrics,
                    x="timestamp_seconds",
                    y="vehicle_count",
                    title="Real-Time Vehicle Count over Video Timeline",
                    labels={"timestamp_seconds": "Elapsed Time (Seconds)", "vehicle_count": "Vehicles Detected"},
                    template="plotly_dark"
                )
                fig_v.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_v, use_container_width=True)

        with vc2:
            # Vehicle type composition
            if not df_metrics.empty:
                v_totals = {}
                if "cars" in df_metrics.columns and df_metrics["cars"].sum() > 0:
                    v_totals["Cars"] = int(df_metrics["cars"].sum())
                if "motorcycles" in df_metrics.columns and df_metrics["motorcycles"].sum() > 0:
                    v_totals["Two-Wheelers"] = int(df_metrics["motorcycles"].sum())
                if "auto_rickshaws" in df_metrics.columns and df_metrics["auto_rickshaws"].sum() > 0:
                    v_totals["Auto-Rickshaws"] = int(df_metrics["auto_rickshaws"].sum())
                if "buses" in df_metrics.columns and df_metrics["buses"].sum() > 0:
                    v_totals["Buses"] = int(df_metrics["buses"].sum())
                if "trucks" in df_metrics.columns and df_metrics["trucks"].sum() > 0:
                    v_totals["Trucks / LCVs"] = int(df_metrics["trucks"].sum())

                if v_totals:
                    fig_pie = px.pie(
                        names=list(v_totals.keys()),
                        values=list(v_totals.values()),
                        title="Vehicle Classification Breakdown",
                        template="plotly_dark",
                        hole=0.4,
                        color_discrete_sequence=["#4dabf7", "#ffd43b", "#51cf66", "#ff6b6b", "#da77f2"]
                    )
                    fig_pie.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_pie, use_container_width=True)

        if not df_mov.empty:
            st.markdown("##### 🚗 Individual Tracked Vehicle Kinematics Data")
            st.dataframe(df_mov.head(25), use_container_width=True)

    with vtab3:
        st.markdown("#### 📐 Perspective Homography & Trajectory Overlay")
        pc1, pc2 = st.columns([1, 1])

        with pc1:
            if os.path.exists(img_overlay):
                st.image(img_overlay, caption=f"4-Point Homography Road Plane Quad ({safe_sess})", use_container_width=True)
            else:
                st.info(f"No perspective calibration overlay found for `{safe_sess}`.")

        with pc2:
            if not df_traj.empty and "center_x" in df_traj.columns and "center_y" in df_traj.columns:
                fig_traj = px.scatter(
                    df_traj.head(400),
                    x="center_x",
                    y="center_y",
                    color="vehicle_type" if "vehicle_type" in df_traj.columns else None,
                    title="Tracked Vehicle Road-Contact Trajectories",
                    labels={"center_x": "Image X (Pixels)", "center_y": "Image Y (Road Plane)"},
                    template="plotly_dark"
                )
                fig_traj.update_yaxes(autorange="reversed")  # Invert Y to match image space
                fig_traj.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_traj, use_container_width=True)
            else:
                st.info("No trajectory coordinates available for this session.")


if __name__ == "__main__":
    main()
