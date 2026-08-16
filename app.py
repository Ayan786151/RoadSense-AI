"""
================================================================================
ROAD SENSE AI - MASTER DASHBOARD & INTELLIGENT CONTROL CENTER
================================================================================

Main application entrypoint orchestrating:
1. Simulation & Risk Prediction Testing Lab (50 Municipal Zones, 52 Weeks)
2. Live Computer Vision & CCTV Telemetry Hub (session_001, session_002, session_003)
================================================================================
"""

import html
import streamlit as st

# Configure global page settings
st.set_page_config(
    page_title="RoadSense AI — Traffic Intelligence & Risk Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import simulation dashboard module
try:
    from dashboard.simulation_dashboard import render_simulation_dashboard
except ImportError:
    from simulation_dashboard import render_simulation_dashboard


def main():
    st.sidebar.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
    st.sidebar.title("RoadSense AI")
    st.sidebar.caption("Intelligent Traffic Risk & Priority Intelligence")
    st.sidebar.markdown("---")

    # Mode Selector
    app_mode = st.sidebar.radio(
        "Navigation Hub",
        [
            "🧪 Simulation & Risk Testing Lab (50 Zones, 52 Wks)",
            "📹 Live CCTV Vision & Speed Telemetry"
        ],
        index=0
    )

    st.sidebar.markdown("---")

    if app_mode.startswith("🧪"):
        render_simulation_dashboard()
    else:
        render_live_vision_dashboard()

    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="font-size: 12px; color: #9aa0a6; text-align: center;">'
        'RoadSense AI v1.0<br>'
        'Simulation • Vision • ML Risk • Priority'
        '</div>',
        unsafe_allow_html=True
    )


def render_live_vision_dashboard():
    """Renders the Live CCTV Sessions explorer."""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0f2027, #203a43, #2c5364); padding: 22px 28px; border-radius: 12px; margin-bottom: 24px; color: white;">
        <h2 style="margin: 0; font-weight: 700;">📹 Live CCTV Vision & Metric Kinematics</h2>
        <p style="margin: 6px 0 0 0; opacity: 0.88; font-size: 14px;">
            Real-time YOLO vehicle tracking, 4-point perspective homography speed estimation, and session observations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    import os
    import glob
    import pandas as pd

    sessions = sorted(glob.glob("data/sessions/*"))
    if not sessions:
        st.info("📂 No recorded CCTV sessions found in `data/sessions/`. Run the vision pipeline first to generate session data.")
        return

    session_names = [os.path.basename(s) for s in sessions]
    selected_sess = st.selectbox("Select Session", session_names, index=len(session_names)-1)

    # Sanitize session name for HTML output
    safe_sess = html.escape(str(selected_sess))

    sess_dir = os.path.join("data/sessions", selected_sess)
    obs_file = os.path.join(sess_dir, "live_traffic_observations.csv")
    mov_file = os.path.join(sess_dir, "vehicle_movement_metrics.csv")
    img_overlay = os.path.join(sess_dir, "calibration_overlay.png")

    c1, c2 = st.columns([1, 1])

    if os.path.exists(obs_file):
        df_obs = pd.read_csv(obs_file)
        with c1:
            st.markdown("#### 📊 Session Summary")
            st.metric("Total Observation Records", len(df_obs))
            if "vehicle_count" in df_obs.columns:
                st.metric("Peak Vehicle Count", int(df_obs["vehicle_count"].max()))
            if "average_speed_kmh" in df_obs.columns and df_obs["average_speed_kmh"].notnull().any():
                st.metric("Mean Speed", f"{df_obs['average_speed_kmh'].dropna().mean():.1f} km/h")
    else:
        with c1:
            st.warning(f"No observation data found for session `{safe_sess}`.")

    with c2:
        if os.path.exists(img_overlay):
            st.markdown("#### 📐 Perspective Calibration Overlay")
            st.image(img_overlay, caption=f"Homography Quad Overlay ({safe_sess})", use_container_width=True)

    if os.path.exists(mov_file):
        df_mov = pd.read_csv(mov_file)
        st.markdown("#### 🚗 Tracked Vehicle Movement Details")
        st.dataframe(df_mov.head(30), use_container_width=True)


if __name__ == "__main__":
    main()
