"""
================================================================================
ROADSENSE AI — GEOSPATIAL CITY MAP & COMMAND RADAR
DESIGN SYSTEM: KINETIC INFRASTRUCTURE INTELLIGENCE (STITCH MCP)
================================================================================
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path


# Center coordinates (Bengaluru urban metro reference grid)
CITY_LAT_CENTER = 12.9716
CITY_LON_CENTER = 77.5946


def generate_zone_coordinates(num_zones: int = 50) -> pd.DataFrame:
    """Generates or loads geospatial coordinate grid for municipal zones."""
    loc_file = Path("data/location_mapping.csv")
    if loc_file.exists():
        loc_df = pd.read_csv(loc_file)
        loc_df["zone_id"] = loc_df["zone_id"].astype(str)
        loc_df["zone_name"] = loc_df["zone_id"] + ": " + loc_df["location_name"] + " (" + loc_df["city"] + ")"
        return loc_df[["zone_id", "latitude", "longitude", "zone_name"]]

    np.random.seed(42)
    zones = []
    
    # Fallback synthetic grid
    for z in range(1, num_zones + 1):
        radius = np.random.uniform(0.01, 0.12)
        angle = np.random.uniform(0, 2 * np.pi)
        lat = CITY_LAT_CENTER + radius * np.sin(angle)
        lon = CITY_LON_CENTER + radius * np.cos(angle)
        
        zones.append({
            "zone_id": f"Zone_{z:02d}",
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "zone_name": f"Zone {z:02d} ({'CBD Sector' if radius < 0.04 else ('Ring Highway' if radius > 0.08 else 'Metro Corridor')})"
        })
    return pd.DataFrame(zones)


def render_city_command_map():
    """Renders the Geospatial City Command Map tab in Streamlit."""
    st.markdown("""
    <div class="telemetry-header">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; letter-spacing: 0.06em; text-transform: uppercase;">
            GEOSPATIAL RADAR 04
        </div>
        <h2 style="margin: 4px 0 0 0; font-size: 22px;">
            Geospatial Urban Risk Radar & Municipal Mapping
        </h2>
        <p style="margin: 6px 0 0 0; color: #a1a1aa; font-size: 13px;">
            Spatial risk density analysis, congestion contours, and priority surveillance across 50 municipal sectors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #18181b; border: 1px solid #27272a; border-radius: 6px; padding: 18px 20px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #38bdf8; letter-spacing: 0.08em; text-transform: uppercase;">
                JUDGING CRITERIA • HAZARD FACTOR RISK WEIGHTAGE & RELATIVE IMPACT HIERARCHY
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #a1a1aa; background: #27272a; padding: 2px 8px; border-radius: 3px;">
                TOTAL WEIGHT = 100%
            </div>
        </div>
        <p style="font-size: 12px; color: #a1a1aa; margin: 4px 0 14px 0;">
            <b>Why factors have different weightages:</b> Over-speeding and red-light violations carry <b>3.5× more weight than static road congestion</b> because kinetic impact energy scales quadratically (<code style="color: #38bdf8;">E_k = ½mv²</code>). Congestion causes delays but slow-moving vehicles rarely result in fatal crashes.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
            <div style="background: #121215; border: 1px solid #ef4444; border-left: 4px solid #ef4444; padding: 12px 14px; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #ef4444; font-weight: 700;">TIER 1 • CRITICAL</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 800; color: #ef4444;">38%</span>
                </div>
                <div style="font-size: 13px; font-weight: 700; color: #fafafa; margin-top: 4px;">⚡ Over-Speeding & Velocity Variance</div>
                <div style="font-size: 11px; color: #a1a1aa; margin-top: 4px;">High speeds & erratic braking; #1 determinant of fatal collision impact energy.</div>
            </div>
            <div style="background: #121215; border: 1px solid #f59e0b; border-left: 4px solid #f59e0b; padding: 12px 14px; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #f59e0b; font-weight: 700;">TIER 2 • HIGH</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 800; color: #f59e0b;">26%</span>
                </div>
                <div style="font-size: 13px; font-weight: 700; color: #fafafa; margin-top: 4px;">🚦 Red-Light & Stop-Line Violations</div>
                <div style="font-size: 11px; color: #a1a1aa; margin-top: 4px;">Intersection intrusions directly triggering severe right-angle T-bone collisions.</div>
            </div>
            <div style="background: #121215; border: 1px solid #818cf8; border-left: 4px solid #818cf8; padding: 12px 14px; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #818cf8; font-weight: 700;">TIER 3 • MODERATE-HIGH</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 800; color: #818cf8;">20%</span>
                </div>
                <div style="font-size: 13px; font-weight: 700; color: #fafafa; margin-top: 4px;">🛡️ Helmet Absence & Triple-Riding</div>
                <div style="font-size: 11px; color: #a1a1aa; margin-top: 4px;">Vulnerable 2-wheeler user exposure (accounts for 70%+ of urban casualties).</div>
            </div>
            <div style="background: #121215; border: 1px solid #27272a; border-left: 4px solid #38bdf8; padding: 12px 14px; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #38bdf8; font-weight: 700;">TIER 4 • MODERATE</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 800; color: #38bdf8;">11%</span>
                </div>
                <div style="font-size: 13px; font-weight: 700; color: #fafafa; margin-top: 4px;">🚗 Road Congestion & Queue Backlog</div>
                <div style="font-size: 11px; color: #a1a1aa; margin-top: 4px;">High commuter delays but low vehicle speeds; lower kinetic crash severity.</div>
            </div>
            <div style="background: #121215; border: 1px solid #27272a; border-left: 4px solid #71717a; padding: 12px 14px; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #a1a1aa; font-weight: 700;">TIER 5 • BASELINE</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 800; color: #a1a1aa;">5%</span>
                </div>
                <div style="font-size: 13px; font-weight: 700; color: #fafafa; margin-top: 4px;">🌧️ Weather & Low-Light Visibility</div>
                <div style="font-size: 11px; color: #a1a1aa; margin-top: 4px;">Frictional surface grip modifier & night glare (enhanced via CLAHE).</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Load latest week simulation / predictions data
    data_path = Path("data/simulation_temporal_features.csv")
    if not data_path.exists():
        st.warning("Simulation dataset not found in data/. Please generate simulation data first.")
        return

    df_sim = pd.read_csv(data_path)
    latest_week = int(df_sim["week"].max())

    m_col1, m_col2, m_col3 = st.columns([1, 1, 2])
    with m_col1:
        sel_week = st.slider("TIMELINE WEEK", 1, latest_week, latest_week)
    with m_col2:
        metric_choice = st.selectbox("RADAR METRIC", ["Congestion Score", "Vehicle Density", "Predicted Risk Score"])

    week_data = df_sim[df_sim["week"] == sel_week].copy()
    week_data["zone_id"] = week_data["zone_id"].astype(str)

    coords_df = generate_zone_coordinates(len(week_data))
    coords_df["zone_id"] = coords_df["zone_id"].astype(str)

    # Clean merge with deduplicated columns
    cols_to_drop = [c for c in ["latitude", "longitude", "zone_name"] if c in week_data.columns]
    if cols_to_drop:
        week_data = week_data.drop(columns=cols_to_drop)

    merged = pd.merge(week_data, coords_df, on="zone_id", how="left")

    # Map metric to column
    col_map = {
        "Congestion Score": "congestion",
        "Vehicle Density": "vehicle_density",
        "Predicted Risk Score": "incident_occurred"
    }
    target_col = col_map.get(metric_choice, "congestion")

    fig = px.scatter_mapbox(
        merged,
        lat="latitude",
        lon="longitude",
        size="vehicle_density",
        color=target_col,
        color_continuous_scale="Viridis",
        size_max=22,
        zoom=11,
        mapbox_style="open-street-map",
        hover_name="zone_name",
        hover_data={
            "latitude": False,
            "longitude": False,
            "zone_id": True,
            "vehicle_density": ":.0f veh/km",
            "congestion": ":.1f",
            "incident_occurred": True
        },
        title=f"Geospatial Risk Radar — Week {sel_week}"
    )

    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=540,
        paper_bgcolor="#18181b",
        font={"family": "Inter", "color": "#fafafa"},
        mapbox=dict(
            style="open-street-map"
        )
    )

    st.plotly_chart(fig, width="stretch")

    # Hotspot summary priority table
    st.markdown("#### High-Priority Civil Intervention Sectors")
    top_sectors = merged.sort_values(by=target_col, ascending=False).head(5)[["zone_id", "zone_name", "vehicle_density", "congestion", "incident_occurred"]].copy()
    top_sectors.columns = ["ZONE ID", "SECTOR LOCATION", "VEHICLE DENSITY (VEH/KM)", "CONGESTION INDEX", "INCIDENT HISTORY"]
    top_sectors.index = range(1, len(top_sectors) + 1)
    top_sectors.index.name = "PRIORITY"
    
    st.dataframe(top_sectors, width="stretch")
