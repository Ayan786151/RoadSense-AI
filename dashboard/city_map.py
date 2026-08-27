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
