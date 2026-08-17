"""
================================================================================
ROAD SENSE AI - GEOSPATIAL GAUSSIAN HEATMAP & LUMINOUS ARTERIAL RADAR
MODULE: CONTINUOUS GAUSSIAN DENSITY HEATMAP & NEON LIGHT TRAFFIC CORRIDORS
================================================================================

Renders high-aesthetic continuous traffic risk visualization:
1. Continuous Gaussian Kernel Blur Density Heatmap (smooth gradient radiation)
2. Stretches of Neon Light Trails along urban arterial road corridors
3. City selector (Mumbai, Delhi, Bengaluru, Lucknow, or Pan-India Command)
4. Hotspot danger telemetry and live civil intervention radar
================================================================================
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path


# Neon color palettes for continuous Gaussian diffusion & light stretches
COLOR_PALETTES = {
    "Cyberpunk Neon (Electric Amber & Magenta)": [
        [0.0, "rgba(15, 23, 42, 0.0)"],
        [0.2, "rgba(56, 189, 248, 0.4)"],
        [0.4, "rgba(251, 191, 36, 0.7)"],
        [0.7, "rgba(244, 63, 94, 0.85)"],
        [1.0, "rgba(255, 255, 255, 0.98)"]
    ],
    "Thermal Infrared Radar (Classic FLIR)": [
        [0.0, "rgba(0, 0, 0, 0.0)"],
        [0.2, "rgba(30, 64, 175, 0.4)"],
        [0.45, "rgba(16, 185, 129, 0.7)"],
        [0.75, "rgba(239, 68, 68, 0.88)"],
        [1.0, "rgba(254, 240, 138, 0.98)"]
    ],
    "Deep-Sea Bioluminescence (Cyan to Lime Glow)": [
        [0.0, "rgba(2, 6, 23, 0.0)"],
        [0.25, "rgba(6, 182, 212, 0.4)"],
        [0.55, "rgba(45, 212, 191, 0.75)"],
        [0.85, "rgba(132, 204, 22, 0.9)"],
        [1.0, "rgba(255, 255, 255, 0.98)"]
    ],
    "Golden Ember (High-Density Flow)": [
        [0.0, "rgba(10, 10, 10, 0.0)"],
        [0.3, "rgba(180, 83, 9, 0.45)"],
        [0.6, "rgba(245, 158, 11, 0.8)"],
        [0.85, "rgba(239, 68, 68, 0.9)"],
        [1.0, "rgba(254, 243, 199, 1.0)"]
    ]
}


def load_locations_and_data(selected_week: int):
    """Loads location mappings and joins with multi-week simulation metrics."""
    data_path = Path("data/simulation_temporal_features.csv")
    loc_path = Path("data/location_mapping.csv")

    if not data_path.exists() or not loc_path.exists():
        return None

    df_sim = pd.read_csv(data_path)
    df_loc = pd.read_csv(loc_path)

    df_sim["zone_id"] = df_sim["zone_id"].astype(str)
    df_loc["zone_id"] = df_loc["zone_id"].astype(str)

    week_df = df_sim[df_sim["week"] == selected_week].copy()

    # Drop any overlapping geo columns in simulation csv
    for col in ["latitude", "longitude", "city", "location_name"]:
        if col in week_df.columns:
            week_df = week_df.drop(columns=[col])

    merged = pd.merge(week_df, df_loc, on="zone_id", how="left")
    merged["display_name"] = merged["zone_id"] + " • " + merged["location_name"] + " (" + merged["city"] + ")"
    return merged


def generate_corridor_light_stretches(df_city: pd.DataFrame, target_metric: str = "congestion", num_points: int = 15):
    """
    Interpolates continuous luminous light stretches connecting arterial nodes
    along primary urban transit corridors.
    """
    if len(df_city) < 2:
        return [], []

    # Sort spatially to form logical arterial corridors
    sorted_df = df_city.sort_values(by=["latitude", "longitude"]).reset_index(drop=True)
    
    corridor_lats = []
    corridor_lons = []
    corridor_weights = []

    for i in range(len(sorted_df) - 1):
        p1 = sorted_df.iloc[i]
        p2 = sorted_df.iloc[i + 1]

        # Generate interpolated stretch of light between node pairs
        lats = np.linspace(p1["latitude"], p2["latitude"], num_points)
        lons = np.linspace(p1["longitude"], p2["longitude"], num_points)
        weights = np.linspace(p1[target_metric], p2[target_metric], num_points)

        corridor_lats.extend(lats)
        corridor_lons.extend(lons)
        corridor_weights.extend(weights)

    return corridor_lats, corridor_lons, corridor_weights


def render_city_command_map():
    """Renders the Gaussian Heatmap & Neon Light Corridors Command Map."""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #090d16, #111827, #1e293b); padding: 22px 28px; border-radius: 12px; margin-bottom: 24px; color: white; border: 1px solid #1e293b; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
        <h2 style="margin: 0; font-weight: 800; display: flex; align-items: center; gap: 12px; letter-spacing: 0.5px;">
            🌌 Geospatial Gaussian Heatmap & Luminous Arterial Radar
        </h2>
        <p style="margin: 6px 0 0 0; opacity: 0.85; font-size: 14px;">
            Continuous <b>Gaussian-Kernel Blurred Density Radiation</b> & <b>Luminous Neon Traffic Stretches</b> visualizing live urban incident risk across city sectors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    data_path = Path("data/simulation_temporal_features.csv")
    if not data_path.exists():
        st.warning("Simulation data not found. Please run the simulation first.")
        return

    df_sim = pd.read_csv(data_path)
    latest_week = int(df_sim["week"].max())

    # Control Bar
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    with c1:
        sel_week = st.slider("Timeline Week", 1, latest_week, latest_week)
    with c2:
        metric_choice = st.selectbox(
            "Telemetry Variable",
            ["Congestion Score (0-100)", "Vehicle Density (veh/km)", "Predicted Risk Probability (P)"]
        )
    with c3:
        blur_radius = st.slider("Gaussian Blur Radius", 15, 60, 32, help="Controls the continuous Gaussian kernel spatial diffusion radius.")
    with c4:
        theme_name = st.selectbox("Visual Glow Theme", list(COLOR_PALETTES.keys()), index=0)

    # Resolve metric column
    if "Congestion" in metric_choice:
        target_col = "congestion"
        metric_label = "Congestion"
    elif "Density" in metric_choice:
        target_col = "vehicle_density"
        metric_label = "Vehicles/km"
    else:
        target_col = "incident_occurred"
        metric_label = "Risk Probability"

    merged = load_locations_and_data(sel_week)
    if merged is None:
        st.error("Could not load location coordinates. Check `data/location_mapping.csv`.")
        return

    # City filter selector
    cities = ["All Metro Hubs (Pan-India)"] + sorted(merged["city"].dropna().unique().tolist())
    sel_city = st.selectbox("Select Urban Focus Area", cities, index=0)

    if sel_city != "All Metro Hubs (Pan-India)":
        plot_df = merged[merged["city"] == sel_city].copy()
        center_lat = float(plot_df["latitude"].mean())
        center_lon = float(plot_df["longitude"].mean())
        zoom_level = 11.8
    else:
        plot_df = merged.copy()
        center_lat = 22.5937
        center_lon = 78.9629
        zoom_level = 4.3

    # Generate interpolated arterial light stretches along road corridors
    stretch_lats, stretch_lons, stretch_z = generate_corridor_light_stretches(plot_df, target_metric=target_col, num_points=20)

    # Combine node points + corridor stretch points for continuous density radiation
    all_lats = plot_df["latitude"].tolist() + stretch_lats
    all_lons = plot_df["longitude"].tolist() + stretch_lons
    all_z = plot_df[target_col].tolist() + stretch_z

    selected_colorscale = COLOR_PALETTES[theme_name]

    # Create Plotly Figure with Gaussian Heatmap & Neon Light Corridors
    fig = go.Figure()

    # Layer 1: Continuous Gaussian Blur Density Heatmap (Diffused Radiation Field)
    fig.add_trace(go.Densitymapbox(
        lat=all_lats,
        lon=all_lons,
        z=all_z,
        radius=blur_radius,
        colorscale=selected_colorscale,
        showscale=True,
        colorbar=dict(
            title=dict(text=metric_label, font=dict(color="#f8fafc", size=12)),
            tickfont=dict(color="#94a3b8"),
            len=0.75,
            x=0.98,
            thickness=14
        ),
        hoverinfo="skip"
    ))

    # Layer 2: Glowing Neon Arterial Light Stretches (Luminous Highway Corridors)
    if len(plot_df) > 1 and sel_city != "All Metro Hubs (Pan-India)":
        sorted_nodes = plot_df.sort_values(by=["latitude", "longitude"])
        corridor_lat_line = []
        corridor_lon_line = []
        for i in range(len(sorted_nodes) - 1):
            corridor_lat_line.extend([sorted_nodes.iloc[i]["latitude"], sorted_nodes.iloc[i+1]["latitude"], None])
            corridor_lon_line.extend([sorted_nodes.iloc[i]["longitude"], sorted_nodes.iloc[i+1]["longitude"], None])

        # Outer soft luminous halo beam
        fig.add_trace(go.Scattermapbox(
            lat=corridor_lat_line,
            lon=corridor_lon_line,
            mode="lines",
            line=dict(width=7, color="rgba(251, 191, 36, 0.35)"),
            hoverinfo="skip",
            showlegend=False
        ))

        # Inner intense neon core light stretch
        fig.add_trace(go.Scattermapbox(
            lat=corridor_lat_line,
            lon=corridor_lon_line,
            mode="lines",
            line=dict(width=2.5, color="#ffffff"),
            hoverinfo="skip",
            showlegend=False
        ))

    # Layer 3: Interactive Surveillance Camera Pin Nodes
    fig.add_trace(go.Scattermapbox(
        lat=plot_df["latitude"],
        lon=plot_df["longitude"],
        mode="markers",
        marker=dict(
            size=10,
            color="#38bdf8",
            opacity=0.9
        ),
        text=plot_df["display_name"],
        customdata=np.stack((
            plot_df["zone_id"],
            plot_df["vehicle_density"],
            plot_df["congestion"],
            plot_df["incident_occurred"]
        ), axis=-1),
        hovertemplate="<b>%{text}</b><br><br>" +
                      "🚗 Vehicle Density: %{customdata[1]:.0f} veh/km<br>" +
                      "🚦 Congestion Level: %{customdata[2]:.1f}<br>" +
                      "🚨 Incident Risk (y): %{customdata[3]}<br>" +
                      "<extra></extra>",
        showlegend=False
    ))

    # Dark-Matter Mapbox Style
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_level
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=620,
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Bottom Hotspot Telemetry Dashboard
    st.markdown("#### 🚨 Top Civil Priority Sectors & Arterial Corridors")
    top3 = plot_df.sort_values(by=target_col, ascending=False).head(3)

    h_col1, h_col2, h_col3 = st.columns(3)
    for idx, (c, (_, row)) in enumerate(zip([h_col1, h_col2, h_col3], top3.iterrows())):
        with c:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid #ef4444; border-radius: 8px; padding: 14px; margin-bottom: 10px; border-top: 1px solid #1e293b; border-right: 1px solid #1e293b; border-bottom: 1px solid #1e293b;">
                <div style="font-weight: 700; color: #f87171; font-size: 14px;">#{idx+1} {row['display_name']}</div>
                <div style="font-size: 12px; color: #94a3b8; margin: 4px 0;">Density: <b>{row['vehicle_density']:.0f} veh/km</b> | Congestion: <b>{row['congestion']:.1f}</b></div>
                <div style="font-size: 11px; color: #cbd5e1; background: rgba(239, 68, 68, 0.18); padding: 4px 8px; border-radius: 4px; display: inline-block;">
                    Status: {'CRITICAL CONGESTION' if row['congestion'] > 60 else 'ELEVATED FLOW'}
                </div>
            </div>
            """, unsafe_allow_html=True)
