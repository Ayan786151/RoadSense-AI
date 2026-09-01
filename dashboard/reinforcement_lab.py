"""
================================================================================
ROADSENSE AI — 4-WEEK ROLLING ML RISK PREDICTOR & REINFORCEMENT LAB
MODULE: dashboard/reinforcement_lab.py
DESIGN SYSTEM: KINETIC INFRASTRUCTURE INTELLIGENCE (STITCH MCP / ZINC BRUTALISM)
================================================================================
"""

import os
import json
import html
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Tuple, Optional, Any, List

from models.reinforcement_engine import (
    load_temporal_dataset,
    load_or_train_risk_model,
    evaluate_single_5th_week,
    run_walk_forward_reinforcement_simulation,
    compute_reinforcement_summary,
    calculate_reinforcement_signal,
    get_feature_columns,
    auto_detect_dataset_columns,
    run_custom_dataset_reinforcement_pipeline,
    train_with_self_correcting_loop,
    TARGET_COL
)
from data.database_manager import (
    get_factor_intelligence_summary,
    query_master_database,
    FACTOR_TAXONOMY
)
from data.crash_dataset_adapter import (
    transform_crash_records_to_4week_panel,
    CRASH_FACTOR_DEFINITIONS
)

try:
    from backend.temporal_organizer import (
        find_input_csv_files,
        organize_csv_dataset,
        INPUT_CSV_DIR,
        ORGANIZED_DIR,
        DAY_NAMES
    )
    from backend.train_and_evaluate import (
        train_and_evaluate_model,
        MASTER_CSV_PATH
    )
except ImportError:
    try:
        from temporal_organizer import (
            find_input_csv_files,
            organize_csv_dataset,
            INPUT_CSV_DIR,
            ORGANIZED_DIR,
            DAY_NAMES
        )
        from train_and_evaluate import (
            train_and_evaluate_model,
            MASTER_CSV_PATH
        )
    except ImportError:
        pass


# ==============================================================================
# 1. CACHED DATA & MODEL LOADERS
# ==============================================================================

@st.cache_data
def get_reinforcement_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads temporal simulation dataset and location mapping."""
    sim_path = "data/simulation_temporal_features.csv"
    if not os.path.exists(sim_path):
        sim_path = "data/temporal_features.csv"
    loc_path = "data/location_mapping.csv"

    sim_df = load_temporal_dataset(sim_path)

    if os.path.exists(loc_path) and "location_name" not in sim_df.columns:
        loc_df = pd.read_csv(loc_path)
        merged = pd.merge(sim_df, loc_df, on="zone_id", how="left")
    else:
        merged = sim_df.copy()
        if "location_name" not in merged.columns:
            merged["location_name"] = merged["zone_id"]
        if "city" not in merged.columns:
            merged["city"] = "Metropolis"
        if "latitude" not in merged.columns:
            merged["latitude"] = 19.0760
        if "longitude" not in merged.columns:
            merged["longitude"] = 72.8777
        loc_df = pd.DataFrame()

    return merged, loc_df


@st.cache_resource
def get_risk_model():
    """Loads or trains the Supervised ML Risk Model pipeline."""
    model_path = "models/best_risk_model.pkl"
    data_path = "data/simulation_temporal_features.csv"
    return load_or_train_risk_model(model_path, data_path)


@st.cache_data
def get_walk_forward_simulation_results(_model, _df: pd.DataFrame, threshold: float = 0.50) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs and caches multi-month walk-forward reinforcement evaluation."""
    eval_csv = "data/reinforcement_evaluation.csv"
    
    # If already computed with default threshold, load directly
    if threshold == 0.50 and os.path.exists(eval_csv) and os.path.getsize(eval_csv) > 5000:
        results_df = pd.read_csv(eval_csv)
    else:
        results_df = run_walk_forward_reinforcement_simulation(_model, _df, start_week=5, end_week=52, threshold=threshold)
    
    summary = compute_reinforcement_summary(results_df)
    return results_df, summary


# ==============================================================================
# 2. UI HELPER COMPONENTS (ZINC / EDITORIAL BRUTALISM)
# ==============================================================================

def render_kpi_card(label: str, value: str, subtext: str, val_color: str = "#fafafa", border_color: str = "#27272a"):
    """Sleek minimalist KPI tile."""
    st.markdown(f"""
    <div style="background: #18181b; border: 1px solid {border_color}; border-radius: 4px; padding: 14px 18px; margin-bottom: 12px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.05em;">{html.escape(label)}</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; color: {val_color}; margin: 4px 0;">{value}</div>
        <div style="font-family: 'Inter', sans-serif; font-size: 12px; color: #71717a;">{html.escape(subtext)}</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 3. MAIN DASHBOARD RENDERER
# ==============================================================================

def render_reinforcement_lab():
    """Renders the 4-Week Rolling Input -> 5th-Week ML Predictor & Reinforcement Command Center."""
    st.markdown("""
    <div class="telemetry-header">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; letter-spacing: 0.08em; text-transform: uppercase;">
            CIVIL TELEMETRY MODULE 08
        </div>
        <h2 style="margin: 4px 0 0 0; font-size: 22px; font-weight: 700; color: #fafafa; letter-spacing: -0.02em;">
            4-Week Rolling ML Risk Predictor & Reinforcement Feedback Lab
        </h2>
        <p style="margin: 6px 0 0 0; color: #a1a1aa; font-size: 13px; max-width: 1050px; line-height: 1.5;">
            Sequential 4-week window feeder with environmental circumstance modulation, 5th-week risk inference, ground-truth comparison, and positive/negative reinforcement rewards for continual learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df_full, loc_df = get_reinforcement_datasets()
    model = get_risk_model()

    if model is None:
        st.error("ML Risk Model could not be loaded. Please ensure models/best_risk_model.pkl exists.")
        return

    tab_factor_profiler, tab_backend_organizer, tab_step_inspector, tab_walk_forward, tab_custom_db, tab_self_correct = st.tabs([
        " DATASET INTELLIGENCE & FACTOR PROFILER",
        " BACKEND CSV ORGANIZER & ACCURACY LAB",
        " 4-WEEK ROLLING STEP INSPECTOR",
        " MULTI-MONTH WALK-FORWARD BENCHMARK",
        " CUSTOM DATABASE EVALUATOR",
        " SELF-CORRECTING TRAINING LAB"
    ])

    # --------------------------------------------------------------------------
    # TAB 0: DATASET INTELLIGENCE & FACTOR PROFILER
    # --------------------------------------------------------------------------
    with tab_factor_profiler:
        st.markdown("####  Master Database Factor Dictionary & Dataset Intelligence Studio")
        st.caption("Inspect, analyze, and understand all 56 multi-modal traffic variables, environmental factors, vehicle composition mixes, kinematic states, and temporal momentum features before feeding them into the ML training pipeline.")

        # Load Master DB & Factor Taxonomy
        df_master, factor_dict = get_factor_intelligence_summary()

        # Master Database KPI Summary Cards
        fk1, fk2, fk3, fk4 = st.columns(4)
        with fk1:
            render_kpi_card("Consolidated Factors", f"{len(df_master.columns)} Variables", "All Factors Organized in Master DB", "#38bdf8")
        with fk2:
            render_kpi_card("Master Observations", f"{len(df_master):,} Corridors", "50 Zones × 52 Weeks (Full Year)", "#22c55e")
        with fk3:
            render_kpi_card("SQLite Database", "road_sense_master.db", "Indexed for Sub-Millisecond SQL Queries", "#f59e0b")
        with fk4:
            inc_cnt = int(df_master["incident_occurred"].sum())
            render_kpi_card("Historical Incident Rate", f"{(inc_cnt/len(df_master))*100:.1f}%", f"{inc_cnt:,} Incident Ground Truth Events", "#f43f5e")

        st.markdown("---")

        # 1. Interactive Factor Dictionary & Understanding Table
        st.markdown("##### 1. Master Factor Taxonomy & Municipal Data Dictionary")
        st.caption("Filter by factor category or search any column to understand its physical meaning, unit, data type, and summary statistics.")

        cat_options = ["ALL CATEGORIES", "Spatial Identification", "Environmental & Climate", "Infrastructure & Pressure", "Traffic Flow & Kinematics", "Vehicle Modal Mix", "Kinematic States", "Temporal / Derived Factor", "Safety Target"]
        fc_col1, fc_col2 = st.columns([1.5, 2])
        with fc_col1:
            selected_cat = st.selectbox("FILTER BY FACTOR CATEGORY", cat_options, index=0)
        with fc_col2:
            search_factor = st.text_input("SEARCH ANY FACTOR OR KEYWORD", placeholder="e.g. weather, motorcycle, speed, slope, pressure...")

        # Build table records
        table_rows = []
        for col_name, meta in factor_dict.items():
            cat = meta.get("category", "General")
            if selected_cat != "ALL CATEGORIES" and selected_cat not in cat:
                continue
            if search_factor and (search_factor.lower() not in col_name.lower() and search_factor.lower() not in meta.get("description", "").lower()):
                continue

            stats = meta.get("statistics", {})
            if "min" in stats:
                stat_str = f"Min: {stats['min']} | Max: {stats['max']} | Mean: {stats['mean']} (Std: {stats['std']})"
            else:
                stat_str = f"Unique: {stats.get('unique_count', 0)} ({', '.join(str(v) for v in stats.get('unique_values', [])[:4])}...)"

            table_rows.append({
                "Factor Column": col_name,
                "Category": cat,
                "Data Type": meta.get("data_type", "Numeric"),
                "Unit": meta.get("unit", "-"),
                "Municipal Significance & Physical Meaning": meta.get("description", ""),
                "Statistical Distribution": stat_str
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=360)

        st.markdown("---")

        # 2. Interactive Distribution & Factor Visualizer
        st.markdown("##### 2. Interactive Factor Distribution & Correlation Visualizer")
        st.caption("Select any variable from the database to examine its distribution histogram, kernel density, and impact on traffic incident risk.")

        num_cols_master = [c for c in df_master.columns if pd.api.types.is_numeric_dtype(df_master[c])]
        v_col1, v_col2 = st.columns([1.5, 2.5])
        with v_col1:
            chosen_viz_factor = st.selectbox("SELECT FACTOR TO VISUALIZE", num_cols_master, index=min(9, len(num_cols_master)-1))
            st.markdown(f"""
            <div style="background: #18181b; border: 1px solid #27272a; border-radius: 4px; padding: 12px; margin-top: 8px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa;">SELECTED FACTOR:</div>
                <div style="font-size: 14px; font-weight: 700; color: #38bdf8; margin: 2px 0;">{chosen_viz_factor}</div>
                <div style="font-size: 12px; color: #d4d4d8;">{factor_dict.get(chosen_viz_factor, {}).get('description', '')}</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #71717a; margin-top: 6px;">
                    Category: {factor_dict.get(chosen_viz_factor, {}).get('category', '')} | Unit: {factor_dict.get(chosen_viz_factor, {}).get('unit', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with v_col2:
            fig_hist = px.histogram(
                df_master,
                x=chosen_viz_factor,
                color="incident_occurred",
                barmode="overlay",
                nbins=35,
                color_discrete_map={0: "#38bdf8", 1: "#f43f5e"},
                labels={"incident_occurred": "Incident Occurred"},
                title=f"Distribution of '{chosen_viz_factor}' (Safe Corridors vs Incident Locations)"
            )
            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="#18181b",
                plot_bgcolor="#18181b",
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # 3. SQLite Master Database Console & Query Runner
        st.markdown("##### 3. SQLite Master Database Console & SQL Query Runner")
        st.caption("Query the SQLite table `master_traffic_records` directly using standard SQL syntax or select a quick query shortcut.")

        query_shortcuts = {
            "Custom Query": "SELECT zone_id, week, city, zone_type, weather, congestion, average_speed, incident_occurred FROM master_traffic_records LIMIT 15;",
            "Top 5 Hazardous Cities by Incidents": "SELECT city, COUNT(DISTINCT zone_id) as zones, SUM(incident_occurred) as total_incidents, ROUND(AVG(congestion), 1) as avg_congestion, ROUND(AVG(average_speed), 1) as avg_speed_kmh FROM master_traffic_records GROUP BY city ORDER BY total_incidents DESC LIMIT 5;",
            "Weather Shock Impact on Incidents": "SELECT weather, COUNT(*) as observations, SUM(incident_occurred) as total_incidents, ROUND(AVG(incident_occurred)*100.0, 1) as incident_rate_pct, ROUND(AVG(average_speed), 1) as avg_speed_kmh FROM master_traffic_records GROUP BY weather ORDER BY incident_rate_pct DESC;",
            "Capacity Pressure > 1.0 Bottlenecks": "SELECT zone_id, week, city, zone_type, traffic_pressure, congestion, incident_occurred FROM master_traffic_records WHERE traffic_pressure > 1.15 ORDER BY traffic_pressure DESC LIMIT 15;",
            "Vehicle Modal Composition (Two-Wheelers vs Incidents)": "SELECT zone_type, ROUND(AVG(car_percentage), 1) as car_pct, ROUND(AVG(motorcycle_percentage), 1) as moto_pct, ROUND(AVG(bus_percentage), 1) as bus_pct, ROUND(AVG(truck_percentage), 1) as truck_pct, SUM(incident_occurred) as incidents FROM master_traffic_records GROUP BY zone_type ORDER BY incidents DESC;"
        }

        q_sc = st.selectbox("PRE-BUILT SQL QUERY SHORTCUTS", list(query_shortcuts.keys()), index=1)
        default_sql = query_shortcuts[q_sc]

        sql_text = st.text_area("SQL QUERY (Table: master_traffic_records)", value=default_sql, height=80)
        
        q_btn_col1, q_btn_col2 = st.columns([1.5, 3])
        with q_btn_col1:
            run_sql = st.button(" EXECUTE SQL QUERY", type="primary", use_container_width=True)
        with q_btn_col2:
            with open("data/unified_traffic_database.csv", "rb") as f:
                csv_bytes_master = f.read()
            st.download_button(
                " DOWNLOAD UNIFIED MASTER DATABASE (unified_traffic_database.csv)",
                data=csv_bytes_master,
                file_name="unified_traffic_database.csv",
                mime="text/csv",
                use_container_width=True
            )

        if run_sql or sql_text:
            try:
                res_df = query_master_database(sql_text)
                st.markdown(f"<div style='font-family: \"JetBrains Mono\", monospace; font-size: 11px; color: #22c55e; margin: 4px 0;'> QUERY RETURNED {len(res_df):,} ROWS</div>", unsafe_allow_html=True)
                st.dataframe(res_df, use_container_width=True, height=260)
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")

    # --------------------------------------------------------------------------
    # TAB 1: BACKEND CSV ORGANIZER & ACCURACY LAB
    # --------------------------------------------------------------------------
    with tab_backend_organizer:
        st.markdown("####  Backend CSV Ingestion, Temporal Week/Day Organizer & Model Accuracy Studio")
        st.caption("Drop any raw CSV into `backend/input_csv/` (or upload below). The backend temporal engine automatically parses timestamps, organizes records into **Weeks (1–52)** and **Days of the Week (Monday–Sunday)**, creates partitions in `backend/organized_data/`, and trains a machine learning model to evaluate prediction accuracy.")

        # Section 1: Ingestion & CSV Discovery
        st.markdown("##### 1. Backend Input Directory & File Selection")
        backend_files = find_input_csv_files()

        b_c1, b_c2 = st.columns([1.5, 1])
        with b_c1:
            if backend_files:
                chosen_backend_csv = st.selectbox(
                    "DETECTED CSV FILES IN `backend/input_csv/`",
                    backend_files,
                    format_func=lambda p: f" {os.path.basename(p)} ({os.path.getsize(p)/1024:.1f} KB)",
                    index=0
                )
            else:
                st.warning("No CSV files found in `backend/input_csv/`. Please place a CSV file in that folder or upload one below.")
                chosen_backend_csv = None

        with b_c2:
            up_file = st.file_uploader(
                "OR DRAG & DROP / UPLOAD TO `backend/input_csv/`",
                type=["csv"],
                key="backend_upload_tab"
            )
            if up_file is not None:
                dest_path = os.path.join(INPUT_CSV_DIR, up_file.name)
                with open(dest_path, "wb") as f:
                    f.write(up_file.getbuffer())
                st.success(f" Saved **{up_file.name}** to `backend/input_csv/`!")
                chosen_backend_csv = dest_path

        st.markdown("---")

        # Section 2: Organize into Weeks and Days of the Week
        st.markdown("##### 2. Temporal Organization into Weeks (1–52) & Day of the Week (Mon–Sun)")
        st.caption("The engine will parse timestamps, calculate calendar week indexes, extract Monday through Sunday, and partition files into `backend/organized_data/`.")

        org_col_btn, _ = st.columns([2, 2])
        with org_col_btn:
            run_organize_btn = st.button(" ORGANIZE DATASET BY WEEKS & DAYS OF WEEK", type="primary", use_container_width=True)

        if run_organize_btn and chosen_backend_csv:
            with st.spinner("Extracting ISO weeks (1-52), day names (Mon-Sun), and creating partitions..."):
                try:
                    df_organized, org_summary = organize_csv_dataset(chosen_backend_csv)
                    st.session_state["backend_org_summary"] = org_summary
                    st.session_state["backend_org_df"] = df_organized
                    st.success(f" Successfully organized {len(df_organized):,} observations into {org_summary['weeks_count']} Weeks and 7 Days of the Week!")
                except Exception as e:
                    st.error(f"Error organizing dataset: {e}")

        # Check if organized summary exists in session or on disk
        summary_on_disk = os.path.join(ORGANIZED_DIR, "organization_summary.json")
        cur_summary = st.session_state.get("backend_org_summary", None)
        if cur_summary is None and os.path.exists(summary_on_disk):
            try:
                with open(summary_on_disk, "r", encoding="utf-8") as f:
                    cur_summary = json.load(f)
                    st.session_state["backend_org_summary"] = cur_summary
            except Exception:
                pass

        if cur_summary:
            # Display Organization KPI Summary
            ok1, ok2, ok3, ok4 = st.columns(4)
            with ok1:
                render_kpi_card("Total Observations", f"{cur_summary['total_records']:,}", f"From {cur_summary['source_file']}", "#38bdf8")
            with ok2:
                render_kpi_card("Weeks Spanned", f"{cur_summary['weeks_count']} Weeks", f"Week {cur_summary['min_week']} to {cur_summary['max_week']}", "#22c55e")
            with ok3:
                render_kpi_card("Day of Week Partitions", "7 CSV Files", "Monday through Sunday", "#f59e0b")
            with ok4:
                render_kpi_card("Incident Rate", f"{cur_summary['overall_incident_rate_pct']:.1f}%", f"{cur_summary['total_incidents']:,} Total Incidents", "#f43f5e")

            # Day of Week & Weekly Distribution Visualizations
            v_day_col, v_week_col = st.columns([1, 1])
            with v_day_col:
                # Prepare Day of Week chart data
                day_breakdown = cur_summary.get("day_of_week_breakdown", {})
                day_plot_rows = []
                for d_name in DAY_NAMES:
                    if d_name in day_breakdown:
                        d_info = day_breakdown[d_name]
                        day_plot_rows.append({
                            "Day of Week": d_name,
                            "Total Records": d_info["records"],
                            "Incidents": d_info["incidents"],
                            "Incident Rate (%)": d_info["incident_rate"]
                        })
                df_day_plot = pd.DataFrame(day_plot_rows)

                fig_day = px.bar(
                    df_day_plot,
                    x="Day of Week",
                    y="Incidents",
                    color="Incident Rate (%)",
                    text="Incidents",
                    color_continuous_scale="Reds",
                    title="Traffic Incidents by Day of the Week (Surge vs. Calm Days)"
                )
                fig_day.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#18181b",
                    plot_bgcolor="#18181b",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                fig_day.update_traces(textposition='outside')
                st.plotly_chart(fig_day, use_container_width=True)

            with v_week_col:
                # Weekly volume chart
                if os.path.exists(cur_summary.get("master_csv_path", "")):
                    master_preview = pd.read_csv(cur_summary["master_csv_path"])
                    weekly_agg = master_preview.groupby("week").agg(
                        total_records=("incident_occurred", "count"),
                        incidents=("incident_occurred", "sum")
                    ).reset_index()

                    fig_week = go.Figure()
                    fig_week.add_trace(go.Scatter(
                        x=weekly_agg["week"],
                        y=weekly_agg["total_records"],
                        mode="lines",
                        name="Total Traffic Records",
                        line=dict(color="#38bdf8", width=2)
                    ))
                    fig_week.add_trace(go.Bar(
                        x=weekly_agg["week"],
                        y=weekly_agg["incidents"],
                        name="Incident Hazards",
                        marker_color="#f43f5e",
                        opacity=0.7
                    ))
                    fig_week.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#18181b",
                        plot_bgcolor="#18181b",
                        title=f"Weekly Volume & Incident Trajectory across {cur_summary['weeks_count']} Weeks",
                        height=280,
                        margin=dict(l=20, r=20, t=40, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_week, use_container_width=True)

            with st.expander(" PREVIEW CONSOLIDATED MASTER ORGANIZED PANEL"):
                if os.path.exists(cur_summary.get("master_csv_path", "")):
                    preview_df = pd.read_csv(cur_summary["master_csv_path"])
                    st.dataframe(preview_df.head(50), use_container_width=True, height=250)

            st.markdown("---")

            # Section 3: Model Training & Accuracy Evaluation
            st.markdown("##### 3. Supervised Model Training & Accuracy Evaluation")
            st.caption("Trains a supervised machine learning risk forecaster on the organized dataset and computes overall accuracy, confusion matrix, and day-of-the-week performance.")

            t_col1, t_col2 = st.columns([1, 1.5])
            with t_col1:
                custom_threshold_backend = st.slider(
                    "RISK DECISION THRESHOLD (P >= threshold -> Warning)",
                    0.20, 0.80, 0.50, 0.05,
                    key="backend_thresh_slider"
                )
            with t_col2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                run_train_btn = st.button(" TRAIN MODEL & EVALUATE ACCURACY", type="primary", use_container_width=True)

            if run_train_btn or "backend_accuracy_report" in st.session_state:
                if run_train_btn:
                    with st.spinner("Splitting chronologically, training Logistic Regression & Random Forest, and evaluating accuracy..."):
                        try:
                            acc_report = train_and_evaluate_model(
                                organized_csv_path=cur_summary.get("master_csv_path", None),
                                threshold=custom_threshold_backend
                            )
                            st.session_state["backend_accuracy_report"] = acc_report
                            st.success(" Model Training & Accuracy Benchmark Completed!")
                        except Exception as e:
                            st.error(f"Error training model: {e}")

                rep = st.session_state.get("backend_accuracy_report", None)
                if rep:
                    best_m = rep["best_model_metrics"]
                    cm = best_m["confusion_matrix"]

                    # Metric Tiles
                    ak1, ak2, ak3, ak4, ak5 = st.columns(5)
                    with ak1:
                        render_kpi_card("Overall Accuracy", f"{best_m['accuracy_pct']:.1f}%", f"{cm['total_tested']:,} Test Samples", "#22c55e")
                    with ak2:
                        render_kpi_card("Recall (Sensitivity)", f"{best_m['recall_pct']:.1f}%", "Hazards Detected", "#38bdf8")
                    with ak3:
                        render_kpi_card("Precision", f"{best_m['precision_pct']:.1f}%", "True Alarm Rate", "#f59e0b")
                    with ak4:
                        render_kpi_card("F1-Score", f"{best_m['f1_score']:.4f}", "Harmonic Mean", "#a855f7")
                    with ak5:
                        render_kpi_card("ROC-AUC", f"{best_m['roc_auc']:.4f}", "Discriminative Power", "#f43f5e")

                    # Confusion Matrix & Day of Week Comparison
                    cm_col, day_acc_col = st.columns([1, 1.4])

                    with cm_col:
                        st.markdown("###### Confusion Matrix (Test Set)")
                        cm_data = [
                            ["Actual Safe (0)", f"TN: {cm['true_negatives']:,}", f"FP: {cm['false_positives']:,} (False Alarm)"],
                            ["Actual Hazard (1)", f"FN: {cm['false_negatives']:,} (Missed)", f"TP: {cm['true_positives']:,} (Captured)"]
                        ]
                        cm_df = pd.DataFrame(cm_data, columns=["Ground Truth", "Predicted Safe (0)", "Predicted Hazard (1)"])
                        st.dataframe(cm_df, use_container_width=True, hide_index=True)

                        st.caption(f"Champion Model: **{rep['selected_model']}** | Brier Calibration Loss: **{best_m['brier_score']:.4f}**")

                    with day_acc_col:
                        st.markdown("###### Accuracy Comparison by Day of the Week")
                        day_acc_rows = []
                        for d_name, d_met in rep.get("day_of_week_accuracy", {}).items():
                            day_acc_rows.append({
                                "Day": d_name,
                                "Accuracy (%)": d_met["accuracy_pct"],
                                "Recall (%)": d_met["recall_pct"],
                                "Avg Predicted Risk": d_met["avg_predicted_risk"]
                            })
                        df_day_acc = pd.DataFrame(day_acc_rows)

                        fig_day_acc = px.bar(
                            df_day_acc,
                            x="Day",
                            y=["Accuracy (%)", "Recall (%)"],
                            barmode="group",
                            color_discrete_map={"Accuracy (%)": "#22c55e", "Recall (%)": "#38bdf8"},
                            title="Day-of-Week Model Accuracy vs. Hazard Recall"
                        )
                        fig_day_acc.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#18181b",
                            plot_bgcolor="#18181b",
                            height=250,
                            margin=dict(l=20, r=20, t=30, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_day_acc, use_container_width=True)

                    # Top Feature Importances
                    if rep.get("top_feature_importances"):
                        st.markdown("###### Top Feature Contributions to Traffic Risk")
                        feat_df = pd.DataFrame(rep["top_feature_importances"][:10])
                        fig_feat = px.bar(
                            feat_df,
                            x="importance",
                            y="feature",
                            orientation="h",
                            color="importance",
                            color_continuous_scale="Blues",
                            title="Top 10 Feature Importances"
                        )
                        fig_feat.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#18181b",
                            plot_bgcolor="#18181b",
                            yaxis=dict(autorange="reversed"),
                            height=240,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )
                        st.plotly_chart(fig_feat, use_container_width=True)

                    # Test Predictions Table
                    with st.expander(" INSPECT TEST PREDICTIONS VS. GROUND TRUTH"):
                        if os.path.exists(rep.get("predictions_file", "")):
                            pred_table = pd.read_csv(rep["predictions_file"])
                            st.dataframe(pred_table.head(50), use_container_width=True, height=260)

    # --------------------------------------------------------------------------
    # TAB 2: INTERACTIVE 4-WEEK -> 5TH-WEEK STEP INSPECTOR
    # --------------------------------------------------------------------------
    with tab_step_inspector:
        st.markdown("####  Sequential 4-Week Feeder $\\rightarrow$ 5th-Week Risk Forecast & Reinforcement Meter")
        st.caption("Select any municipal zone and 4-week sequence. The model ingests the 4-week historical momentum and 5th-week circumstances, predicts the 5th-week incident risk, compares with ground truth, and computes positive/negative reinforcement feedback.")

        # Control Bar
        c1, c2, c3 = st.columns([2, 2, 1.5])
        
        with c1:
            zone_list = sorted(df_full["zone_id"].unique())
            # Format with readable name if available
            def format_zone_label(z_id):
                row = df_full[df_full["zone_id"] == z_id].iloc[0]
                loc_name = row.get("location_name", z_id)
                city = row.get("city", "Metropolis")
                z_type = row.get("zone_type", "Urban")
                return f" {z_id}: {loc_name} ({city} — {z_type})"

            selected_zone = st.selectbox("SELECT MUNICIPAL ZONE", zone_list, format_func=format_zone_label, index=0)

        with c2:
            max_week = int(df_full["week"].max())
            selected_target_week = st.slider(
                "TARGET 5TH WEEK TO PREDICT (Feeds Prior 4 Weeks)",
                min_value=5,
                max_value=max_week,
                value=5,
                help="Week t (5th week) is predicted using the 4 historical weeks [t-4, t-3, t-2, t-1] plus week t's environmental circumstances."
            )

        with c3:
            threshold = st.slider("DECISION THRESHOLD", min_value=0.20, max_value=0.80, value=0.50, step=0.05, help="Probability cutoff for classifying as High Risk / Incident Warning.")

        st.markdown("---")

        # Run Single Step Evaluation
        step_result = evaluate_single_5th_week(model, df_full, selected_zone, selected_target_week, threshold=threshold)

        if step_result is None:
            st.warning("Insufficient historical data for the selected week. Minimum 4 preceding weeks required.")
        else:
            circ = step_result["circumstances"]
            reinf = step_result["reinforcement"]
            truth = step_result["actual_ground_truth"]
            feeder_df = step_result["feeder_df"]
            feeder_weeks = step_result["feeder_weeks"]
            prob = step_result["predicted_risk_probability"]

            # 1. Row of 4-Week Feeder Summary & 5th-Week Circumstances
            st.markdown(f"##### 1. Input Context: 4-Week Historical Window (Weeks {min(feeder_weeks)} to {max(feeder_weeks)}) $\\rightarrow$ Target Week {selected_target_week}")
            
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                render_kpi_card("4-Week Avg Congestion", f"{circ['rolling_4w_avg_congestion']:.1f} / 100", f"Slope: {circ['congestion_trend_4w']:+.2f}/wk", "#38bdf8")
            with f_col2:
                render_kpi_card("4-Week Avg Vehicle Density", f"{circ['rolling_4w_avg_vehicle_density']:.0f} veh/km²", f"Slope: {circ['vehicle_density_trend_4w']:+.1f}/wk", "#a78bfa")
            with f_col3:
                render_kpi_card("4-Week Avg Speed", f"{circ['rolling_4w_avg_speed']:.1f} km/h", f"Slope: {circ['speed_trend_4w']:+.2f}/wk", "#34d399")
            with f_col4:
                render_kpi_card("4-Week Total Incidents", f"{circ['rolling_4w_incident_count']} incidents", f"Rate: {circ['rolling_4w_incident_count']/4:.2f}/wk", "#fb7185")

            # 4-Week Feeder Timeline Chart + 5th-Week Reveal
            plot_df = pd.concat([feeder_df, step_result["target_row_df"]]).copy()
            plot_df["timeline_label"] = plot_df["week"].apply(lambda w: f"Week {w} (Feeder)" if w < selected_target_week else f"Week {w} (TARGET)")
            
            fig_trend = go.Figure()
            # Congestion trace
            fig_trend.add_trace(go.Scatter(
                x=plot_df["week"],
                y=plot_df["congestion"],
                mode="lines+markers+text",
                name="Congestion Index",
                line=dict(color="#38bdf8", width=3),
                marker=dict(size=8, color=["#38bdf8"]*4 + ["#f43f5e"]),
                text=[f"W{w}: {c:.0f}" for w, c in zip(plot_df["week"], plot_df["congestion"])],
                textposition="top center"
            ))
            # Speed trace
            fig_trend.add_trace(go.Scatter(
                x=plot_df["week"],
                y=plot_df["average_speed"],
                mode="lines+markers",
                name="Average Speed (km/h)",
                line=dict(color="#34d399", width=2, dash="dot"),
                marker=dict(size=6)
            ))
            # Vertical separator line
            fig_trend.add_vline(
                x=selected_target_week - 0.5,
                line_width=1.5,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"5th-Week Prediction Boundary (Week {selected_target_week})",
                annotation_position="top left",
                annotation_font=dict(color="#ef4444", size=11, family="JetBrains Mono")
            )

            fig_trend.update_layout(
                title=f"4-Week Multi-Signal Trajectory $\\rightarrow$ Target Week {selected_target_week} [{selected_zone}]",
                template="plotly_dark",
                paper_bgcolor="#121215",
                plot_bgcolor="#18181b",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Calendar Week", dtick=1, gridcolor="#27272a"),
                yaxis=dict(title="Index / km/h", gridcolor="#27272a")
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # 5th-Week Environmental Circumstances Bar
            st.markdown(f"""
            <div style="background: #18181b; border: 1px solid #27272a; border-radius: 4px; padding: 12px 18px; margin-bottom: 16px; display: flex; flex-wrap: wrap; gap: 20px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #a1a1aa;">
                <div><span style="color: #71717a;">WEATHER:</span> <span style="color: #fafafa; font-weight: 600;">{circ['weather']}</span></div>
                <div><span style="color: #71717a;">ROAD CONDITION:</span> <span style="color: #fafafa; font-weight: 600;">{circ['road_condition']}</span></div>
                <div><span style="color: #71717a;">SPECIAL EVENT:</span> <span style="color: {'#f59e0b' if circ['special_event'] else '#10b981'}; font-weight: 600;">{'YES (ACTIVE EVENT)' if circ['special_event'] else 'NO'}</span></div>
                <div><span style="color: #71717a;">TRAFFIC PRESSURE:</span> <span style="color: #fafafa; font-weight: 600;">{circ['traffic_pressure']:.2f}</span></div>
                <div><span style="color: #71717a;">VEH/POP RATIO:</span> <span style="color: #fafafa; font-weight: 600;">{circ['vehicle_population_ratio']:.2f}</span></div>
            </div>
            """, unsafe_allow_html=True)

            # 2. Side-by-Side ML Prediction vs Actual Ground Truth Scorecard
            st.markdown("##### 2. ML Prediction vs Actual Ground Truth Scorecard")
            
            score_col1, score_col2 = st.columns([1, 1])

            with score_col1:
                risk_tier = "CRITICAL HAZARD" if prob >= 0.75 else ("HIGH RISK" if prob >= 0.55 else ("MODERATE" if prob >= 0.35 else "LOW / SAFE"))
                tier_color = "#ef4444" if prob >= 0.55 else ("#f59e0b" if prob >= 0.35 else "#10b981")
                
                st.markdown(f"""
                <div style="background: #18181b; border: 1px solid #3f3f46; border-radius: 4px; padding: 18px 20px; height: 100%;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">
                         MACHINE LEARNING 5TH-WEEK FORECAST
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700; color: {tier_color}; margin: 8px 0;">
                        {prob*100:.1f}%
                    </div>
                    <div style="display: flex; gap: 8px; margin-bottom: 12px;">
                        <span style="background: #27272a; color: {tier_color}; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 4px 10px; border-radius: 2px; font-weight: 600;">
                            {risk_tier}
                        </span>
                        <span style="background: #27272a; color: #fafafa; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 4px 10px; border-radius: 2px;">
                            {'CLASS 1 (INCIDENT WARNING)' if reinf['predicted_class'] == 1 else 'CLASS 0 (SAFE / NORMAL)'}
                        </span>
                    </div>
                    <div style="font-size: 12px; color: #a1a1aa; line-height: 1.4;">
                        Model evaluated 4-week historical momentum slope and modulated by current weather ({circ['weather']}) and traffic pressure ({circ['traffic_pressure']:.2f}).
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with score_col2:
                actual_occ = truth["actual_incident_occurred"]
                actual_text = "INCIDENT OCCURRED (HAZARD ACTIVE)" if actual_occ == 1 else "NO INCIDENT (SAFE FLOW)"
                actual_color = "#ef4444" if actual_occ == 1 else "#10b981"
                
                st.markdown(f"""
                <div style="background: #18181b; border: 1px solid #3f3f46; border-radius: 4px; padding: 18px 20px; height: 100%;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">
                         ACTUAL 5TH-WEEK GROUND TRUTH REVEAL
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700; color: {actual_color}; margin: 8px 0;">
                        {truth['actual_incident_count']} Incidents
                    </div>
                    <div style="display: flex; gap: 8px; margin-bottom: 12px;">
                        <span style="background: #27272a; color: {actual_color}; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 4px 10px; border-radius: 2px; font-weight: 600;">
                            {actual_text}
                        </span>
                        <span style="background: #27272a; color: #fafafa; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 4px 10px; border-radius: 2px;">
                            Congestion: {truth['actual_congestion']:.0f} | Speed: {truth['actual_speed']:.1f} km/h
                        </span>
                    </div>
                    <div style="font-size: 12px; color: #a1a1aa; line-height: 1.4;">
                        Real municipal ground truth recorded for Week {selected_target_week}. Red-light violations observed: {truth['actual_violations']:.0f}.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 3. Positive vs Negative Reinforcement Feedback Meter
            st.markdown("##### 3. Positive & Negative Reinforcement Signal Meter")
            
            reinf_score = reinf["total_reinforcement_score"]
            is_pos = (reinf["reinforcement_polarity"] == "POSITIVE_REINFORCEMENT")
            meter_border = "#22c55e" if is_pos else "#ef4444"
            meter_bg = "#052e16" if is_pos else "#450a0a"

            st.markdown(f"""
            <div style="background: {meter_bg}; border: 1px solid {meter_border}; border-radius: 4px; padding: 20px 24px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: {'#86efac' if is_pos else '#fca5a5'}; letter-spacing: 0.08em; text-transform: uppercase;">
                            REINFORCEMENT SIGNAL: {reinf['reinforcement_polarity']}
                        </div>
                        <div style="font-size: 18px; font-weight: 700; color: #fafafa; margin-top: 4px;">
                            {reinf['badge_text']}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 700; color: {'#4ade80' if is_pos else '#f87171'};">
                            {reinf_score:+.2f} pts
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa;">
                            Net Step Reward
                        </div>
                    </div>
                </div>
                <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 14px 0;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-family: 'JetBrains Mono', monospace; font-size: 12px;">
                    <div>
                        <span style="color: #a1a1aa;">Action Payoff:</span> 
                        <span style="color: {'#4ade80' if reinf['base_action_reward'] > 0 else '#f87171'}; font-weight: 600;">{reinf['base_action_reward']:+.1f} pts</span>
                    </div>
                    <div>
                        <span style="color: #a1a1aa;">Probability Calibration:</span> 
                        <span style="color: {'#4ade80' if reinf['calibration_reward'] > 0 else '#f87171'}; font-weight: 600;">{reinf['calibration_reward']:+.2f} pts</span>
                    </div>
                    <div>
                        <span style="color: #a1a1aa;">Brier Error Loss:</span> 
                        <span style="color: #fafafa; font-weight: 600;">{reinf['brier_error']:.4f}</span>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 13px; color: #e4e4e7; line-height: 1.5; font-family: 'Inter', sans-serif;">
                    <strong>Civic Feedback:</strong> {reinf['diagnostic_explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 2: MULTI-MONTH WALK-FORWARD BENCHMARK
    # --------------------------------------------------------------------------
    with tab_walk_forward:
        st.markdown("####  Multi-Month Walk-Forward Reinforcement Benchmark (Weeks 5 to 52 across 50 Zones)")
        st.caption("Evaluates all 2,400 consecutive 5th-week prediction steps chronologically. Tracks cumulative reinforcement rewards, rolling accuracy, and monthly continual learning curves.")

        sim_results_df, summary = get_walk_forward_simulation_results(model, df_full)

        # Global Top KPIs
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            render_kpi_card("Total 5th-Week Steps", f"{summary['total_evaluated_instances']:,}", "Weeks 5 to 52 (50 zones)", "#fafafa")
        with kpi2:
            render_kpi_card("Overall Accuracy", f"{summary['overall_accuracy']:.1f}%", f"F1-Score: {summary['f1_score']:.1f}%", "#22c55e")
        with kpi3:
            render_kpi_card("Precision / Recall", f"{summary['precision']:.1f}% / {summary['recall']:.1f}%", "Incident Foresight", "#38bdf8")
        with kpi4:
            render_kpi_card("Cumulative Reward", f"{summary['total_cumulative_reward']:+,.0f} pts", f"Avg: {summary['average_reward_per_instance']:+.2f} pts/step", "#a78bfa")
        with kpi5:
            pos_pct = summary["reinforcement_distribution"]["positive_ratio_pct"]
            render_kpi_card("Positive Feedback", f"{pos_pct:.1f}%", f"{summary['reinforcement_distribution']['positive_reinforcements']} / {summary['total_evaluated_instances']}", "#34d399")

        st.markdown("---")

        # Visualizations
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            # Cumulative Reward Curve
            fig_rew = px.line(
                sim_results_df,
                x="step_index",
                y="cumulative_reward",
                title="Cumulative Reinforcement Reward Trajectory (\\sum R_t)",
                labels={"step_index": "Sequential 5th-Week Decision Step", "cumulative_reward": "Cumulative Reward (pts)"}
            )
            fig_rew.update_traces(line=dict(color="#10b981", width=2.5))
            fig_rew.update_layout(
                template="plotly_dark",
                paper_bgcolor="#121215",
                plot_bgcolor="#18181b",
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(gridcolor="#27272a"),
                yaxis=dict(gridcolor="#27272a")
            )
            st.plotly_chart(fig_rew, use_container_width=True)

        with chart_col2:
            # Rolling Accuracy Curve
            fig_acc = px.line(
                sim_results_df,
                x="step_index",
                y="rolling_accuracy",
                title="Rolling Prediction Accuracy Trajectory (100-Step Window)",
                labels={"step_index": "Sequential 5th-Week Decision Step", "rolling_accuracy": "Rolling Accuracy"}
            )
            fig_acc.update_traces(line=dict(color="#38bdf8", width=2.5))
            fig_acc.add_hline(y=0.80, line_dash="dash", line_color="#a1a1aa", annotation_text="80% Target Benchmark", annotation_position="bottom right")
            fig_acc.update_layout(
                template="plotly_dark",
                paper_bgcolor="#121215",
                plot_bgcolor="#18181b",
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(gridcolor="#27272a"),
                yaxis=dict(gridcolor="#27272a", tickformat=".0%")
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        # Monthly Continual Learning Progress Table
        st.markdown("#####  Monthly Reinforcement & Accuracy Progress")
        monthly_df = pd.DataFrame(summary["monthly_learning_trajectory"])
        st.dataframe(
            monthly_df.rename(columns={
                "month": "Calendar Window",
                "observations": "Evaluated 5th Weeks",
                "accuracy": "Accuracy (%)",
                "total_reward": "Monthly Reward (pts)",
                "hazards_captured": "Hazards Captured",
                "hazards_missed": "Hazards Missed"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Filterable Logs
        with st.expander(" BROWSE COMPLETE 2,400-STEP REINFORCEMENT LOG"):
            st.dataframe(
                sim_results_df[[
                    "week", "zone_id", "feeder_window", "actual_incident", "predicted_prob",
                    "outcome_type", "reinforcement_polarity", "step_reward", "cumulative_reward", "diagnostic"
                ]],
                use_container_width=True,
                height=300
            )

    # --------------------------------------------------------------------------
    # TAB 3: CUSTOM DATABASE EVALUATOR (SMART FACTOR MAPPER & ON-THE-FLY ML)
    # --------------------------------------------------------------------------
    with tab_custom_db:
        st.markdown("####  Universal Multi-Month Database Ingestion & Adaptive ML Evaluator")
        st.caption("Upload your own database with **any custom factors or columns** (such as the 18-column police crash dataset). The smart engine will auto-detect columns, synthesize 4-week momentum features, train a dynamic ML model on-the-fly, and run walk-forward positive/negative reinforcement evaluation.")

        db_source_option = st.radio(
            "CHOOSE DATASET SOURCE TO TRAIN & EVALUATE ON:",
            [
                " Upload Your Own Custom CSV File",
                " Pre-Loaded 18-Factor Police Crash Dataset (BEAT_OF_OCCURRENCE, POSTED_SPEED_LIMIT, WEATHER, LIGHTING, INJURIES)",
                " Pre-Loaded 56-Factor Master Traffic Simulation Database"
            ],
            horizontal=True
        )

        user_df = None
        if "Upload Your Own" in db_source_option:
            uploaded_file = st.file_uploader(
                "UPLOAD ANY CSV TRAFFIC / CRASH / SAFETY DATABASE",
                type=["csv"],
                help="Upload any database file with custom columns (e.g. CRASH_DATE, BEAT_OF_OCCURRENCE, POSTED_SPEED_LIMIT, WEATHER_CONDITION, etc.)"
            )
            if uploaded_file is not None:
                try:
                    raw_uploaded = pd.read_csv(uploaded_file)
                    # Check if it is a raw incident crash log with dates
                    if any(c in raw_uploaded.columns for c in ["CRASH_DATE", "RASH_DATE", "crash_date", "Date", "date"]) and any(c in raw_uploaded.columns for c in ["BEAT_OF_OCCURRENCE", "beat", "zone_id", "Location"]):
                        with st.spinner("Detected Raw Incident-Level Log. Aggregating into 4-Week Sequential Panel..."):
                            user_df = transform_crash_records_to_4week_panel(raw_uploaded)
                            st.success(f" Successfully converted raw incident log into **4-Week Sequential Panel** ({len(user_df):,} rows × {len(user_df.columns)} columns)!")
                    else:
                        user_df = raw_uploaded
                        st.success(f" Successfully loaded: **{uploaded_file.name}** ({len(user_df):,} rows × {len(user_df.columns)} columns)")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
        elif "18-Factor Police Crash" in db_source_option:
            raw_crash_path = "data/chicago_police_crashes_raw.csv"
            if not os.path.exists(raw_crash_path):
                from data.crash_dataset_adapter import generate_benchmark_police_crash_dataset
                generate_benchmark_police_crash_dataset().to_csv(raw_crash_path, index=False)
            
            raw_crash_df = pd.read_csv(raw_crash_path)
            user_df = transform_crash_records_to_4week_panel(raw_crash_df)
            st.info(f" Loaded **18-Factor Police Crash Dataset** transformed into 4-Week Rolling Panel ({len(user_df):,} beat-weeks × {len(user_df.columns)} columns across 20 Police Beats).")
        else:
            user_df = df_full.copy()
            st.info(f" Loaded **56-Factor Master Traffic Simulation Dataset** ({len(user_df):,} rows × {len(user_df.columns)} columns).")

        if user_df is not None:
            # 1. Inspect & Auto-Detect Columns
            detected = auto_detect_dataset_columns(user_df)

            st.markdown("#####  Smart Column & Factor Mapping")
            st.markdown("""
            <div style="background: #18181b; border: 1px solid #27272a; border-radius: 4px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; color: #a1a1aa;">
                The system has automatically analyzed your dataset schema. Verify or customize the column mappings below before training.
            </div>
            """, unsafe_allow_html=True)

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                all_cols = detected["all_columns"]
                default_time_idx = all_cols.index(detected["time_col"]) if detected["time_col"] in all_cols else 0
                custom_time_col = st.selectbox(
                    " TIME / SEQUENCE COLUMN",
                    all_cols,
                    index=default_time_idx,
                    help="Column indicating calendar week, month, date, or step (e.g. week, month, timestamp)."
                )

            with m_col2:
                entity_options = ["(None / Single Unified Zone)"] + all_cols
                default_ent_idx = (all_cols.index(detected["entity_col"]) + 1) if detected["entity_col"] in all_cols else 0
                selected_ent_label = st.selectbox(
                    " ZONE / ENTITY COLUMN",
                    entity_options,
                    index=default_ent_idx,
                    help="Column indicating distinct corridors or zones (e.g. zone_id, location, camera_id)."
                )
                custom_entity_col = None if selected_ent_label.startswith("(") else selected_ent_label

            with m_col3:
                default_target_idx = all_cols.index(detected["target_col"]) if detected["target_col"] in all_cols else (len(all_cols) - 1)
                custom_target_col = st.selectbox(
                    " TARGET OUTCOME TO PREDICT",
                    all_cols,
                    index=default_target_idx,
                    help="Column representing the target incident, accident, or hazard to forecast."
                )

            # Available Factors Selection
            st.markdown("#####  Select Features & Circumstance Factors")
            excluded_set = {custom_time_col, custom_entity_col, custom_target_col}
            available_factors = [c for c in all_cols if c not in excluded_set]

            selected_factors = st.multiselect(
                "CHOOSE ENVIRONMENTAL, TRAFFIC & ROAD FACTORS TO FEED INTO ML",
                available_factors,
                default=available_factors,
                help="Select all numeric and categorical factors to use for 4-week momentum and 5th-week circumstances."
            )

            thresh_col, btn_col = st.columns([1, 1])
            with thresh_col:
                custom_threshold = st.slider("DECISION THRESHOLD (P >= cutoff -> Class 1)", 0.20, 0.80, 0.50, 0.05, key="custom_thresh_slider")
            with btn_col:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                run_custom_btn = st.button(" TRAIN ADAPTIVE ML & RUN 4-WEEK REINFORCEMENT BENCHMARK", type="primary", use_container_width=True)

            # Session State for Custom Results
            if "custom_reinforcement_output" not in st.session_state or run_custom_btn:
                if run_custom_btn:
                    if not selected_factors:
                        st.error("Please select at least one feature factor.")
                    else:
                        with st.spinner("Synthesizing 4-week temporal momentum, fitting dynamic ML pipeline, and evaluating walk-forward reinforcement..."):
                            try:
                                custom_out = run_custom_dataset_reinforcement_pipeline(
                                    df_raw=user_df,
                                    time_col=custom_time_col,
                                    entity_col=custom_entity_col,
                                    target_col=custom_target_col,
                                    selected_features=selected_factors,
                                    threshold=custom_threshold
                                )
                                st.session_state["custom_reinforcement_output"] = custom_out
                                st.success("Dynamic ML Training & Walk-Forward Reinforcement Evaluation Completed!")
                            except Exception as e:
                                st.error(f"Error during execution: {e}")

            if "custom_reinforcement_output" in st.session_state and st.session_state["custom_reinforcement_output"]:
                res_dict = st.session_state["custom_reinforcement_output"]
                c_results_df = res_dict["results_df"]
                c_summary = res_dict["summary"]
                c_imp_df = res_dict["feature_importance_df"]

                st.markdown("---")
                st.markdown("####  Custom Dataset Reinforcement Scorecard")

                # KPI Cards
                ck1, ck2, ck3, ck4, ck5 = st.columns(5)
                with ck1:
                    render_kpi_card("Evaluated 5th Periods", f"{c_summary['total_evaluated_instances']:,}", "Sequential Steps", "#fafafa")
                with ck2:
                    render_kpi_card("Overall Accuracy", f"{c_summary['overall_accuracy']:.1f}%", f"F1: {c_summary['f1_score']:.1f}%", "#22c55e")
                with ck3:
                    render_kpi_card("Precision / Recall", f"{c_summary['precision']:.1f}% / {c_summary['recall']:.1f}%", "Hazard Foresight", "#38bdf8")
                with ck4:
                    render_kpi_card("Cumulative Reward", f"{c_summary['total_cumulative_reward']:+,.0f} pts", f"Avg: {c_summary['average_reward_per_instance']:+.2f} pts", "#a78bfa")
                with ck5:
                    c_pos_pct = c_summary["reinforcement_distribution"]["positive_ratio_pct"]
                    render_kpi_card("Positive Feedback", f"{c_pos_pct:.1f}%", f"{c_summary['reinforcement_distribution']['positive_reinforcements']} / {c_summary['total_evaluated_instances']}", "#34d399")

                # Charts
                g_col1, g_col2 = st.columns([1, 1])
                with g_col1:
                    # Cumulative Reward Curve
                    fig_c_rew = px.line(
                        c_results_df,
                        x="step_index",
                        y="cumulative_reward",
                        title="Cumulative Reinforcement Reward Trajectory (\\sum R_t)",
                        labels={"step_index": "Decision Step", "cumulative_reward": "Reward Points"}
                    )
                    fig_c_rew.update_traces(line=dict(color="#10b981", width=2.5))
                    fig_c_rew.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#121215",
                        plot_bgcolor="#18181b",
                        height=320,
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis=dict(gridcolor="#27272a"),
                        yaxis=dict(gridcolor="#27272a")
                    )
                    st.plotly_chart(fig_c_rew, use_container_width=True)

                with g_col2:
                    # Feature Importance of Custom Factors
                    if not c_imp_df.empty:
                        top_imp = c_imp_df.head(10).sort_values("Importance", ascending=True)
                        fig_imp = px.bar(
                            top_imp,
                            x="Importance",
                            y="Factor",
                            orientation="h",
                            title="Top Custom Factors Influencing 5th-Period Predictions",
                            color="Importance",
                            color_continuous_scale="Viridis"
                        )
                        fig_imp.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#121215",
                            plot_bgcolor="#18181b",
                            height=320,
                            margin=dict(l=20, r=20, t=40, b=20),
                            xaxis=dict(gridcolor="#27272a"),
                            yaxis=dict(gridcolor="#27272a"),
                            coloraxis_showscale=False
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

                # Custom Data Step Inspector
                st.markdown("#####  Inspect Individual 5th-Period Predictions on Your Dataset")
                insp_col1, insp_col2 = st.columns([1, 1])
                with insp_col1:
                    avail_entities = sorted(c_results_df["entity"].unique())
                    insp_entity = st.selectbox("Select Entity / Zone to Inspect", avail_entities)
                with insp_col2:
                    ent_steps = c_results_df[c_results_df["entity"] == insp_entity]
                    avail_times = sorted(ent_steps["time_period"].unique())
                    insp_time = st.selectbox("Select Target Time Period", avail_times)

                chosen_row = ent_steps[ent_steps["time_period"] == insp_time].iloc[0]
                
                # Render Inspector Card
                c_is_pos = (chosen_row["reinforcement_polarity"] == "POSITIVE_REINFORCEMENT")
                c_badge_bg = "#052e16" if c_is_pos else "#450a0a"
                c_badge_border = "#22c55e" if c_is_pos else "#ef4444"

                st.markdown(f"""
                <div style="background: {c_badge_bg}; border: 1px solid {c_badge_border}; border-radius: 4px; padding: 16px 20px; margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: {'#86efac' if c_is_pos else '#fca5a5'}; text-transform: uppercase;">
                                {chosen_row['reinforcement_polarity']}
                            </div>
                            <div style="font-size: 16px; font-weight: 700; color: #fafafa; margin-top: 2px;">
                                {chosen_row['outcome_type']}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; color: {'#4ade80' if c_is_pos else '#f87171'};">
                                {chosen_row['step_reward']:+.2f} pts
                            </div>
                        </div>
                    </div>
                    <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 10px 0;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #e4e4e7; display: flex; gap: 24px; flex-wrap: wrap;">
                        <div>Predicted Prob: <strong>{chosen_row['predicted_prob']*100:.1f}%</strong></div>
                        <div>Actual Target: <strong>{chosen_row['actual_target']}</strong></div>
                        <div>Action Payoff: <strong>{chosen_row['base_action_reward']:+.1f} pts</strong></div>
                        <div>Calibration Payoff: <strong>{chosen_row['calibration_reward']:+.2f} pts</strong></div>
                    </div>
                    <div style="font-size: 12px; color: #a1a1aa; margin-top: 8px;">
                        {chosen_row['diagnostic']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Download Button
                csv_bytes = c_results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    " DOWNLOAD COMPLETE CUSTOM PREDICTION & REINFORCEMENT CSV",
                    data=csv_bytes,
                    file_name="custom_reinforcement_evaluation.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("Upload your custom CSV file above or check the demo dataset box to start mapping factors.")

    # --------------------------------------------------------------------------
    # TAB 4: SELF-CORRECTING TRAINING LAB (MISTAKE DIAGNOSIS & REMEDIATION)
    # --------------------------------------------------------------------------
    with tab_self_correct:
        st.markdown("####  Self-Correcting Mistake-Driven Training Loop (Continual Learning)")
        st.caption("The Machine Learning model rolls across multi-month historical sequences. After predicting each 5th-week instance, the system inspects **where and why it made mistakes**, applies focal error-weighted experience replay, and continually fine-tunes its decision policy to prevent past failures.")

        # Check if persistent memory state exists on disk
        memory_state_path = "models/training_memory_state.json"
        saved_memory_info = None
        if os.path.exists(memory_state_path):
            try:
                with open(memory_state_path, "r", encoding="utf-8") as f:
                    saved_memory_info = json.load(f)
            except Exception:
                saved_memory_info = None

        if saved_memory_info:
            st.markdown(f"""
            <div style="background: #052e16; border: 1px solid #22c55e; border-radius: 4px; padding: 14px 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #86efac; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">
                         PERSISTENT LONG-TERM DISK MEMORY: ACTIVE & SYNCHRONIZED
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa;">
                        LAST TRAINED: {saved_memory_info.get('last_trained_timestamp', 'Recent')}
                    </div>
                </div>
                <div style="font-size: 13px; color: #fafafa; margin-top: 6px;">
                    All learned decision trees, mistake remediation weights, and error memories are saved to binary disk files:
                    <code style="color: #4ade80;">models/self_correcting_risk_model.pkl</code> and <code style="color: #4ade80;">models/best_risk_model.pkl</code>.
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 12px; color: #a1a1aa; margin-top: 4px;">
                     <strong>Persistence Guarantee:</strong> Closing localhost, stopping Streamlit, or restarting your computer will <strong>NOT</strong> wipe the memory. The model and its mistake remediations automatically reload from disk upon startup.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Control Bar
        sc_col1, sc_col2, sc_col3 = st.columns([2, 1.5, 1.5])
        with sc_col1:
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            run_sc_train_btn = st.button(" RUN SELF-CORRECTING MISTAKE REMEDIATION TRAINING", type="primary", use_container_width=True)
        with sc_col2:
            sc_retrain_freq = st.selectbox("RETRAINING CADENCE", [4, 2, 8], format_func=lambda f: f"Every {f} Weeks (Replay Cycle)", index=0)
        with sc_col3:
            sc_threshold = st.slider("INCIDENT RISK THRESHOLD", 0.20, 0.80, 0.50, 0.05, key="sc_thresh_slider")

        # Session State for Self-Correcting Training
        if "self_correcting_training_output" not in st.session_state or run_sc_train_btn:
            if run_sc_train_btn:
                with st.spinner("Executing full 52-week self-correcting training loop with mistake diagnosis and experience replay..."):
                    try:
                        sc_output = train_with_self_correcting_loop(
                            df=df_full,
                            threshold=sc_threshold,
                            retrain_frequency_weeks=sc_retrain_freq
                        )
                        st.session_state["self_correcting_training_output"] = sc_output
                        st.success("Self-Correction Training Complete! Mistake log and fine-tuned model saved to disk.")
                    except Exception as e:
                        st.error(f"Error during self-correcting training: {e}")

        if "self_correcting_training_output" in st.session_state and st.session_state["self_correcting_training_output"]:
            sc_data = st.session_state["self_correcting_training_output"]
            m_log_df = sc_data["mistake_log_df"]
            steps_df = sc_data["steps_df"]

            st.markdown("---")
            st.markdown("##### 1. Self-Correction Training Scorecard (Before vs After Adaptation)")

            # Top KPI Summary Cards
            sk1, sk2, sk3, sk4 = st.columns(4)
            with sk1:
                render_kpi_card("Mistakes Analyzed & Remediated", f"{sc_data['total_mistakes_identified']:,}", "False Negatives & False Alarms", "#f59e0b")
            with sk2:
                acc_gain = sc_data['accuracy_gain_pct']
                render_kpi_card("Accuracy Evolution", f"{sc_data['self_corrected_accuracy']:.1f}%", f"Baseline: {sc_data['baseline_accuracy']:.1f}% ({'+' if acc_gain >= 0 else ''}{acc_gain:.2f}%)", "#22c55e")
            with sk3:
                fa_red = sc_data['false_alarm_reduction']
                render_kpi_card("False Alarms Eliminated", f"{fa_red:+d} Alarms", f"{sc_data['baseline_false_alarms']} -> {sc_data['self_corrected_false_alarms']} False Alarms", "#38bdf8")
            with sk4:
                rew_gain = sc_data['reward_gain']
                render_kpi_card("Reinforcement Reward Gain", f"{sc_data['self_corrected_cumulative_reward']:+,.0f} pts", f"Net Gain: {'+' if rew_gain >= 0 else ''}{rew_gain:.1f} pts", "#a78bfa")

            # Comparative Progress Table
            comp_df = pd.DataFrame([
                {
                    "Metric": "Overall Prediction Accuracy",
                    "Initial Baseline Model": f"{sc_data['baseline_accuracy']:.2f}%",
                    "Self-Corrected Model (With Mistake Replay)": f"{sc_data['self_corrected_accuracy']:.2f}%",
                    "Net Improvement": f"{'+' if acc_gain >= 0 else ''}{acc_gain:.2f}%"
                },
                {
                    "Metric": "False Alarm Count (False Positives)",
                    "Initial Baseline Model": f"{sc_data['baseline_false_alarms']} false alarms",
                    "Self-Corrected Model (With Mistake Replay)": f"{sc_data['self_corrected_false_alarms']} false alarms",
                    "Net Improvement": f"{fa_red:+d} false alarms avoided"
                },
                {
                    "Metric": "Cumulative Reinforcement Payoff",
                    "Initial Baseline Model": f"{sc_data['baseline_cumulative_reward']:+,.1f} pts",
                    "Self-Corrected Model (With Mistake Replay)": f"{sc_data['self_corrected_cumulative_reward']:+,.1f} pts",
                    "Net Improvement": f"{'+' if rew_gain >= 0 else ''}{rew_gain:.1f} pts"
                }
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # 2. Interactive Mistake Diagnosis & Remediation Ledger Table
            st.markdown("##### 2. Detailed Mistake Diagnosis & Remediation Ledger")
            st.caption("Every prediction mistake detected during the training loop was automatically diagnosed for root causes and remediated with sample loss weighting and decision boundary shifts.")

            filter_type = st.radio(
                "Filter Mistake Category",
                ["All Identified Mistakes", " False Negatives Only (Missed Hazards)", " False Positives Only (False Alarms)"],
                horizontal=True
            )

            display_mistakes = m_log_df.copy()
            if "False Negatives" in filter_type:
                display_mistakes = display_mistakes[display_mistakes["outcome_type"] == "FALSE_NEGATIVE"]
            elif "False Positives" in filter_type:
                display_mistakes = display_mistakes[display_mistakes["outcome_type"] == "FALSE_POSITIVE"]

            st.dataframe(
                display_mistakes.rename(columns={
                    "mistake_id": "Mistake ID",
                    "week": "Target Week",
                    "zone_id": "Corridor Zone",
                    "feeder_window": "4-Week Feeder Context",
                    "outcome_type": "Error Type",
                    "predicted_prob": "Predicted Risk P",
                    "actual_target": "Actual Incident",
                    "where_it_went_wrong": "Where It Went Wrong (Root Cause Diagnosis)",
                    "fix_action_applied": "Fix Action Applied (Focal Weight & Replay Fix)",
                    "focal_weight_boost": "Focal Weight Boost",
                    "resolution_status": "Remediation Status"
                }),
                use_container_width=True,
                height=340
            )

            # Download Buttons
            st.markdown("##### 3. Export Self-Corrected Model & Remediation Logs")
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                if os.path.exists("models/self_correcting_risk_model.pkl"):
                    with open("models/self_correcting_risk_model.pkl", "rb") as f:
                        m_bytes = f.read()
                    st.download_button(
                        " DOWNLOAD SELF-CORRECTED ML MODEL (self_correcting_risk_model.pkl)",
                        data=m_bytes,
                        file_name="self_correcting_risk_model.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
            with d_col2:
                csv_m_bytes = m_log_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    " DOWNLOAD MISTAKE REMEDIATION LOG (mistake_remediation_log.csv)",
                    data=csv_m_bytes,
                    file_name="mistake_remediation_log.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("Click **' RUN SELF-CORRECTING MISTAKE REMEDIATION TRAINING'** to begin the training loop.")
