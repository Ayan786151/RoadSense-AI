"""
================================================================================
ROAD SENSE AI - INTERACTIVE VISION STUDIO & VIOLATION ENFORCEMENT HUB
================================================================================
Allows running live vehicle detection, no-helmet violation tracking, red-light
breaking detection, CLAHE night preprocessor, perspective speed telemetry,
and IISc / custom model training pipelines directly from the browser UI.
================================================================================
"""

import os
import glob
import time
import cv2
import html
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pathlib import Path
from ultralytics import YOLO

from vision.vehicle_detector import (
    process_video,
    resolve_vehicle_class,
    fuse_driver_pillion_riders,
    DEFAULT_MODEL_NAME
)
from vision.enhancement import adaptive_preprocess_frame
from vision.train_uvh26 import UVH26_CLASSES
from vision.helmet_detector import HelmetViolationDetector
from vision.red_light_detector import RedLightViolationDetector
from vision.triple_riding_detector import TripleRidingDetector
from intelligence.echallan_generator import create_echallan_record, render_echallan_html, PENAL_CODE_DIRECTORY

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


# Bounding box color palette (BGR)
CLASS_COLORS = {
    "motorcycle": (0, 215, 255),    # Golden Yellow
    "auto_rickshaw": (255, 255, 0), # Cyan
    "car": (0, 255, 127),           # Emerald Green
    "bus": (0, 140, 255),           # Vivid Orange
    "truck": (0, 0, 255)            # Bright Red
}

CLASS_DISPLAY_LABELS = {
    "motorcycle": "2-Wheeler",
    "auto_rickshaw": "Auto",
    "car": "Car",
    "bus": "Bus",
    "truck": "Truck/LCV"
}


def resolve_youtube_stream_url(url: str):
    """Extracts direct streamable URL and metadata from a YouTube video or live stream."""
    if not yt_dlp:
        raise RuntimeError("yt-dlp package is not installed. Please run 'pip install yt-dlp'.")
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best[height<=720]/best',
        'quiet': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        stream_url = info.get('url')
        title = info.get('title', 'YouTube Stream')
        is_live = info.get('is_live', False)
        return stream_url, title, is_live


def download_youtube_clip(url: str, output_path: str, max_duration_sec: int = 30):
    """Downloads a short preview clip from YouTube for offline local showcase."""
    if not yt_dlp:
        raise RuntimeError("yt-dlp package is not installed.")
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best[height<=720]/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'download_ranges': lambda info_dict, ydl: [{'start_time': 0, 'end_time': max_duration_sec}]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def render_vision_studio():
    """Renders the Interactive Vision Studio, Violation Enforcement & Model Hub."""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f1c2c, #928DAB); padding: 22px 28px; border-radius: 12px; margin-bottom: 24px; color: white;">
        <h2 style="margin: 0; font-weight: 700; display: flex; align-items: center; gap: 12px;">
            🚀 Interactive Computer Vision Studio & AI Violation Hub
        </h2>
        <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 14px;">
            Run live video detection with <b>No-Helmet Tracking</b>, <b>Red-Light Breaking Enforcement</b>, adaptive lighting, and train specialized safety models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_runner, tab_violations, tab_uvh26 = st.tabs([
        "🎬 Run Live Video & Violation Detector",
        "🪖 Violation Enforcement & Model Trainer",
        "🇮🇳 IISc Indian Traffic Benchmark Hub"
    ])

    # ==========================================================================
    # TAB 1: RUN LIVE VIDEO & VIOLATION DETECTOR
    # ==========================================================================
    with tab_runner:
        st.markdown("#### 📹 Real-Time CCTV Video Processor with Multi-Violation Tracking")
        st.caption("Select traffic footage or paste a live YouTube stream, configure violation enforcement rules, and process video frames live.")

        media_source = st.radio(
            "Media Input Source",
            ["📁 Local Video Footage (videos/*.mp4)", "🔴 YouTube Live Stream / Video URL"],
            horizontal=True
        )

        col1, col2 = st.columns([1, 1])

        yt_stream_url = None
        is_yt_source = (media_source == "🔴 YouTube Live Stream / Video URL")

        with col1:
            if not is_yt_source:
                existing_videos = glob.glob("videos/*.mp4") + glob.glob("videos/*.avi")
                vid_options = existing_videos if existing_videos else ["videos/traffic.mp4"]
                selected_video = st.selectbox("Select Input Traffic Video", vid_options, index=0)
            else:
                yt_input_url = st.text_input(
                    "YouTube Video / Live Stream URL",
                    value="https://www.youtube.com/watch?v=1H0iTzv2jiQ",
                    help="Paste any YouTube video or live traffic webcam stream URL."
                )
                selected_video = yt_input_url
                
                cache_col1, cache_col2 = st.columns([1, 1])
                with cache_col1:
                    yt_max_frames = st.slider("Live Stream Process Cap (Frames)", 50, 1500, 300, step=50, help="Caps frames for live streams to build observation metrics.")
                with cache_col2:
                    if st.button("⬇️ Cache 30s Sample to videos/"):
                        with st.spinner("Downloading 30s sample clip for offline showcase..."):
                            try:
                                cache_file = "videos/yt_cached_sample.mp4"
                                download_youtube_clip(yt_input_url, cache_file, max_duration_sec=30)
                                st.success(f"Saved to `{cache_file}`! You can now select it in Local Videos.")
                            except Exception as dl_err:
                                st.error(f"Failed to cache YouTube clip: {dl_err}")
            
            model_choices = [
                "yolo11s.pt (Universal Traffic Model — High Precision)",
                "yolo11n.pt (Ultra-Fast 60+ FPS)",
                "models/iisc_yolov11s_indian_traffic.pt (IISc Bangalore SafeCity Weights)"
            ]
            selected_model_str = st.selectbox("Detection Model Weights", model_choices, index=0)
            if "iisc_yolov11s" in selected_model_str:
                model_target = "models/iisc_yolov11s_indian_traffic.pt"
            elif "yolo11n" in selected_model_str:
                model_target = "yolo11n.pt"
            else:
                model_target = "yolo11s.pt"

            preview_size = st.select_slider("Live Video Display Size", options=["Compact (540px)", "Standard (720px)", "Full Width"], value="Standard (720px)")

        with col2:
            new_session_name = st.text_input("Output Session ID", value=f"session_{int(time.time()) % 1000:03d}")
            conf_thresh = st.slider("Detection Confidence Threshold", 0.20, 0.65, 0.35, step=0.05)
            frame_skip = st.slider("Process Every Nth Frame (Speed Multiplier)", 1, 15, 5)
            enable_night = st.checkbox("🌙 Force Adaptive CLAHE Night Enhancement", value=False)

        st.markdown("##### 🚨 Active AI Violation Enforcement Modules:")
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        with v_col1:
            track_helmets = st.checkbox("🪖 No-Helmet Tracking", value=True, help="Detects two-wheeler riders without protective helmets.")
        with v_col2:
            track_red_lights = st.checkbox("🚦 Red-Light Breaking", value=True, help="Monitors virtual stop-line intrusion during red signal phases.")
        with v_col3:
            track_triple = st.checkbox("👥 Triple-Riding Violations", value=True, help="Detects overloaded two-wheelers carrying >2 persons.")
        with v_col4:
            stop_line_ratio = st.slider("Stop-Line Position", 0.40, 0.90, 0.65, step=0.05, help="Vertical screen height ratio of the intersection stop line.")

        st.markdown("---")

        run_btn = st.button("▶️ Start Live Detection & Violation Enforcement", type="primary", use_container_width=True)

        # Container for live preview
        st.markdown("<br>", unsafe_allow_html=True)
        progress_bar = st.empty()
        status_text = st.empty()
        preview_container = st.empty()

        if run_btn:
            stream_source = selected_video
            can_proceed = True

            if is_yt_source:
                status_text.info(f"⏳ Resolving YouTube stream URL for `{selected_video}`...")
                try:
                    yt_stream_url, yt_title, yt_is_live = resolve_youtube_stream_url(selected_video)
                    stream_source = yt_stream_url
                    st.toast(f"Connected to YouTube: {yt_title}", icon="🔴")
                except Exception as yt_err:
                    status_text.empty()
                    st.error(f"❌ YouTube Stream Error: {yt_err}\n\nPlease check the URL or ensure the stream is publicly accessible.")
                    can_proceed = False
            elif not os.path.exists(selected_video):
                st.error(f"Selected video not found: {selected_video}")
                can_proceed = False

            if can_proceed:
                progress = progress_bar.progress(0)
                status_text.info(f"⏳ Initializing YOLO tracker ({model_target}) & Violation Detectors...")

                try:
                    cap = cv2.VideoCapture(stream_source)
                    if not cap.isOpened():
                        st.error(f"Could not open video stream: {stream_source}")
                        st.stop()

                    raw_total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if is_yt_source:
                        total_f = yt_max_frames if raw_total_f <= 0 else min(raw_total_f, yt_max_frames)
                    else:
                        total_f = raw_total_f if raw_total_f > 0 else 500
                    
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    
                    yolo_model = YOLO(model_target)
                    helmet_detector = HelmetViolationDetector() if track_helmets else None
                    red_light_detector = RedLightViolationDetector(stop_line_y_ratio=stop_line_ratio) if track_red_lights else None
                    triple_detector = TripleRidingDetector() if track_triple else None

                    frame_idx = 0
                    processed_records = []
                    violation_records = []
                    
                    sess_dir = Path("data/sessions") / new_session_name
                    sess_dir.mkdir(parents=True, exist_ok=True)

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_idx += 1
                        if is_yt_source and frame_idx >= total_f:
                            break

                        if frame_idx % frame_skip != 0:
                            continue

                        timestamp = round(frame_idx / fps, 2)

                        # Adaptive CLAHE Enhancement
                        proc_frame, audit = adaptive_preprocess_frame(frame, force_enhancement=enable_night)

                        # Universal YOLO tracking
                        results = yolo_model.track(
                            proc_frame,
                            persist=True,
                            tracker="bytetrack.yaml",
                            conf=conf_thresh,
                            imgsz=640,
                            verbose=False
                        )
                        result = results[0]

                        detected_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "auto_rickshaw": 0}
                        total_v = 0
                        frame_tracked_vehicles = []

                        if result.boxes is not None:
                            boxes = result.boxes
                            for i in range(len(boxes)):
                                cid = int(boxes.cls[i].item())
                                vtype = resolve_vehicle_class(cid, yolo_model.names)
                                if not vtype:
                                    continue

                                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                                t_id = int(boxes.id[i].item()) if boxes.id is not None else None
                                x1, y1, x2, y2 = xyxy

                                detected_counts[vtype] = detected_counts.get(vtype, 0) + 1
                                total_v += 1
                                
                                frame_tracked_vehicles.append({
                                    "track_id": t_id,
                                    "vehicle_type": vtype,
                                    "center_x": (x1 + x2) / 2.0,
                                    "center_y": float(y2),
                                    "bbox": [x1, y1, x2, y2]
                                })

                                # Visual styling
                                color = CLASS_COLORS.get(vtype, (0, 255, 0))
                                label_text = CLASS_DISPLAY_LABELS.get(vtype, vtype)
                                
                                # Draw clean bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Draw filled label tag
                                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                                tag_y1 = max(y1 - th - 6, 0)
                                cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 6, tag_y1 + th + 6), color, -1)
                                cv2.putText(frame, label_text, (x1 + 3, tag_y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

                                # 1. Helmet Compliance Verification
                                if track_helmets and helmet_detector and vtype == "motorcycle":
                                    h_eval = helmet_detector.analyze_motorcycle_rider(frame, xyxy, t_id, timestamp)
                                    helmet_detector.draw_annotation(frame, h_eval)
                                    if not h_eval["has_helmet"] and t_id is not None:
                                        violation_records.append({
                                            "timestamp_seconds": timestamp,
                                            "frame_number": frame_idx,
                                            "track_id": t_id,
                                            "vehicle_type": "2-Wheeler",
                                            "violation_type": "NO_HELMET",
                                            "severity": "HIGH",
                                            "confidence": f"{h_eval['confidence']*100:.0f}%",
                                            "details": h_eval["reason"]
                                        })

                                # 2. Triple-Riding Verification
                                if track_triple and triple_detector and vtype == "motorcycle":
                                    tr_eval = triple_detector.analyze_motorcycle_occupancy(frame, xyxy, t_id, timestamp)
                                    if tr_eval["is_triple_riding"] and t_id is not None:
                                        violation_records.append({
                                            "timestamp_seconds": timestamp,
                                            "frame_number": frame_idx,
                                            "track_id": t_id,
                                            "vehicle_type": "2-Wheeler",
                                            "violation_type": "TRIPLE_RIDING",
                                            "severity": "HIGH",
                                            "confidence": f"{tr_eval['confidence']*100:.0f}%",
                                            "details": f"Overloaded: {tr_eval['estimated_riders']} persons seated on 2-wheeler"
                                        })

                        # 3. Red-Light Violation Check
                        if track_red_lights and red_light_detector:
                            active_rl, phase = red_light_detector.process_frame_violations(frame, frame_tracked_vehicles, timestamp)
                            red_light_detector.draw_annotation(frame, phase, active_rl)
                            for viol in active_rl:
                                violation_records.append({
                                    "timestamp_seconds": timestamp,
                                    "frame_number": frame_idx,
                                    "track_id": viol["track_id"],
                                    "vehicle_type": viol["vehicle_type"],
                                    "violation_type": "RED_LIGHT_RUNNING",
                                    "severity": "CRITICAL",
                                    "confidence": "95%",
                                    "details": f"Crossed Stop-Line during RED phase (y={viol['cross_y']})"
                                })

                        processed_records.append({
                            "timestamp_seconds": timestamp,
                            "vehicle_count": total_v,
                            "cars": detected_counts.get("car", 0),
                            "motorcycles": detected_counts.get("motorcycle", 0),
                            "buses": detected_counts.get("bus", 0),
                            "trucks": detected_counts.get("truck", 0),
                            "auto_rickshaws": detected_counts.get("auto_rickshaw", 0)
                        })

                        pct = min(1.0, frame_idx / max(total_f, 1))
                        progress.progress(pct)
                        
                        no_helmet_cnt = len(helmet_detector.logged_violations) if helmet_detector else 0
                        red_light_cnt = len(red_light_detector.logged_violations) if red_light_detector else 0
                        triple_cnt = len(triple_detector.logged_violations) if triple_detector else 0
                        status_text.text(f"Frame {frame_idx}/{total_f} • Vehicles: {total_v} | 🚨 Violations: 🪖 No-Helmet: {no_helmet_cnt}, 🚦 Red-Light: {red_light_cnt}, 👥 Triple-Riding: {triple_cnt}")

                        # Live visual preview update
                        if frame_idx % (frame_skip * 3) == 0:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            if preview_size == "Compact (540px)":
                                preview_container.image(rgb_frame, width=540, caption=f"Live HUD Preview (Frame {frame_idx})")
                            elif preview_size == "Standard (720px)":
                                preview_container.image(rgb_frame, width=720, caption=f"Live HUD Preview (Frame {frame_idx})")
                            else:
                                preview_container.image(rgb_frame, use_container_width=True, caption=f"Live HUD Preview (Frame {frame_idx})")

                    cap.release()
                    progress.progress(1.0)

                    # Save traffic metrics
                    out_df = pd.DataFrame(processed_records)
                    out_csv = sess_dir / "vision_traffic_metrics.csv"
                    out_df.to_csv(out_csv, index=False)

                    obs_csv = sess_dir / "live_traffic_observations.csv"
                    obs_df = out_df.copy()
                    obs_df["average_speed_kmh"] = np.nan
                    obs_df["speed_source"] = "uncalibrated"
                    obs_df.to_csv(obs_csv, index=False)

                    # Save violations log
                    viol_df = pd.DataFrame(violation_records)
                    if not viol_df.empty:
                        viol_df = viol_df.drop_duplicates(subset=["track_id", "violation_type"]).reset_index(drop=True)
                    viol_csv = sess_dir / "violation_events.csv"
                    viol_df.to_csv(viol_csv, index=False)

                    status_text.success(f"✅ Video processing complete! Metrics & Violation logs saved to `{new_session_name}`.")
                    st.balloons()

                    # Calculate fine recovery
                    total_fines = 0
                    for _, vrow in viol_df.iterrows():
                        vtype = vrow.get("violation_type", "")
                        fine = PENAL_CODE_DIRECTORY.get(vtype, {}).get("fine_inr", 1000)
                        total_fines += fine

                    # Summary Metrics Cards
                    st.markdown("### 📊 Detection & Violation Summary")
                    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
                    sm1.metric("Processed Frames", len(out_df))
                    sm2.metric("Peak Vehicle Density", int(out_df["vehicle_count"].max()))
                    sm3.metric("🪖 No-Helmet", len(helmet_detector.logged_violations) if helmet_detector else 0)
                    sm4.metric("🚦 Red-Light", len(red_light_detector.logged_violations) if red_light_detector else 0)
                    sm5.metric("💰 Fine Potential", f"₹{total_fines:,}")

                    # Violation Evidence Log Table & E-Challan Inspector
                    if not viol_df.empty:
                        st.markdown("#### 🚨 Detected Traffic Safety Violations & E-Challan Dispatch")
                        st.dataframe(viol_df, use_container_width=True)

                        c_down1, c_down2 = st.columns([1, 1])
                        with c_down1:
                            st.download_button(
                                "📥 Download Violation Evidence CSV",
                                viol_df.to_csv(index=False),
                                file_name=f"{new_session_name}_violations.csv",
                                mime="text/csv"
                            )
                        with c_down2:
                            st.markdown(f"**Total Legal Citations Pending:** `{len(viol_df)} Citations` (₹{total_fines:,} INR)")

                        # Interactive E-Challan Ticket Viewer
                        st.markdown("##### 📜 Instant Digital E-Challan Ticket Inspector")
                        violation_options = [f"#{row['track_id']} - {row['violation_type']} (Frame {row['frame_number']})" for _, row in viol_df.iterrows()]
                        selected_challan_idx = st.selectbox("Select Violation to Generate Official Digital Citation:", range(len(violation_options)), format_func=lambda i: violation_options[i])
                        
                        selected_viol = viol_df.iloc[selected_challan_idx].to_dict()
                        challan_doc = create_echallan_record(selected_viol)
                        ticket_html = render_echallan_html(challan_doc)
                        
                        st.components.v1.html(ticket_html, height=380, scrolling=True)

                    else:
                        st.info("✅ Zero traffic safety violations recorded in this observation session.")

                    # Plot results
                    st.markdown("##### 📈 Real-Time Vehicle Flow Breakdown")
                    fig_res = px.line(
                        out_df,
                        x="timestamp_seconds",
                        y=["cars", "motorcycles", "buses", "trucks"],
                        title="Vehicle Class Flow over Video Timeline",
                        labels={"timestamp_seconds": "Elapsed Time (Seconds)", "value": "Count", "variable": "Category"},
                        color_discrete_map={"cars": "#51cf66", "motorcycles": "#ffd43b", "buses": "#ff922b", "trucks": "#ff6b6b"},
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_res, use_container_width=True)

                except Exception as e:
                    status_text.error(f"Error during video processing: {e}")

    # ==========================================================================
    # TAB 2: VIOLATION ENFORCEMENT & MODEL TRAINER HUB
    # ==========================================================================
    with tab_violations:
        st.markdown("#### 🪖 Safety Violation AI Model Training Hub")
        st.caption("Fine-tune custom YOLO models to detect No-Helmet and Traffic Light violations on your dataset.")

        t_col1, t_col2 = st.columns([1, 1])

        with t_col1:
            st.markdown("##### 🛠️ Violation Model Training Parameters")
            train_task = st.selectbox("Target Violation Task", ["Helmet / No-Helmet Detection", "Traffic Signal Phase Detection"])
            task_key = "helmet" if "Helmet" in train_task else "traffic_light"
            base_model = st.selectbox("Backbone Model", ["yolo11n.pt (Fastest, Edge-Deployable)", "yolo11s.pt (Balanced, High Accuracy)", "yolo11m.pt (Heavyweight)"])
            base_model_file = base_model.split(" ")[0]

            epochs = st.slider("Training Epochs", 5, 100, 30, step=5)
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)

        with t_col2:
            st.markdown("##### 📁 Dataset Configuration")
            dataset_path = st.text_input("Dataset YAML Path", value=f"data/{task_key}_dataset.yaml")
            output_dir = st.text_input("Output Weights Directory", value="models/violation_models")

            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid #e11d48; border-radius: 8px; padding: 14px; margin-top: 15px;">
                <div style="color: #f43f5e; font-weight: 700; font-size: 13px;">💻 One-Line Terminal Training Command</div>
                <pre style="background: #000; color: #fb7185; padding: 10px; border-radius: 6px; font-size: 12px; margin-top: 8px; overflow-x: auto;">python -m vision.train_violation_model \\
  --task {task_key} \\
  --data {dataset_path} \\
  --model {base_model_file} \\
  --epochs {epochs} \\
  --batch {batch_size} \\
  --output {output_dir}</pre>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 3: IISC UVH-26 & BMD-45 BENCHMARK HUB
    # ==========================================================================
    with tab_uvh26:
        st.markdown("#### 🇮🇳 IISc AIM Indian Traffic Benchmark Datasets")
        st.caption("Developed by AI for Integrated Mobility (AIM) @ Indian Institute of Science (IISc), Bengaluru.")

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #3b82f6; border-radius: 10px; padding: 16px; min-height: 190px;">
                <div style="color: #60a5fa; font-weight: 700; font-size: 15px;">📊 IISc UVH-26 Benchmark</div>
                <div style="font-size: 12px; color: #94a3b8; margin: 4px 0 10px 0;">Urban Vehicle Heterogeneity (2025)</div>
                <ul style="font-size: 13px; color: #e2e8f0; margin: 0; padding-left: 18px; line-height: 1.6;">
                    <li><b>26,646 1080p CCTV Images</b></li>
                    <li><b>1.8 Million Bounding Boxes</b></li>
                    <li>14 Indian Vehicle Classes (Auto, Tempo, etc.)</li>
                    <li>Fast fine-tuning convergence</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with d_col2:
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #10b981; border-radius: 10px; padding: 16px; min-height: 190px;">
                <div style="color: #34d399; font-weight: 700; font-size: 15px;">🚀 IISc BMD-45 Flagship Dataset</div>
                <div style="font-size: 12px; color: #94a3b8; margin: 4px 0 10px 0;">Bengaluru Mobility Dataset (2026)</div>
                <ul style="font-size: 13px; color: #e2e8f0; margin: 0; padding-left: 18px; line-height: 1.6;">
                    <li><b>45,000 CCTV Images</b></li>
                    <li><b>3,600+ Safe City Cameras</b></li>
                    <li><b>480,000+ Verified Annotations</b></li>
                    <li>Maximum geographic and viewpoint diversity</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        u_col1, u_col2 = st.columns([3, 2])

        with u_col1:
            st.markdown("##### 🚗 Unified 14 Indian Vehicle Categories:")
            class_tags = "".join([f'<span style="display: inline-block; background: rgba(77, 171, 247, 0.15); color: #74c0fc; border: 1px solid rgba(77, 171, 247, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 12px; margin: 3px;">{cls.replace("_", " ").title()}</span>' for cls in UVH26_CLASSES])
            st.markdown(f'<div style="margin-bottom: 20px;">{class_tags}</div>', unsafe_allow_html=True)

        with u_col2:
            st.markdown("##### ⚙️ One-Click Fine-Tuning Setup")
            selected_ds = st.selectbox("Target IISc Dataset", ["iisc-aim/BMD-45 (45k Images)", "iisc-aim/UVH-26 (26k Images)"])
            ds_name = "iisc-aim/BMD-45" if "BMD-45" in selected_ds else "iisc-aim/UVH-26"
            
            st.markdown(f"""
            <div style="background: rgba(18, 30, 49, 0.7); border: 1px solid rgba(77, 171, 247, 0.3); border-radius: 10px; padding: 16px; margin-top: 10px;">
                <div style="font-size: 13px; font-weight: 700; color: #69db7c;">🎯 Terminal / Cloud Training Command</div>
                <pre style="background: #000; color: #51cf66; padding: 10px; border-radius: 6px; font-size: 12px; margin-top: 8px; overflow-x: auto;">python -m vision.train_uvh26 \\
  --data {ds_name} \\
  --model yolo11s.pt \\
  --epochs 30 \\
  --batch 16</pre>
            </div>
            """, unsafe_allow_html=True)
