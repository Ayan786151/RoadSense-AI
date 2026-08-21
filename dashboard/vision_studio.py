"""
================================================================================
ROADSENSE AI — COMPUTER VISION & CIVIL VIOLATION ENFORCEMENT STUDIO
DESIGN SYSTEM: KINETIC INFRASTRUCTURE INTELLIGENCE (STITCH MCP)
================================================================================
"""

import os
import glob
import time
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from ultralytics import YOLO

from vision.enhancement import adaptive_preprocess_frame
from vision.vehicle_detector import resolve_vehicle_class
from vision.train_uvh26 import UVH26_CLASSES
from vision.helmet_detector import HelmetViolationDetector
from vision.red_light_detector import RedLightViolationDetector
from vision.triple_riding_detector import TripleRidingDetector
from vision.adaptive_engine import AutonomousAdaptiveEngine
from intelligence.echallan_generator import create_echallan_record, render_echallan_html, PENAL_CODE_DIRECTORY

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


# Subtle, high-contrast BGR bounding box palette
CLASS_COLORS = {
    "motorcycle": (250, 250, 250),   # White
    "auto_rickshaw": (180, 180, 180), # Silver Gray
    "car": (220, 220, 220),          # Off White
    "bus": (160, 160, 160),          # Neutral Gray
    "truck": (130, 130, 130)         # Dark Gray
}

CLASS_DISPLAY_LABELS = {
    "motorcycle": "2-WHEELER",
    "auto_rickshaw": "AUTO",
    "car": "CAR",
    "bus": "BUS",
    "truck": "TRUCK/LCV"
}


def resolve_youtube_stream_url(url: str):
    """Extracts direct streamable URL and metadata from a YouTube video or live stream."""
    if not yt_dlp:
        raise RuntimeError("yt-dlp package is not installed. Please run 'pip install yt-dlp'.")
    ydl_opts = {
        'format': 'best/bestvideo/best',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'web_embedded', 'mweb']
            }
        }
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        stream_url = info.get('url')
        if not stream_url and 'formats' in info:
            for f in reversed(info['formats']):
                if f.get('url'):
                    stream_url = f.get('url')
                    break
        title = info.get('title', 'YouTube Stream')
        is_live = info.get('is_live', False)
        return stream_url, title, is_live


def download_youtube_clip(url: str, output_path: str, max_duration_sec: int = 30):
    """Downloads a short preview clip from YouTube for offline local showcase."""
    if not yt_dlp:
        raise RuntimeError("yt-dlp package is not installed.")
    ydl_opts = {
        'format': 'best/bestvideo/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'web_embedded', 'mweb']
            }
        },
        'download_ranges': lambda info_dict, ydl: [{'start_time': 0, 'end_time': max_duration_sec}]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def render_vision_studio():
    """Renders the Minimalist Computer Vision Studio and Violation Enforcement Hub."""
    st.markdown("""
    <div class="telemetry-header">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; letter-spacing: 0.06em; text-transform: uppercase;">
            VISION CORE 03
        </div>
        <h2 style="margin: 4px 0 0 0; font-size: 22px;">
            Computer Vision Telemetry & Violation Enforcement
        </h2>
        <p style="margin: 6px 0 0 0; color: #a1a1aa; font-size: 13px;">
            Real-time multi-class tracking, automated stop-line intrusion enforcement, rider helmet compliance, and zero-hardcoded scene adaptation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_runner, tab_violations, tab_uvh26 = st.tabs([
        "LIVE VIDEO SURVEILLANCE & VIOLATION RADAR",
        "SAFETY MODEL TRAINING ENGINE",
        "IISC URBAN TRAFFIC BENCHMARK DATASET"
    ])

    # ==========================================================================
    # TAB 1: RUN LIVE VIDEO & VIOLATION DETECTOR
    # ==========================================================================
    with tab_runner:
        st.markdown("#### Real-Time Video Ingestion & Multi-Violation Tracking")
        st.caption("Select local traffic footage or configure a live stream feed for real-time kinematic processing.")

        media_source = st.radio(
            "MEDIA INPUT SOURCE",
            ["Local Video Archive (videos/*.mp4)", "YouTube Live Stream / Video URL"],
            horizontal=True
        )

        col1, col2 = st.columns([1, 1])

        yt_stream_url = None
        is_yt_source = (media_source == "YouTube Live Stream / Video URL")
        continuous_live = False

        with col1:
            if not is_yt_source:
                existing_videos = glob.glob("videos/*.mp4") + glob.glob("videos/*.avi")
                vid_options = existing_videos if existing_videos else ["videos/traffic.mp4"]
                selected_video = st.selectbox("SELECT INPUT VIDEO", vid_options, index=0)
            else:
                yt_input_url = st.text_input(
                    "YOUTUBE STREAM URL",
                    value="https://www.youtube.com/watch?v=1H0iTzv2jiQ",
                    help="Public YouTube traffic surveillance camera URL."
                )
                selected_video = yt_input_url
                
                cache_col1, cache_col2 = st.columns([1, 1])
                with cache_col1:
                    continuous_live = st.checkbox("Continuous Live Feed (Infinite Stream)", value=True)
                    if not continuous_live:
                        yt_max_frames = st.slider("Frame Limit", 50, 3000, 500, step=50)
                    else:
                        yt_max_frames = 999999999
                with cache_col2:
                    if st.button("Cache 30s Offline Sample"):
                        with st.spinner("Downloading 30s clip for offline showcase..."):
                            try:
                                cache_file = "videos/yt_cached_sample.mp4"
                                download_youtube_clip(yt_input_url, cache_file, max_duration_sec=30)
                                st.success(f"Saved to {cache_file}")
                            except Exception as dl_err:
                                st.error(f"Cache failed: {dl_err}")
            
            model_choices = [
                "yolo11n.pt (Ultra-Fast 60+ FPS — Recommended for CPU)",
                "yolo11s.pt (Universal Traffic Model — High Precision)",
                "models/iisc_yolov11s_indian_traffic.pt (IISc Bangalore SafeCity Weights)"
            ]
            selected_model_str = st.selectbox("DETECTION MODEL WEIGHTS", model_choices, index=0)
            if "iisc_yolov11s" in selected_model_str:
                model_target = "models/iisc_yolov11s_indian_traffic.pt"
            elif "yolo11s" in selected_model_str:
                model_target = "yolo11s.pt"
            else:
                model_target = "yolo11n.pt"

            preview_size = st.select_slider("DISPLAY SCALE", options=["Compact (540px)", "Standard (720px)", "Full Width"], value="Standard (720px)")

        with col2:
            new_session_name = st.text_input("OUTPUT SESSION ID", value=f"session_{int(time.time()) % 1000:03d}")
            conf_thresh = st.slider("CONFIDENCE THRESHOLD", 0.20, 0.65, 0.35, step=0.05)
            frame_skip = st.slider("FRAME SKIP MULTIPLIER", 1, 15, 3)
            enable_night = st.checkbox("Force CLAHE Low-Light Enhancement", value=False)

        st.markdown("##### Enforcement Modules:")
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        with v_col1:
            track_helmets = st.checkbox("No-Helmet Tracking", value=True)
        with v_col2:
            track_red_lights = st.checkbox("Red-Light Stop-Line Radar", value=True)
        with v_col3:
            track_triple = st.checkbox("Triple-Riding Violations", value=True)
        with v_col4:
            auto_mode = st.checkbox("Autonomous Scene Perception Engine", value=True)
            if not auto_mode:
                stop_line_ratio = st.slider("Manual Stop-Line", 0.40, 0.90, 0.65, step=0.05)
            else:
                stop_line_ratio = 0.60

        st.markdown("---")

        run_btn = st.button("START LIVE SURVEILLANCE & VIOLATION ENFORCEMENT", type="primary", width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)
        progress_bar = st.empty()
        status_text = st.empty()
        preview_container = st.empty()

        if run_btn:
            stream_source = selected_video
            can_proceed = True

            if is_yt_source:
                status_text.info(f"Resolving live stream URL for {selected_video}...")
                try:
                    yt_stream_url, yt_title, yt_is_live = resolve_youtube_stream_url(selected_video)
                    stream_source = yt_stream_url
                    st.toast(f"Connected: {yt_title}")
                except Exception as yt_err:
                    status_text.empty()
                    st.error(f"Stream resolution error: {yt_err}")
                    can_proceed = False
            elif not os.path.exists(selected_video):
                st.error(f"Selected video not found: {selected_video}")
                can_proceed = False

            if can_proceed:
                is_live_infinite = is_yt_source
                if not is_live_infinite:
                    progress = progress_bar.progress(0)
                status_text.info(f"Initializing YOLO model ({model_target}) and Adaptive Perception engine...")

                try:
                    def open_live_capture(url):
                        """Open VideoCapture with short FFmpeg timeout for live streams."""
                        c = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                        c.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
                        c.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
                        c.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                        return c

                    cap = open_live_capture(stream_source) if is_yt_source else cv2.VideoCapture(stream_source)
                    if not cap.isOpened():
                        st.error(f"Could not open video stream: {stream_source}")
                        st.stop()

                    if is_yt_source:
                        total_f = 999999999
                    else:
                        raw_total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        total_f = raw_total_f if raw_total_f > 0 else 500
                    
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    
                    yolo_model = YOLO(model_target)
                    adaptive_engine = AutonomousAdaptiveEngine() if auto_mode else None
                    helmet_detector = HelmetViolationDetector() if track_helmets else None
                    red_light_detector = RedLightViolationDetector(stop_line_y_ratio=stop_line_ratio) if track_red_lights else None
                    triple_detector = TripleRidingDetector() if track_triple else None

                    frame_idx = 0
                    processed_records = []
                    violation_records = []
                    consecutive_fails = 0
                    
                    sess_dir = Path("data/sessions") / new_session_name
                    sess_dir.mkdir(parents=True, exist_ok=True)

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            if is_yt_source:
                                consecutive_fails += 1
                                if consecutive_fails > 5:
                                    status_text.warning("Connection lost after 5 retries. Reconnecting...")
                                    consecutive_fails = 0
                                try:
                                    cap.release()
                                    status_text.info("Refreshing live stream buffer...")
                                    stream_source, _, _ = resolve_youtube_stream_url(selected_video)
                                    cap = open_live_capture(stream_source)
                                    time.sleep(0.5)
                                    ret, frame = cap.read()
                                    if not ret:
                                        time.sleep(1.0)
                                        continue
                                    consecutive_fails = 0
                                except Exception:
                                    time.sleep(1.0)
                                    continue
                            else:
                                break
                        else:
                            consecutive_fails = 0
                        
                        frame_idx += 1
                        if not is_yt_source and frame_idx >= total_f:
                            break

                        if frame_idx % frame_skip != 0:
                            continue

                        timestamp = round(frame_idx / fps, 2)

                        # Adaptive lighting enhancement
                        if adaptive_engine:
                            proc_frame, audit = adaptive_engine.auto_enhance_environment(frame)
                        else:
                            proc_frame, audit = adaptive_preprocess_frame(frame, force_enhancement=enable_night)

                        # Universal YOLO tracking (CPU optimized imgsz)
                        results = yolo_model.track(
                            proc_frame,
                            persist=True,
                            tracker="bytetrack.yaml",
                            conf=conf_thresh,
                            imgsz=480,
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
                                color = CLASS_COLORS.get(vtype, (220, 220, 220))
                                label_text = CLASS_DISPLAY_LABELS.get(vtype, vtype)
                                
                                # Clean 1px bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                                
                                # Clean label tag
                                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                                tag_y1 = max(y1 - th - 4, 0)
                                cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 4, tag_y1 + th + 4), (24, 24, 27), -1)
                                cv2.putText(frame, label_text, (x1 + 2, tag_y1 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

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

                        # 3. Autonomous Adaptive Engine Stop-Line & Signal Phase Adaptation
                        if adaptive_engine and track_red_lights and red_light_detector:
                            auto_y, auto_conf = adaptive_engine.auto_detect_stop_line(frame, frame_tracked_vehicles)
                            red_light_detector.stop_line_y_ratio = auto_y / float(frame.shape[0])
                            
                            auto_phase, auto_p_conf, auto_reason = adaptive_engine.auto_detect_signal_phase(frame, result, frame_tracked_vehicles, fps)
                            red_light_detector.signal_state_override = auto_phase

                        # 4. Red-Light Violation Check
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

                        no_helmet_cnt = len(helmet_detector.logged_violations) if helmet_detector else 0
                        red_light_cnt = len(red_light_detector.logged_violations) if red_light_detector else 0
                        triple_cnt = len(triple_detector.logged_violations) if triple_detector else 0

                        if is_live_infinite:
                            status_text.markdown(f"""
                            <div class="telemetry-badge" style="width: 100%; text-align: left; padding: 8px 12px; margin-bottom: 8px;">
                                STATUS: ACTIVE • FRAME: {frame_idx} | LIVE VEHICLES: {total_v} | NO-HELMET: {no_helmet_cnt} | RED-LIGHT: {red_light_cnt} | TRIPLE: {triple_cnt}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            pct = min(1.0, frame_idx / max(total_f, 1))
                            progress.progress(pct)
                            status_text.text(f"FRAME {frame_idx}/{total_f} • VEHICLES: {total_v} | NO-HELMET: {no_helmet_cnt} | RED-LIGHT: {red_light_cnt} | TRIPLE: {triple_cnt}")

                        # Live visual preview update (Downscaled WebSocket transfer for max fluidity)
                        if frame_idx % frame_skip == 0:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            disp_w = 540 if preview_size == "Compact (540px)" else (720 if preview_size == "Standard (720px)" else 854)
                            disp_h = int(disp_w * frame.shape[0] / max(1, frame.shape[1]))
                            small_rgb = cv2.resize(rgb_frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                            preview_container.image(small_rgb, caption=f"Live Surveillance Telemetry (Frame {frame_idx})")

                    cap.release()
                    if not is_live_infinite:
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

                    status_text.success(f"Processing session concluded. Telemetry logged to {new_session_name}.")

                    # Calculate fine recovery
                    total_fines = 0
                    for _, vrow in viol_df.iterrows():
                        vtype = vrow.get("violation_type", "")
                        fine = PENAL_CODE_DIRECTORY.get(vtype, {}).get("fine_inr", 1000)
                        total_fines += fine

                    # Summary Metrics
                    st.markdown("### Detection & Violation Summary")
                    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
                    sm1.metric("PROCESSED FRAMES", len(out_df))
                    sm2.metric("PEAK DENSITY", int(out_df["vehicle_count"].max()))
                    sm3.metric("NO-HELMET COUNT", len(helmet_detector.logged_violations) if helmet_detector else 0)
                    sm4.metric("RED-LIGHT COUNT", len(red_light_detector.logged_violations) if red_light_detector else 0)
                    sm5.metric("FINE POTENTIAL", f"INR {total_fines:,}")

                    # Violation Evidence Log Table & E-Challan Inspector
                    if not viol_df.empty:
                        st.markdown("#### Violation Citations & Dispatch Queue")
                        st.dataframe(viol_df, width="stretch")

                        c_down1, c_down2 = st.columns([1, 1])
                        with c_down1:
                            st.download_button(
                                "Download Violation Evidence CSV",
                                viol_df.to_csv(index=False),
                                file_name=f"{new_session_name}_violations.csv",
                                mime="text/csv"
                            )
                        with c_down2:
                            st.markdown(f"**Total Legal Citations Logged:** `{len(viol_df)} Citations` (INR {total_fines:,})")

                        # Interactive E-Challan Ticket Viewer
                        st.markdown("##### Digital E-Challan Ticket Inspector")
                        violation_options = [f"#{row['track_id']} - {row['violation_type']} (Frame {row['frame_number']})" for _, row in viol_df.iterrows()]
                        selected_challan_idx = st.selectbox("Select Citation to Inspect:", range(len(violation_options)), format_func=lambda i: violation_options[i])
                        
                        selected_viol = viol_df.iloc[selected_challan_idx].to_dict()
                        challan_doc = create_echallan_record(selected_viol)
                        ticket_html = render_echallan_html(challan_doc)
                        
                        st.components.v1.html(ticket_html, height=380, scrolling=True)

                    else:
                        st.info("Zero traffic safety violations logged in this observation session.")

                    # Plot results
                    st.markdown("##### Vehicle Class Flow Breakdown")
                    fig_res = px.line(
                        out_df,
                        x="timestamp_seconds",
                        y=["cars", "motorcycles", "buses", "trucks"],
                        title="Vehicle Flow Dynamics over Session Timeline",
                        labels={"timestamp_seconds": "Elapsed Time (Seconds)", "value": "Count", "variable": "Category"},
                        color_discrete_map={"cars": "#fafafa", "motorcycles": "#a1a1aa", "buses": "#71717a", "trucks": "#52525b"}
                    )
                    fig_res.update_layout(
                        paper_bgcolor="#18181b",
                        plot_bgcolor="#18181b",
                        font={"family": "Inter", "color": "#fafafa"},
                        xaxis=dict(gridcolor="#27272a"),
                        yaxis=dict(gridcolor="#27272a")
                    )
                    st.plotly_chart(fig_res, width="stretch")

                except Exception as e:
                    status_text.error(f"Error during video processing: {e}")

    # ==========================================================================
    # TAB 2: VIOLATION ENFORCEMENT & MODEL TRAINER HUB
    # ==========================================================================
    with tab_violations:
        st.markdown("#### Safety Model Training Engine")
        st.caption("Fine-tune custom YOLO architectures on local datasets for intersection safety enforcement.")

        t_col1, t_col2 = st.columns([1, 1])

        with t_col1:
            st.markdown("##### Model Architecture Parameters")
            train_task = st.selectbox("TARGET ENFORCEMENT TASK", ["Helmet / No-Helmet Detection", "Traffic Signal Phase Detection"])
            task_key = "helmet" if "Helmet" in train_task else "traffic_light"
            base_model = st.selectbox("BACKBONE MODEL", ["yolo11n.pt (Fastest, Edge-Deployable)", "yolo11s.pt (Balanced, High Accuracy)", "yolo11m.pt (Heavyweight)"])
            base_model_file = base_model.split(" ")[0]

            epochs = st.slider("TRAINING EPOCHS", 5, 100, 30, step=5)
            batch_size = st.selectbox("BATCH SIZE", [8, 16, 32, 64], index=1)

        with t_col2:
            st.markdown("##### Dataset Configuration")
            dataset_path = st.text_input("DATASET YAML PATH", value=f"data/{task_key}_dataset.yaml")
            output_dir = st.text_input("OUTPUT WEIGHTS DIRECTORY", value="models/violation_models")

            st.markdown(f"""
            <div class="telemetry-card">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">TERMINAL TRAINING COMMAND</div>
                <pre style="background: #09090b; color: #fafafa; border: 1px solid #27272a; padding: 10px; border-radius: 3px; font-size: 12px; margin-top: 8px; overflow-x: auto;">python -m vision.train_violation_model \\
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
        st.markdown("#### IISc AIM Indian Traffic Benchmark Datasets")
        st.caption("AI for Integrated Mobility (AIM) Lab @ Indian Institute of Science (IISc), Bengaluru.")

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("""
            <div class="telemetry-card" style="min-height: 180px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">DATASET BENCHMARK 01</div>
                <div style="font-size: 16px; font-weight: 700; color: #fafafa; margin: 4px 0 8px 0;">IISc UVH-26 Benchmark</div>
                <div style="font-size: 12px; color: #a1a1aa; margin-bottom: 8px;">Urban Vehicle Heterogeneity (2025)</div>
                <ul style="font-size: 13px; color: #d4d4d8; margin: 0; padding-left: 18px; line-height: 1.6;">
                    <li>26,646 1080p CCTV Images</li>
                    <li>1.8 Million Bounding Boxes</li>
                    <li>26 Heterogeneous Indian Vehicle Categories</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with d_col2:
            st.markdown("""
            <div class="telemetry-card" style="min-height: 180px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a1a1aa; text-transform: uppercase;">DATASET BENCHMARK 02</div>
                <div style="font-size: 16px; font-weight: 700; color: #fafafa; margin: 4px 0 8px 0;">IISc BMD-45 Benchmark</div>
                <div style="font-size: 12px; color: #a1a1aa; margin-bottom: 8px;">Bangalore Mobility Dynamics</div>
                <ul style="font-size: 13px; color: #d4d4d8; margin: 0; padding-left: 18px; line-height: 1.6;">
                    <li>45 Urban Municipal Sectors</li>
                    <li>Multi-Camera Trajectory Fusion</li>
                    <li>Kinematic Road-Plane Velocity Truth</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
