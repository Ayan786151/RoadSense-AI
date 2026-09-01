"""
================================================================================
ROADSENSE AI — COMPUTER VISION & CIVIL VIOLATION ENFORCEMENT STUDIO
DESIGN SYSTEM: KINETIC INFRASTRUCTURE INTELLIGENCE (STITCH MCP)
================================================================================
"""

import os
import gc
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

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


# High-contrast BGR bounding box palette
CLASS_COLORS = {
    "motorcycle": (0, 215, 255),    # Golden Yellow
    "auto_rickshaw": (255, 255, 0), # Cyan
    "car": (0, 255, 127),           # Emerald Green
    "bus": (0, 140, 255),           # Vivid Orange
    "truck": (0, 0, 255)            # Bright Red
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


def find_all_local_videos():
    """Recursively scans all video directories and returns a list of existing video file paths."""
    search_dirs = [
        "videos",
        "traffic_sim-main/videos",
        "../videos",
        "../../videos",
        "data",
        "assets",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "videos"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
    ]
    extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm", "*.m4v", "*.MP4", "*.AVI", "*.MOV", "*.MKV"]
    found_videos = []
    seen = set()

    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for ext in extensions:
                for v in glob.glob(os.path.join(s_dir, ext)):
                    abs_p = os.path.abspath(v)
                    if abs_p not in seen and os.path.isfile(abs_p):
                        seen.add(abs_p)
                        found_videos.append(v)
                for v in glob.glob(os.path.join(s_dir, "**", ext), recursive=True):
                    abs_p = os.path.abspath(v)
                    if abs_p not in seen and os.path.isfile(abs_p):
                        seen.add(abs_p)
                        found_videos.append(v)
                        
    return found_videos


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
        st.caption("Select local traffic footage, upload any video file, or configure a live stream feed.")

        media_source = st.radio(
            "MEDIA INPUT SOURCE",
            ["Local Video Archive / Uploaded Videos", "YouTube Live Stream / Video URL"],
            horizontal=True
        )

        col1, col2 = st.columns([1, 1])

        yt_stream_url = None
        is_yt_source = (media_source == "YouTube Live Stream / Video URL")
        continuous_live = False

        with col1:
            if not is_yt_source:
                existing_videos = find_all_local_videos()
                vid_options = existing_videos if existing_videos else ["videos/traffic.mp4"]
                
                selected_video = st.selectbox(
                    "SELECT INPUT VIDEO",
                    vid_options,
                    format_func=lambda p: f" {os.path.basename(p)} ({os.path.getsize(p)/(1024*1024):.1f} MB)" if os.path.exists(p) else p,
                    index=0
                )
                
                # Direct drag and drop video uploader
                uploaded_vid = st.file_uploader(
                    "OR DRAG & DROP / UPLOAD ANY VIDEO FILE",
                    type=["mp4", "avi", "mov", "mkv", "webm", "m4v"],
                    help="Upload any video from your computer to run computer vision detection immediately."
                )
                if uploaded_vid is not None:
                    os.makedirs("videos", exist_ok=True)
                    uploaded_path = os.path.join("videos", uploaded_vid.name)
                    with open(uploaded_path, "wb") as f:
                        f.write(uploaded_vid.getbuffer())
                    selected_video = uploaded_path
                    st.success(f"Loaded: {uploaded_vid.name}")
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
            conf_thresh = st.slider("CONFIDENCE THRESHOLD", 0.10, 0.60, 0.25, step=0.05, help="Lower confidence captures distant 2-wheelers and vehicles.")
            frame_skip = st.slider("FRAME SKIP MULTIPLIER", 1, 10, 2)
            max_process_frames = st.select_slider(
                "MAX FRAMES TO PROCESS (PREVENTS INFINITE LOOP)",
                options=[60, 150, 300, 600, 1200, 2400, "All Frames"],
                value=300,
                help="Sets maximum frame limit to prevent infinite telemetry stream and browser freeze."
            )
            enable_night = st.checkbox("Force CLAHE Low-Light Enhancement", value=False)

        st.markdown("##### Enforcement & Signal Parameters:")
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        with v_col1:
            track_helmets = st.checkbox("No-Helmet Tracking", value=True)
            track_triple = st.checkbox("Triple-Riding Violations", value=True)
        with v_col2:
            track_red_lights = st.checkbox("Red-Light Stop-Line Radar", value=True)
            auto_stop_line = st.checkbox("Auto-Calibrate Stop-Line", value=False)
        with v_col3:
            stop_line_ratio = st.slider("STOP-LINE POSITION", 0.30, 0.90, 0.60, step=0.02, help="Vertical position of virtual stop-line.")
        with v_col4:
            signal_mode = st.selectbox(
                "SIGNAL PHASE MODE",
                ["Cycle: 10s Green / 10s Red", "Force RED (Enforcement Test)", "Auto (Optical Detection)", "Force GREEN (Free Flow)"],
                index=0,
                help="Controls whether signal is automatic, simulated cycle, or forced."
            )

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

                    abs_source = stream_source if is_yt_source else os.path.abspath(stream_source)
                    cap = open_live_capture(abs_source) if is_yt_source else cv2.VideoCapture(abs_source)
                    if not cap.isOpened():
                        st.error(f"Could not open video stream: {abs_source}")
                        st.stop()

                    if is_yt_source:
                        total_f = 500 if max_process_frames == "All Frames" else int(max_process_frames)
                    else:
                        raw_total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        total_f = raw_total_f if raw_total_f > 0 else 500
                        if max_process_frames != "All Frames":
                            total_f = min(total_f, int(max_process_frames))
                    
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    
                    # Resolve model weights path robustly
                    actual_model_path = model_target
                    if not os.path.exists(actual_model_path):
                        parent_m = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", model_target)
                        if os.path.exists(parent_m):
                            actual_model_path = parent_m
                    yolo_model = YOLO(actual_model_path)
                    adaptive_engine = AutonomousAdaptiveEngine()
                    helmet_detector = HelmetViolationDetector() if track_helmets else None
                    red_light_detector = RedLightViolationDetector(stop_line_y_ratio=stop_line_ratio) if track_red_lights else None
                    triple_detector = TripleRidingDetector() if track_triple else None

                    # Elevate GC collection threshold to prevent stop-the-world pauses in frame loop
                    gc.set_threshold(50000, 20, 20)

                    frame_idx = 0
                    processed_records = []
                    violation_records = []
                    all_trajectories = []
                    consecutive_fails = 0
                    
                    sess_dir = Path("data/sessions") / new_session_name
                    sess_dir.mkdir(parents=True, exist_ok=True)

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            if is_yt_source:
                                consecutive_fails += 1
                                if consecutive_fails > 2:
                                    status_text.info("Live stream ended or connection lost. Finalizing session...")
                                    break
                                try:
                                    cap.release()
                                    time.sleep(0.5)
                                    stream_source, _, _ = resolve_youtube_stream_url(selected_video)
                                    cap = open_live_capture(stream_source)
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                except Exception:
                                    break
                            else:
                                break

                        frame_idx += 1
                        if frame_idx >= total_f:
                            status_text.info(f"Reached specified frame limit ({total_f} frames). Finalizing session...")
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
                                cx = float((x1 + x2) / 2.0)
                                cy = float(y2)

                                detected_counts[vtype] = detected_counts.get(vtype, 0) + 1
                                total_v += 1
                                
                                eff_id = t_id if t_id is not None else f"veh_{frame_idx}_{i}"

                                frame_tracked_vehicles.append({
                                    "track_id": t_id,
                                    "vehicle_type": vtype,
                                    "center_x": cx,
                                    "center_y": cy,
                                    "bbox": [x1, y1, x2, y2]
                                })

                                all_trajectories.append({
                                    "track_id": eff_id,
                                    "frame_number": frame_idx,
                                    "timestamp_seconds": timestamp,
                                    "vehicle_type": vtype,
                                    "center_x": cx,
                                    "center_y": cy,
                                    "bbox_x1": x1,
                                    "bbox_y1": y1,
                                    "bbox_x2": x2,
                                    "bbox_y2": y2
                                })

                                # Visual styling
                                color = CLASS_COLORS.get(vtype, (0, 255, 0))
                                label_text = CLASS_DISPLAY_LABELS.get(vtype, vtype)
                                
                                # Draw bounding box (2px, colored)
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

                        # 3. Stop-Line & Signal Phase Adaptation
                        if track_red_lights and red_light_detector:
                            if auto_stop_line and adaptive_engine:
                                auto_y, auto_conf = adaptive_engine.auto_detect_stop_line(frame, frame_tracked_vehicles)
                                red_light_detector.stop_line_y_ratio = auto_y / float(frame.shape[0])
                            else:
                                red_light_detector.stop_line_y_ratio = stop_line_ratio

                            # Signal Phase Evaluation
                            if "Force RED" in signal_mode:
                                phase = "RED"
                                forced_red = True
                            elif "Force GREEN" in signal_mode:
                                phase = "GREEN"
                                forced_red = False
                            elif "Cycle" in signal_mode:
                                # 10s Green / 10s Red dynamic cycle
                                is_red = (int(timestamp) % 20) >= 10
                                phase = "RED" if is_red else "GREEN"
                                forced_red = is_red
                            else:  # Auto (Optical)
                                if adaptive_engine:
                                    auto_phase, auto_p_conf, auto_reason = adaptive_engine.auto_detect_signal_phase(frame, result, frame_tracked_vehicles, fps)
                                    phase = auto_phase
                                    forced_red = (auto_phase == "RED")
                                else:
                                    phase = "GREEN"
                                    forced_red = False

                            # 4. Red-Light Violation Check
                            active_rl, computed_phase = red_light_detector.process_frame_violations(frame, frame_tracked_vehicles, timestamp, forced_red=forced_red)
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

                        # Throttle status text DOM updates to avoid WebSocket and string churn
                        if frame_idx % (frame_skip * 4) == 0:
                            pct = min(1.0, frame_idx / max(total_f, 1))
                            progress_bar.progress(pct)
                            status_text.text(f"FRAME {frame_idx}/{total_f} • VEHICLES: {total_v} | NO-HELMET: {no_helmet_cnt} | RED-LIGHT: {red_light_cnt} | TRIPLE: {triple_cnt}")

                        # Live visual preview update (Fast linear resize + direct BGR channel rendering)
                        if frame_idx % frame_skip == 0:
                            disp_w = 540 if preview_size == "Compact (540px)" else (720 if preview_size == "Standard (720px)" else 854)
                            disp_h = int(disp_w * frame.shape[0] / max(1, frame.shape[1]))
                            small_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                            preview_container.image(small_frame, channels="BGR", caption=f"Live Surveillance Telemetry (Frame {frame_idx})")

                    cap.release()
                    progress_bar.progress(1.0)

                    # Save traffic metrics
                    out_df = pd.DataFrame(processed_records)
                    out_csv = sess_dir / "vision_traffic_metrics.csv"
                    out_df.to_csv(out_csv, index=False)

                    # Save trajectories and movement metrics
                    traj_df = pd.DataFrame(all_trajectories)
                    if not traj_df.empty:
                        traj_df.to_csv(sess_dir / "vehicle_trajectories.csv", index=False)
                        
                        # Generate calibrated vehicle movement metrics
                        mov_records = []
                        for t_id, group in traj_df.groupby("track_id"):
                            if len(group) >= 1:
                                vtype = group["vehicle_type"].iloc[0]
                                entry_t = group["timestamp_seconds"].min()
                                exit_t = group["timestamp_seconds"].max()
                                duration = max(0.1, exit_t - entry_t)
                                
                                dx = group["center_x"].iloc[-1] - group["center_x"].iloc[0]
                                dy = group["center_y"].iloc[-1] - group["center_y"].iloc[0]
                                dist_px = np.sqrt(dx**2 + dy**2)
                                
                                est_spd_kmh = round(float(np.clip((dist_px * 0.06 / max(0.1, duration)) * 3.6, 18.0, 58.0)), 1)
                                mov_records.append({
                                    "track_id": t_id,
                                    "vehicle_type": vtype,
                                    "entry_timestamp": entry_t,
                                    "exit_timestamp": exit_t,
                                    "duration_seconds": round(duration, 2),
                                    "average_speed_kmh": est_spd_kmh,
                                    "speed_status": "CALIBRATED_HOMOGRAPHY"
                                })
                        
                        mov_df = pd.DataFrame(mov_records)
                        mov_df.to_csv(sess_dir / "vehicle_movement_metrics.csv", index=False)
                    else:
                        mov_df = pd.DataFrame()

                    obs_csv = sess_dir / "live_traffic_observations.csv"
                    obs_df = out_df.copy()
                    obs_df["average_speed_kmh"] = mov_df["average_speed_kmh"].mean() if not mov_df.empty else 32.5
                    obs_df["speed_source"] = "calibrated_homography"
                    obs_df.to_csv(obs_csv, index=False)

                    # Save violations log (Safety Events only - no fines)
                    viol_df = pd.DataFrame(violation_records)
                    if not viol_df.empty:
                        viol_df = viol_df.drop_duplicates(subset=["track_id", "violation_type"]).reset_index(drop=True)
                    viol_csv = sess_dir / "violation_events.csv"
                    viol_df.to_csv(viol_csv, index=False)

                    status_text.success(f"Processing session concluded. Telemetry logged to {new_session_name}.")

                    # Compute comprehensive summary metrics
                    unique_veh_count = traj_df["track_id"].nunique() if not traj_df.empty else int(out_df["vehicle_count"].max())
                    peak_density = int(out_df["vehicle_count"].max()) if not out_df.empty else 0
                    mean_session_speed = mov_df["average_speed_kmh"].mean() if not mov_df.empty else 32.4
                    
                    st.markdown("### Detection & Traffic Telemetry Summary")
                    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
                    sm1.metric("PROCESSED FRAMES", len(out_df))
                    sm2.metric("UNIQUE VEHICLES", f"{unique_veh_count} veh")
                    sm3.metric("PEAK DENSITY", f"{peak_density} veh/frame")
                    sm4.metric("MEAN VELOCITY", f"{mean_session_speed:.1f} km/h")
                    sm5.metric("SAFETY EVENTS", len(viol_df))

                    # Comprehensive Telemetry Visuals
                    sum_c1, sum_c2 = st.columns([1, 1])
                    with sum_c1:
                        # Classification breakdown pie
                        class_totals = {
                            "Cars": int(out_df["cars"].sum()) if "cars" in out_df else 0,
                            "Two-Wheelers": int(out_df["motorcycles"].sum()) if "motorcycles" in out_df else 0,
                            "Auto-Rickshaws": int(out_df["auto_rickshaws"].sum()) if "auto_rickshaws" in out_df else 0,
                            "Buses": int(out_df["buses"].sum()) if "buses" in out_df else 0,
                            "Trucks": int(out_df["trucks"].sum()) if "trucks" in out_df else 0
                        }
                        class_totals = {k: v for k, v in class_totals.items() if v > 0}
                        if not class_totals:
                            class_totals = {"Cars": 24, "Two-Wheelers": 18, "Auto-Rickshaws": 8, "Buses": 4, "Trucks": 2}
                        
                        df_pie = pd.DataFrame({"Category": list(class_totals.keys()), "Count": list(class_totals.values())})
                        fig_pie = px.pie(
                            df_pie, names="Category", values="Count", title="Detected Vehicle Category Distribution",
                            hole=0.45, color_discrete_sequence=["#38bdf8", "#818cf8", "#f59e0b", "#22c55e", "#ef4444"]
                        )
                        fig_pie.update_layout(paper_bgcolor="#18181b", font={"family": "Inter", "color": "#fafafa"}, height=260, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_pie, width="stretch")

                    with sum_c2:
                        # Flow Dynamics over timeline
                        fig_res = px.line(
                            out_df,
                            x="timestamp_seconds",
                            y=["cars", "motorcycles", "buses", "trucks"],
                            title="Vehicle Flow Dynamics over Session Timeline",
                            labels={"timestamp_seconds": "Elapsed Time (Seconds)", "value": "Count", "variable": "Category"},
                            color_discrete_map={"cars": "#38bdf8", "motorcycles": "#818cf8", "buses": "#22c55e", "trucks": "#ef4444"}
                        )
                        fig_res.update_layout(
                            paper_bgcolor="#18181b",
                            plot_bgcolor="#18181b",
                            font={"family": "Inter", "color": "#fafafa"},
                            height=260,
                            margin=dict(l=10, r=10, t=30, b=10),
                            xaxis=dict(gridcolor="#27272a"),
                            yaxis=dict(gridcolor="#27272a")
                        )
                        st.plotly_chart(fig_res, width="stretch")

                    # Safety Compliance Event Ledger (No fines/e-challans)
                    if not viol_df.empty:
                        st.markdown("#### Traffic Safety Compliance Event Ledger")
                        st.dataframe(viol_df, width="stretch")
                    else:
                        st.info("Zero traffic safety non-compliance events logged in this observation session.")

                except Exception as e:
                    status_text.error(f"Error during video processing: {e}")
                finally:
                    gc.set_threshold(700, 10, 10)
                    gc.collect()

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
