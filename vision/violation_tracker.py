"""
================================================================================
ROAD SENSE AI - UNIFIED VIOLATION TRACKER & EVIDENCE ORCHESTRATOR
MODULE: COMPREHENSIVE TRAFFIC VIOLATION DETECTION PIPELINE
================================================================================

Orchestrates multi-modal violation tracking:
1. Two-Wheeler No-Helmet Violations
2. Red-Light Signal Running & Stop-Line Breaches
3. Exports standardized violation evidence logs (violation_events.csv)
================================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

try:
    from vision.helmet_detector import HelmetViolationDetector
    from vision.red_light_detector import RedLightViolationDetector
    from vision.vehicle_detector import resolve_vehicle_class, resolve_output_paths
    from vision.enhancement import adaptive_preprocess_frame
except ImportError:
    from helmet_detector import HelmetViolationDetector
    from red_light_detector import RedLightViolationDetector
    from vehicle_detector import resolve_vehicle_class, resolve_output_paths
    from enhancement import adaptive_preprocess_frame


def process_video_violations(
    video_path: str = "videos/traffic.mp4",
    model_name: str = "yolo11s.pt",
    session_id: Optional[str] = None,
    track_helmets: bool = True,
    track_red_lights: bool = True,
    stop_line_ratio: float = 0.65,
    forced_red: bool = False,
    show_preview: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes full traffic video processing with dual violation tracking.
    """
    metrics_output, trajectory_output, _, session_dir = resolve_output_paths(session_id)
    violation_output = str(Path(session_dir) / "violation_events.csv")

    print("=" * 70)
    print("ROAD SENSE AI - AI VIOLATION DETECTION ENGINE")
    print("NO-HELMET TRACKING + RED-LIGHT BREAKING DETECTION")
    print("=" * 70)
    print(f"Source Video     : {video_path}")
    print(f"Helmet Tracking  : {'ENABLED' if track_helmets else 'DISABLED'}")
    print(f"Red-Light Camera : {'ENABLED' if track_red_lights else 'DISABLED'}")
    print(f"Stop-Line Ratio  : {stop_line_ratio:.2f}")
    print(f"Target Output    : {violation_output}")
    print("=" * 70)

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Initialize models
    model = YOLO(model_name)
    helmet_detector = HelmetViolationDetector() if track_helmets else None
    red_light_detector = RedLightViolationDetector(stop_line_y_ratio=stop_line_ratio) if track_red_lights else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_idx = 0
    PROCESS_EVERY_N_FRAMES = 5
    all_violations = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            continue

        timestamp = round(frame_idx / fps, 3)

        # Preprocess lighting
        proc_frame, _ = adaptive_preprocess_frame(frame)

        # Vehicle tracking
        results = model.track(
            proc_frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.35,
            imgsz=640,
            verbose=False
        )

        res = results[0]
        tracked_vehicles = []

        if res.boxes is not None:
            boxes = res.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                vtype = resolve_vehicle_class(cls_id, model.names)
                if not vtype:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                t_id = int(boxes.id[i].item()) if boxes.id is not None else None
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2.0
                cy = float(y2)

                veh_dict = {
                    "track_id": t_id,
                    "vehicle_type": vtype,
                    "center_x": cx,
                    "center_y": cy,
                    "bbox": [x1, y1, x2, y2]
                }
                tracked_vehicles.append(veh_dict)

                # 1. Helmet Compliance Check on Two-Wheelers
                if track_helmets and helmet_detector and vtype == "motorcycle":
                    h_res = helmet_detector.analyze_motorcycle_rider(frame, xyxy, t_id, timestamp)
                    if show_preview:
                        helmet_detector.draw_annotation(frame, h_res)
                    if not h_res["has_helmet"] and t_id is not None:
                        all_violations.append({
                            "timestamp_seconds": timestamp,
                            "frame_number": frame_idx,
                            "track_id": t_id,
                            "vehicle_type": "motorcycle",
                            "violation_type": "NO_HELMET",
                            "severity": "HIGH",
                            "confidence": h_res["confidence"],
                            "details": h_res["reason"]
                        })

        # 2. Red-Light Violation Check
        if track_red_lights and red_light_detector:
            active_rl, phase = red_light_detector.process_frame_violations(
                frame,
                tracked_vehicles,
                timestamp,
                forced_red=forced_red
            )
            for viol in active_rl:
                all_violations.append({
                    "timestamp_seconds": timestamp,
                    "frame_number": frame_idx,
                    "track_id": viol["track_id"],
                    "vehicle_type": viol["vehicle_type"],
                    "violation_type": "RED_LIGHT_RUNNING",
                    "severity": "CRITICAL",
                    "confidence": 0.95,
                    "details": f"Crossed Stop Line during RED phase (y={viol['cross_y']})"
                })
            if show_preview:
                red_light_detector.draw_annotation(frame, phase, active_rl)

        if show_preview:
            cv2.imshow("RoadSense AI - Violation Enforcement", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    # Deduplicate violations by track_id and violation_type
    viol_df = pd.DataFrame(all_violations)
    if not viol_df.empty:
        viol_df = viol_df.drop_duplicates(subset=["track_id", "violation_type"]).reset_index(drop=True)

    Path(violation_output).parent.mkdir(parents=True, exist_ok=True)
    viol_df.to_csv(violation_output, index=False)

    summary = {
        "total_violations": len(viol_df),
        "no_helmet_violations": int((viol_df["violation_type"] == "NO_HELMET").sum()) if not viol_df.empty else 0,
        "red_light_violations": int((viol_df["violation_type"] == "RED_LIGHT_RUNNING").sum()) if not viol_df.empty else 0,
        "output_file": violation_output
    }

    print("\n" + "=" * 70)
    print("VIOLATION PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total Unique Violations Logged : {summary['total_violations']}")
    print(f"  * 🪖 No-Helmet Violations    : {summary['no_helmet_violations']}")
    print(f"  * 🚦 Red-Light Violations    : {summary['red_light_violations']}")
    print(f"\nSaved violation audit report to:\n  {violation_output}")
    print("=" * 70 + "\n")

    return viol_df, summary


def main():
    parser = argparse.ArgumentParser(description="RoadSense AI - Unified Traffic Violation Enforcement Engine")
    parser.add_argument("--video", type=str, default="videos/traffic.mp4", help="Path to input video")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="YOLO model weights")
    parser.add_argument("--session", type=str, default=None, help="Session identifier (e.g. session_001)")
    parser.add_argument("--no-helmets", action="store_true", help="Disable helmet detection")
    parser.add_argument("--no-red-lights", action="store_true", help="Disable red light violation detection")
    parser.add_argument("--stop-line", type=float, default=0.65, help="Stop line vertical ratio (0.0 - 1.0)")
    parser.add_argument("--force-red", action="store_true", help="Force red light signal phase for demo")
    parser.add_argument("--show", action="store_true", help="Display live video playback window")

    args = parser.parse_args()

    process_video_violations(
        video_path=args.video,
        model_name=args.model,
        session_id=args.session,
        track_helmets=not args.no_helmets,
        track_red_lights=not args.no_red_lights,
        stop_line_ratio=args.stop_line,
        forced_red=args.force_red,
        show_preview=args.show
    )


if __name__ == "__main__":
    main()
