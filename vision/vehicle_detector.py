import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Import adaptive lighting preprocessor
try:
    from vision.enhancement import adaptive_preprocess_frame
except ImportError:
    from enhancement import adaptive_preprocess_frame


# ============================================================
# ROAD SENSE AI - VEHICLE DETECTOR + TRAJECTORY LOGGER
# ============================================================

DEFAULT_VIDEO_PATH = "videos/traffic.mp4"
DEFAULT_METRICS_OUTPUT = "data/vision_traffic_metrics.csv"
DEFAULT_TRAJECTORY_OUTPUT = "data/vehicle_trajectories.csv"

DEFAULT_MODEL_NAME = "models/iisc_yolov11s_indian_traffic.pt" if os.path.exists("models/iisc_yolov11s_indian_traffic.pt") else "yolo11s.pt"

# Class-specific confidence thresholds for precision filtering
CLASS_CONF_THRESHOLDS = {
    "motorcycle": 0.15,
    "auto_rickshaw": 0.18,
    "car": 0.20,
    "bus": 0.25,
    "truck": 0.35
}

# Unified vehicle mapping for COCO, IISc UVH-26, and Indian Traffic classes
def resolve_vehicle_class(class_id: int, model_names: dict, include_riders: bool = True) -> str:
    """
    Dynamically maps class ID to standard vehicle categories.
    Supports COCO (cars, bikes, riders, buses, trucks) and IISc UVH-26 Indian traffic classes (auto-rickshaws, tempos, etc.).
    """
    raw = str(model_names.get(class_id, "")).lower()
    
    if any(k in raw for k in ["auto", "rickshaw", "tuk", "3-wheeler", "three-wheeler", "three_wheeler", "e_rickshaw"]):
        return "auto_rickshaw"
    elif any(k in raw for k in ["motorcycle", "bike", "bicycle", "two-wheeler", "two_wheeler", "2-wheeler", "scooter"]):
        return "motorcycle"
    elif include_riders and any(k in raw for k in ["person", "rider"]):
        return "motorcycle"  # In CCTV traffic lanes, persons detected on roadway are riders on two-wheelers
    elif any(k in raw for k in ["bus", "mini-bus", "mini_bus", "van"]):
        return "bus"
    elif any(k in raw for k in ["truck", "lcv", "tempo", "tempo-traveller", "tempo_traveller", "lorry", "container", "tractor"]):
        return "truck"
    elif any(k in raw for k in ["car", "hatchback", "sedan", "suv", "muv", "taxi", "jeep"]):
        return "car"
    
    # Fallback to COCO default IDs if names missing
    coco_map = {0: "motorcycle", 1: "motorcycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    return coco_map.get(class_id, None)


# ============================================================
# DRIVER + PILLION RIDER FUSION
# ============================================================

def fuse_driver_pillion_riders(
    candidate_boxes: List[np.ndarray],
    candidate_classes: List[str],
    candidate_confs: List[float],
    candidate_ids: List[Optional[int]]
) -> Tuple[List[np.ndarray], List[str], List[float], List[Optional[int]]]:
    """
    Fuses vertically stacked driver + pillion passenger bounding boxes into a single unified motorcycle unit.
    Prevents counting two bikes when two people are riding the same physical motorcycle.
    """
    tw_indices = [i for i, c in enumerate(candidate_classes) if c == "motorcycle"]
    other_indices = [i for i, c in enumerate(candidate_classes) if c != "motorcycle"]
    
    if len(tw_indices) <= 1:
        return candidate_boxes, candidate_classes, candidate_confs, candidate_ids

    merged_tw_boxes = []
    merged_tw_confs = []
    merged_tw_ids = []
    used = [False] * len(tw_indices)

    for i, idx1 in enumerate(tw_indices):
        if used[i]:
            continue
        b1 = candidate_boxes[idx1]
        union_box = list(b1)
        max_conf = candidate_confs[idx1]
        best_id = candidate_ids[idx1]
        used[i] = True

        w1 = b1[2] - b1[0]
        h1 = b1[3] - b1[1]
        cx1 = (b1[0] + b1[2]) / 2.0

        for j in range(i + 1, len(tw_indices)):
            if used[j]:
                continue
            idx2 = tw_indices[j]
            b2 = candidate_boxes[idx2]
            w2 = b2[2] - b2[0]
            h2 = b2[3] - b2[1]
            cx2 = (b2[0] + b2[2]) / 2.0

            x_dist = abs(cx1 - cx2)
            y_gap = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))

            # Driver and pillion are in the same lane and vertically stacked
            if x_dist < max(w1, w2) * 0.75 and y_gap < max(h1, h2) * 0.60:
                union_box = [
                    min(union_box[0], b2[0]),
                    min(union_box[1], b2[1]),
                    max(union_box[2], b2[2]),
                    max(union_box[3], b2[3])
                ]
                max_conf = max(max_conf, candidate_confs[idx2])
                if best_id is None and candidate_ids[idx2] is not None:
                    best_id = candidate_ids[idx2]
                used[j] = True

        merged_tw_boxes.append(np.array(union_box, dtype=int))
        merged_tw_confs.append(max_conf)
        merged_tw_ids.append(best_id)

    final_boxes = [candidate_boxes[i] for i in other_indices] + merged_tw_boxes
    final_classes = [candidate_classes[i] for i in other_indices] + ["motorcycle"] * len(merged_tw_boxes)
    final_confs = [candidate_confs[i] for i in other_indices] + merged_tw_confs
    final_ids = [candidate_ids[i] for i in other_indices] + merged_tw_ids

    return final_boxes, final_classes, final_confs, final_ids


# ============================================================
# RESOLVE SESSION PATHS
# ============================================================

def resolve_output_paths(session_id: str = None):
    """
    Resolves metrics and trajectory output paths based on session_id.
    """
    if session_id:
        session_dir = Path("data") / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = str(session_dir / "vision_traffic_metrics.csv")
        trajectory_path = str(session_dir / "vehicle_trajectories.csv")
        metadata_path = str(session_dir / "session_metadata.json")
        return metrics_path, trajectory_path, metadata_path, str(session_dir)
    return DEFAULT_METRICS_OUTPUT, DEFAULT_TRAJECTORY_OUTPUT, None, "data"


# ============================================================
# OCCLUSION OVERLAP DETECTOR
# ============================================================

def compute_occlusion_flags(boxes_xyxy: List[np.ndarray], threshold: float = 0.15) -> List[bool]:
    """
    Detects if any vehicle bounding box is occluded from the camera's bottom perspective
    by another vehicle positioned lower down in the frame (closer to the camera).

    When a vehicle is behind another in heavy traffic, the front vehicle occludes
    the bottom contact point of the rear vehicle.
    """
    n = len(boxes_xyxy)
    if n <= 1:
        return [False] * n

    is_occluded = [False] * n

    for i in range(n):
        box_a = boxes_xyxy[i]  # [x1, y1, x2, y2]
        area_a = max((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]), 1)

        for j in range(n):
            if i == j:
                continue
            box_b = boxes_xyxy[j]

            # Check if box_b is in front (lower in the image -> box_b[3] > box_a[3])
            if box_b[3] > box_a[1]:
                # Compute intersection
                ix1 = max(box_a[0], box_b[0])
                iy1 = max(box_a[1], box_b[1])
                ix2 = min(box_a[2], box_b[2])
                iy2 = min(box_a[3], box_b[3])

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    overlap_ratio = inter_area / float(area_a)
                    if overlap_ratio >= threshold:
                        is_occluded[i] = True
                        break

    return is_occluded


# ============================================================
# MAIN VIDEO PROCESSING
# ============================================================

def process_video(
    video_path: str = DEFAULT_VIDEO_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    session_id: str = None,
    location_id: str = "loc_01",
    camera_id: str = "cam_01",
    observation_date: str = None,
    show_preview: bool = False,
    force_night_mode: bool = False
):
    metrics_output, trajectory_output, metadata_output, output_dir = resolve_output_paths(session_id)

    if observation_date is None:
        observation_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("=" * 70)
    print("ROAD SENSE AI - COMPUTER VISION MODULE")
    print("VEHICLE DETECTION + TRAJECTORY TRACKING + OCCLUSION & LIGHTING ADAPTATION")
    print("=" * 70)
    if session_id:
        print(f"Active Session: {session_id}")
    print(f"Source Video  : {video_path}")
    print(f"Model Path    : {model_name}")
    print(f"Target Output : {output_dir}")

    # --------------------------------------------------------
    # Check video
    # --------------------------------------------------------

    if not Path(video_path).exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}\n"
            "Please provide a valid video path."
        )

    # --------------------------------------------------------
    # Load model (Fresh instance ensures tracker isolation)
    # --------------------------------------------------------

    print(f"\nLoading YOLO model ({model_name})...")
    model = YOLO(model_name)
    print("YOLO model loaded with fresh tracker state.")

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open the traffic video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0.0

    print("\nVideo information:")
    print(f"  Resolution: {frame_width} x {frame_height}")
    print(f"  FPS:        {fps:.2f}")
    print(f"  Frames:     {total_frames}")
    print(f"  Duration:   {duration:.2f} seconds")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    metric_records = []
    trajectory_records = []
    lighting_conditions_observed = set()

    frame_number = 0
    PROCESS_EVERY_N_FRAMES = 5

    print("\nStarting vehicle detection...")
    print("Trajectory logging: ENABLED (Bottom-Center Road Contact)")
    print("Adaptive Low-Light / Night Enhancement: ACTIVE")
    print("Heavy Traffic Occlusion Detection: ACTIVE")
    if show_preview:
        print("Live preview: ENABLED (Press Q in window to stop)\n")
    else:
        print("Live preview: DISABLED (Headless high-speed mode)\n")

    # --------------------------------------------------------
    # Video loop
    # --------------------------------------------------------

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1
        if frame_number % PROCESS_EVERY_N_FRAMES != 0:
            continue

        timestamp = frame_number / fps

        # ----------------------------------------------------
        # 1. Adaptive Night / Low-Light Preprocessor
        # ----------------------------------------------------
        processed_frame, light_audit = adaptive_preprocess_frame(frame, force_enhancement=force_night_mode)
        lighting_conditions_observed.add(light_audit["lighting_condition"])

        # ----------------------------------------------------
        # 2. YOLO tracking (persist=True maintains IDs within this video)
        # ----------------------------------------------------
        results = model.track(
            processed_frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.13,
            imgsz=960,
            agnostic_nms=True,
            iou=0.48,
            verbose=False
        )

        result = results[0]

        counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
            "auto_rickshaw": 0
        }
        total_vehicles = 0

        # ----------------------------------------------------
        # 3. Process detections & compute occlusion overlap
        # ----------------------------------------------------
        cand_boxes = []
        cand_classes = []
        cand_confs = []
        cand_ids = []

        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                conf_val = float(boxes.conf[i].item()) if boxes.conf is not None else 0.5
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Roadway Horizon Filter: Ignore non-road objects above road plane (sky, billboards, gantries)
                if y2 < int(frame_height * 0.22):
                    continue

                vtype = resolve_vehicle_class(class_id, model.names, include_riders=True)
                if not vtype:
                    continue

                # Apply class-specific confidence filtering
                min_conf = CLASS_CONF_THRESHOLDS.get(vtype, 0.18)
                if conf_val < min_conf:
                    if vtype == "truck" and conf_val >= CLASS_CONF_THRESHOLDS["car"]:
                        vtype = "car"
                    else:
                        continue

                # Geometric check: Reclassify small false-positive trucks to cars
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                if vtype == "truck" and (bbox_w * bbox_h) < (frame_width * frame_height * 0.015) and conf_val < 0.55:
                    vtype = "car"

                cand_boxes.append(xyxy)
                cand_classes.append(vtype)
                cand_confs.append(conf_val)
                track_id = int(boxes.id[i].item()) if boxes.id is not None else None
                cand_ids.append(track_id)

        # Fuse driver + pillion passengers riding the same motorcycle
        valid_boxes, valid_classes, valid_confs, valid_track_ids = fuse_driver_pillion_riders(
            cand_boxes, cand_classes, cand_confs, cand_ids
        )

        for vtype in valid_classes:
            counts[vtype] = counts.get(vtype, 0) + 1
            total_vehicles += 1

        # Compute occlusion flags for all vehicles in this frame
        occlusion_flags = compute_occlusion_flags(valid_boxes, threshold=0.15) if valid_boxes else []

        for i in range(len(valid_boxes)):
            xyxy = valid_boxes[i]
            x1, y1, x2, y2 = xyxy
            vehicle_type = valid_classes[i]
            track_id = valid_track_ids[i]
            is_occ = occlusion_flags[i] if i < len(occlusion_flags) else False

            # Tracking Point: Bottom-center of the bounding box
            # Approximates the vehicle's contact point with the road plane
            center_x = (x1 + x2) / 2.0
            center_y = float(y2)

            if track_id is not None:
                trajectory_records.append({
                    "timestamp_seconds": round(timestamp, 3),
                    "frame_number": frame_number,
                    "track_id": track_id,
                    "vehicle_type": vehicle_type,
                    "center_x": round(center_x, 2),
                    "center_y": round(center_y, 2),
                    "bbox_width": int(x2 - x1),
                    "bbox_height": int(y2 - y1),
                    "is_occluded": bool(is_occ)
                })

            # Visual annotation if preview enabled
            if show_preview:
                box_col = (0, 165, 255) if is_occ else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)
                label = f"{vehicle_type} ID:{track_id}" if track_id is not None else vehicle_type
                if is_occ:
                    label += " [OCCLUDED]"
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 2)

        # ----------------------------------------------------
        # 4. Save frame-level metrics
        # ----------------------------------------------------
        metric_records.append({
            "timestamp_seconds": round(timestamp, 3),
            "vehicle_count": total_vehicles,
            "cars": counts.get("car", 0),
            "motorcycles": counts.get("motorcycle", 0),
            "buses": counts.get("bus", 0),
            "trucks": counts.get("truck", 0),
            "auto_rickshaws": counts.get("auto_rickshaw", 0)
        })

        if show_preview:
            cv2.putText(frame, f"Vehicles: {total_vehicles}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Lighting: {light_audit['lighting_condition']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("RoadSense AI - Vehicle Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Convert & Save DataFrames
    # --------------------------------------------------------
    metrics_df = pd.DataFrame(metric_records)
    trajectory_df = pd.DataFrame(trajectory_records)

    Path(metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(trajectory_output).parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(metrics_output, index=False)
    trajectory_df.to_csv(trajectory_output, index=False)

    unique_tracked = int(trajectory_df["track_id"].nunique()) if not trajectory_df.empty else 0

    if metadata_output:
        session_meta = {
            "session_id": session_id,
            "source_video": str(video_path),
            "location_id": location_id,
            "camera_id": camera_id,
            "observation_date": observation_date,
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 2),
            "frame_count": total_frames,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "record_count": len(metrics_df),
            "trajectory_count": len(trajectory_df),
            "unique_tracked_vehicles": unique_tracked,
            "lighting_conditions": list(lighting_conditions_observed)
        }
        with open(metadata_output, "w") as f:
            json.dump(session_meta, f, indent=4)
        print(f"[+] Session metadata saved to: {metadata_output}")

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("VISION PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nProcessed metric records : {len(metrics_df)}")
    print(f"Trajectory records       : {len(trajectory_df)}")
    if not trajectory_df.empty:
        print(f"Unique tracked vehicles  : {unique_tracked}")
        occluded_pct = (trajectory_df['is_occluded'].sum() / len(trajectory_df)) * 100.0 if 'is_occluded' in trajectory_df.columns else 0.0
        print(f"Occluded observation rate: {occluded_pct:.1f}%")
        print(f"Lighting conditions      : {list(lighting_conditions_observed)}")

    print(f"\nTraffic metrics saved to:\n  {metrics_output}")
    print(f"\nVehicle trajectories saved to:\n  {trajectory_output}")
    print("=" * 70)

    return metrics_df, trajectory_df


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RoadSense AI - Vehicle Detector, Trajectory Tracker & Preprocessor")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="Path to input video file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="YOLO model path or HuggingFace ID (e.g. yolo11n.pt or iisc-aim/UVH-26 weights)")
    parser.add_argument("--session", type=str, default=None, help="Session identifier (e.g. session_001)")
    parser.add_argument("--location-id", type=str, default="loc_01", help="Location identifier")
    parser.add_argument("--camera-id", type=str, default="cam_01", help="Camera identifier")
    parser.add_argument("--date", type=str, default=None, help="Observation date (YYYY-MM-DD)")
    parser.add_argument("--show", action="store_true", help="Display live video playback window")
    parser.add_argument("--night-mode", action="store_true", help="Force night/low-light enhancement mode")

    args = parser.parse_args()

    process_video(
        video_path=args.video,
        model_name=args.model,
        session_id=args.session,
        location_id=args.location_id,
        camera_id=args.camera_id,
        observation_date=args.date,
        show_preview=args.show,
        force_night_mode=args.night_mode
    )


if __name__ == "__main__":
    main()