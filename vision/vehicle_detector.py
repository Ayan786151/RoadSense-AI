import os
import sys
import json
import argparse
from datetime import datetime, timezone
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# ROAD SENSE AI - VEHICLE DETECTOR + TRAJECTORY LOGGER
# ============================================================

DEFAULT_VIDEO_PATH = "videos/traffic.mp4"
DEFAULT_METRICS_OUTPUT = "data/vision_traffic_metrics.csv"
DEFAULT_TRAJECTORY_OUTPUT = "data/vehicle_trajectories.csv"

MODEL_NAME = "yolo11n.pt"

# COCO vehicle classes
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# ============================================================
# RESOLVE SESSION PATHS
# ============================================================

def resolve_output_paths(session_id: str = None):
    """
    Resolves metrics and trajectory output paths based on session_id.
    If session_id is provided, routes to data/sessions/{session_id}/.
    Otherwise defaults to root data/ paths.
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
# MAIN VIDEO PROCESSING
# ============================================================

def process_video(
    video_path: str = DEFAULT_VIDEO_PATH,
    session_id: str = None,
    location_id: str = "loc_01",
    camera_id: str = "cam_01",
    observation_date: str = None,
    show_preview: bool = False
):
    metrics_output, trajectory_output, metadata_output, output_dir = resolve_output_paths(session_id)

    if observation_date is None:
        observation_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("=" * 70)
    print("ROAD SENSE AI - COMPUTER VISION MODULE")
    print("VEHICLE DETECTION + TRAJECTORY TRACKING")
    print("=" * 70)
    if session_id:
        print(f"Active Session: {session_id}")
    print(f"Source Video  : {video_path}")
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

    print("\nLoading YOLO model...")
    model = YOLO(MODEL_NAME)
    print("YOLO model loaded with fresh tracker state.")

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open the traffic video: {video_path}"
        )

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

    frame_number = 0

    # Process every 5th frame
    PROCESS_EVERY_N_FRAMES = 5

    print("\nStarting vehicle detection...")
    print("Trajectory logging: ENABLED")
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
        # YOLO tracking (persist=True maintains IDs within this video)
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        # ----------------------------------------------------
        # Vehicle counters
        # ----------------------------------------------------

        counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }

        total_vehicles = 0

        # ----------------------------------------------------
        # Process detections
        # ----------------------------------------------------

        if result.boxes is not None:
            boxes = result.boxes

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                if class_id not in VEHICLE_CLASSES:
                    continue

                vehicle_type = VEHICLE_CLASSES[class_id]
                counts[vehicle_type] += 1
                total_vehicles += 1

                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Tracking Point: Bottom-center of the bounding box
                # The bottom-center point approximates the vehicle's contact point
                # with the road plane and is therefore more appropriate for
                # perspective transformation than the bounding-box center.
                center_x = (x1 + x2) / 2.0
                center_y = float(y2)

                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].item())

                # Save trajectory
                if track_id is not None:
                    trajectory_records.append({
                        "timestamp_seconds": round(timestamp, 3),
                        "frame_number": frame_number,
                        "track_id": track_id,
                        "vehicle_type": vehicle_type,
                        "center_x": round(center_x, 2),
                        "center_y": round(center_y, 2),
                        "bbox_width": int(x2 - x1),
                        "bbox_height": int(y2 - y1)
                    })

                # Visual annotation if preview enabled
                if show_preview:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)
                    label = vehicle_type
                    if track_id is not None:
                        label += f" ID:{track_id}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

        # ----------------------------------------------------
        # Save frame-level metrics
        # ----------------------------------------------------

        metric_records.append({
            "timestamp_seconds": round(timestamp, 3),
            "vehicle_count": total_vehicles,
            "cars": counts["car"],
            "motorcycles": counts["motorcycle"],
            "buses": counts["bus"],
            "trucks": counts["truck"]
        })

        # ----------------------------------------------------
        # Display preview if requested
        # ----------------------------------------------------

        if show_preview:
            cv2.putText(frame, f"Vehicles: {total_vehicles}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Time: {timestamp:.1f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Trajectory Tracking: ON", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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
    # Convert to DataFrames
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(metric_records)
    trajectory_df = pd.DataFrame(trajectory_records)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    Path(metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(trajectory_output).parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Save frame metrics & trajectories
    # --------------------------------------------------------

    metrics_df.to_csv(metrics_output, index=False)
    trajectory_df.to_csv(trajectory_output, index=False)

    # --------------------------------------------------------
    # Save session metadata if in session mode
    # --------------------------------------------------------

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
            "unique_tracked_vehicles": unique_tracked
        }
        with open(metadata_output, "w") as f:
            json.dump(session_meta, f, indent=4)
        print(f"[+] Session metadata saved to: {metadata_output}")

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VISION PROCESSING COMPLETE")
    print("=" * 70)

    print(f"\nProcessed metric records : {len(metrics_df)}")
    print(f"Trajectory records       : {len(trajectory_df)}")

    if not trajectory_df.empty:
        print(f"Unique tracked vehicles  : {unique_tracked}")
        print(f"Tracked vehicle types    : {trajectory_df['vehicle_type'].nunique()}")

    print(f"\nTraffic metrics saved to:\n  {metrics_output}")
    print(f"\nVehicle trajectories saved to:\n  {trajectory_output}")
    print("=" * 70)

    return metrics_df, trajectory_df


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RoadSense AI - Vehicle Detector and Trajectory Tracker")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="Path to input video file")
    parser.add_argument("--session", type=str, default=None, help="Session identifier (e.g. session_002)")
    parser.add_argument("--location-id", type=str, default="loc_01", help="Location identifier")
    parser.add_argument("--camera-id", type=str, default="cam_01", help="Camera identifier")
    parser.add_argument("--date", type=str, default=None, help="Observation date (YYYY-MM-DD)")
    parser.add_argument("--show", action="store_true", help="Display live video playback window")

    args = parser.parse_args()

    process_video(
        video_path=args.video,
        session_id=args.session,
        location_id=args.location_id,
        camera_id=args.camera_id,
        observation_date=args.date,
        show_preview=args.show
    )


if __name__ == "__main__":
    main()