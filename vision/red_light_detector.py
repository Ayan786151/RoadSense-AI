"""
================================================================================
ROAD SENSE AI - RED LIGHT VIOLATION & VIRTUAL STOP-LINE ENFORCEMENT
MODULE: TRAFFIC SIGNAL COMPLIANCE & INTERSECTION INTRUSION DETECTOR
================================================================================

This module performs real-time traffic signal violation detection:
1. Monitors Traffic Signal Phase (RED, YELLOW, GREEN) via YOLO / chromatic HSV.
2. Maintains a configurable Virtual Stop-Line across roadway lanes.
3. Tracks vehicle bottom-center trajectories across the stop line.
4. If a vehicle crosses the stop line while the light is RED, it flags and
   logs a RED_LIGHT_VIOLATION with vehicle metadata, snapshot, and timestamp.
================================================================================
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2


class TrafficSignalPhaseDetector:
    """
    Analyzes traffic light signal state (RED / YELLOW / GREEN / OFF).
    Supports automated ROI extraction and HSV chromatic energy analysis.
    """

    def __init__(self, manual_roi: Optional[Tuple[int, int, int, int]] = None):
        """
        manual_roi: [x, y, w, h] ROI of the traffic light in the camera frame.
        """
        self.roi = manual_roi
        self.current_state = "GREEN"  # Default initial state
        self.state_confidence = 0.90

    def detect_state_from_frame(
        self,
        frame: np.ndarray,
        detected_tl_boxes: Optional[List[np.ndarray]] = None
    ) -> Tuple[str, float]:
        """
        Evaluates traffic light state from frame using either detected bounding box or ROI.
        """
        tl_crop = None

        if detected_tl_boxes and len(detected_tl_boxes) > 0:
            # Use largest detected traffic light bounding box
            best_box = max(detected_tl_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            x1, y1, x2, y2 = map(int, best_box)
            tl_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        elif self.roi:
            rx, ry, rw, rh = self.roi
            tl_crop = frame[max(0, ry):min(frame.shape[0], ry + rh), max(0, rx):min(frame.shape[1], rx + rw)]

        if tl_crop is None or tl_crop.size == 0:
            return self.current_state, self.state_confidence

        # Convert to HSV color space for robust color segmentation
        hsv = cv2.cvtColor(tl_crop, cv2.COLOR_BGR2HSV)

        # Red ranges (wraps around 0/180)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Yellow range
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])

        # Green range
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([90, 255, 255])

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        red_pixels = np.count_nonzero(mask_red)
        yellow_pixels = np.count_nonzero(mask_yellow)
        green_pixels = np.count_nonzero(mask_green)

        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        total_pixels = tl_crop.shape[0] * tl_crop.shape[1]

        if max_pixels < total_pixels * 0.02:
            # Low chromatic energy - maintain current or return fallback
            return self.current_state, 0.70

        if red_pixels == max_pixels:
            self.current_state = "RED"
            self.state_confidence = min(0.99, red_pixels / float(total_pixels) * 5)
        elif yellow_pixels == max_pixels:
            self.current_state = "YELLOW"
            self.state_confidence = min(0.95, yellow_pixels / float(total_pixels) * 5)
        else:
            self.current_state = "GREEN"
            self.state_confidence = min(0.99, green_pixels / float(total_pixels) * 5)

        return self.current_state, round(self.state_confidence, 2)


class RedLightViolationDetector:
    """
    Monitors vehicle trajectories crossing a designated virtual stop line while
    the traffic light signal phase is RED.
    """

    def __init__(
        self,
        stop_line_y_ratio: float = 0.65,
        signal_state_override: Optional[str] = None
    ):
        """
        stop_line_y_ratio: Relative vertical position of the stop line (0.0 to 1.0)
        signal_state_override: Forced signal state ("RED", "GREEN", "AUTO")
        """
        self.stop_line_y_ratio = stop_line_y_ratio
        self.signal_state_override = signal_state_override
        self.signal_phase_detector = TrafficSignalPhaseDetector()

        # Track vehicle positions across frames: track_id -> previous_center_y
        self.previous_positions: Dict[int, float] = {}
        # Violation registry to prevent duplicate logging
        self.logged_violations: Dict[int, Dict[str, Any]] = {}
        self.active_frame_violations: List[Dict[str, Any]] = []

    def process_frame_violations(
        self,
        frame: np.ndarray,
        tracked_vehicles: List[Dict[str, Any]],
        timestamp: float,
        forced_red: bool = False
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Evaluates all active vehicle trajectories against the stop line and signal phase.

        Parameters:
            frame: Full BGR frame
            tracked_vehicles: List of dicts with [track_id, vehicle_type, center_x, center_y, bbox]
            timestamp: Video timestamp in seconds
            forced_red: If True, forces red signal state for demonstration/testing
        """
        h_frame, w_frame = frame.shape[:2]
        stop_line_y = int(h_frame * self.stop_line_y_ratio)

        # 1. Determine Signal Phase
        if forced_red or self.signal_state_override == "RED":
            signal_phase = "RED"
        elif self.signal_state_override == "GREEN":
            signal_phase = "GREEN"
        else:
            signal_phase, _ = self.signal_phase_detector.detect_state_from_frame(frame)

        self.active_frame_violations = []

        # 2. Check each vehicle trajectory
        for veh in tracked_vehicles:
            t_id = veh.get("track_id")
            if t_id is None:
                continue

            curr_y = float(veh["center_y"])
            v_type = veh.get("vehicle_type", "vehicle")
            bbox = veh.get("bbox", [])

            prev_y = self.previous_positions.get(t_id)
            self.previous_positions[t_id] = curr_y

            if prev_y is not None:
                # Vehicle moving down the frame crossing the stop line (prev_y <= stop_line_y < curr_y)
                crossed_stop_line = (prev_y <= stop_line_y) and (curr_y > stop_line_y)

                if crossed_stop_line and signal_phase == "RED":
                    if t_id not in self.logged_violations:
                        violation_record = {
                            "track_id": t_id,
                            "vehicle_type": v_type,
                            "violation_type": "RED_LIGHT_RUNNING",
                            "timestamp_seconds": timestamp,
                            "signal_state": "RED",
                            "cross_y": round(curr_y, 1),
                            "stop_line_y": stop_line_y,
                            "bbox": bbox
                        }
                        self.logged_violations[t_id] = violation_record
                        self.active_frame_violations.append(violation_record)

        return self.active_frame_violations, signal_phase

    def draw_annotation(
        self,
        frame: np.ndarray,
        signal_phase: str,
        active_violations: List[Dict[str, Any]]
    ):
        """Draws virtual stop line and red light violation visual alerts."""
        h_frame, w_frame = frame.shape[:2]
        stop_line_y = int(h_frame * self.stop_line_y_ratio)

        # Color based on signal phase
        line_color = (0, 0, 255) if signal_phase == "RED" else (0, 255, 0)
        thickness = 3 if signal_phase == "RED" else 2

        # Draw stop line
        cv2.line(frame, (0, stop_line_y), (w_frame, stop_line_y), line_color, thickness)
        
        # Stop line label badge
        badge_text = f"VIRTUAL STOP-LINE [SIGNAL: {signal_phase}]"
        cv2.putText(frame, badge_text, (20, stop_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

        # Highlight violating vehicles
        for viol in active_violations:
            bbox = viol.get("bbox", [])
            if len(bbox) == 4:
                vx1, vy1, vx2, vy2 = map(int, bbox)
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 4)
                alert_text = f"RED LIGHT VIOLATION! ID:{viol['track_id']}"
                cv2.putText(frame, alert_text, (vx1, max(20, vy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Returns total red light violation summary."""
        return {
            "total_red_light_violations": len(self.logged_violations),
            "violations_list": list(self.logged_violations.values())
        }
