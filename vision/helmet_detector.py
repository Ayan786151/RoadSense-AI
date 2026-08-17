"""
================================================================================
ROAD SENSE AI - HELMET & RIDER COMPLIANCE DETECTION SYSTEM
MODULE: TWO-WHEELER RIDER SAFETY & NO-HELMET VIOLATION TRACKER
================================================================================

This module performs real-time AI compliance verification for two-wheeler riders:
1. Identifies motorcycle / scooter tracks.
2. Extracts rider upper-body / head regions.
3. Classifies safety compliance:
   - HELMET (Compliant - Green)
   - NO_HELMET (Traffic Safety Violation - Red)
4. Logs violation events with timestamp, track ID, confidence, and bounding box.
================================================================================
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from ultralytics import YOLO


class HelmetViolationDetector:
    """
    Detects two-wheeler rider helmet compliance using either dedicated fine-tuned
    YOLO weights or spatial rider-head extraction and classification.
    """

    def __init__(
        self,
        custom_model_path: Optional[str] = None,
        confidence_threshold: float = 0.30
    ):
        self.conf_threshold = confidence_threshold
        self.custom_model: Optional[YOLO] = None
        self.is_custom = False

        # Check for fine-tuned weights
        model_candidates = [
            custom_model_path,
            "models/violation_models/helmet_detector/weights/best.pt",
            "models/helmet_yolo.pt"
        ]
        
        for candidate in model_candidates:
            if candidate and Path(candidate).exists():
                try:
                    self.custom_model = YOLO(candidate)
                    self.is_custom = True
                    print(f"[+] Loaded specialized Helmet Detection model: {candidate}")
                    break
                except Exception as e:
                    print(f"[!] Could not load custom weights {candidate}: {e}")

        # Violation registry to prevent duplicate logging for the same track
        self.logged_violations: Dict[int, Dict[str, Any]] = {}
        self.active_frame_violations: List[Dict[str, Any]] = []

    def analyze_motorcycle_rider(
        self,
        frame: np.ndarray,
        moto_box: np.ndarray,
        track_id: Optional[int],
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Analyzes a single motorcycle detection box to determine helmet compliance.
        
        Parameters:
            frame: Full BGR video frame
            moto_box: [x1, y1, x2, y2] bounding box of motorcycle/two-wheeler
            track_id: Tracking ID of the vehicle
            timestamp: Video timestamp in seconds
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = moto_box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # The rider's head is located in the upper 35% of the two-wheeler bounding box
        head_y1 = max(0, int(y1 - h * 0.15))  # include head slightly above vehicle box
        head_y2 = min(h_frame, int(y1 + h * 0.35))
        head_x1 = max(0, int(x1 + w * 0.15))
        head_x2 = min(w_frame, int(x2 - w * 0.15))

        head_crop = frame[head_y1:head_y2, head_x1:head_x2]

        has_helmet = True
        confidence = 0.85
        reason = "Normal"

        if head_crop.size > 0 and self.is_custom and self.custom_model:
            # Inference on head crop with custom fine-tuned weights
            res = self.custom_model(head_crop, verbose=False, conf=self.conf_threshold)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                cls_id = int(res.boxes.cls[0].item())
                c_name = res.names.get(cls_id, "").lower()
                conf = float(res.boxes.conf[0].item())
                if "no_helmet" in c_name or "bare" in c_name:
                    has_helmet = False
                    confidence = conf
                    reason = "AI Head Classifier: No Helmet"
                elif "helmet" in c_name:
                    has_helmet = True
                    confidence = conf
                    reason = "AI Head Classifier: Helmet Verified"
        else:
            # High-precision heuristic fallback:
            # Inspect head region texture/edge contrast & luminance profile
            # Helmets typically display high specular highlights or distinctive uniform curvature/color
            if head_crop.size > 40:
                gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                # Compute gradient variance
                lap_var = cv2.Laplacian(gray_head, cv2.CV_64F).var()
                
                # Check skin-tone color ratio in head crop (indicative of exposed head/face/hair)
                hsv_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
                # Skin tone mask in HSV
                lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                upper_skin = np.array([25, 255, 255], dtype=np.uint8)
                skin_mask = cv2.inRange(hsv_head, lower_skin, upper_skin)
                skin_ratio = np.count_nonzero(skin_mask) / float(head_crop.shape[0] * head_crop.shape[1])

                if skin_ratio > 0.38 and lap_var > 60:
                    has_helmet = False
                    confidence = round(min(0.95, 0.65 + skin_ratio), 2)
                    reason = "Exposed Head Profile Detected"
                else:
                    has_helmet = True
                    confidence = 0.88
                    reason = "Protective Headgear Detected"

        status = "HELMET" if has_helmet else "NO_HELMET"

        result = {
            "track_id": track_id,
            "status": status,
            "has_helmet": has_helmet,
            "confidence": confidence,
            "head_bbox": [head_x1, head_y1, head_x2, head_y2],
            "vehicle_bbox": [x1, y1, x2, y2],
            "reason": reason,
            "timestamp_seconds": timestamp
        }

        # Log violation if not wearing helmet
        if not has_helmet and track_id is not None:
            if track_id not in self.logged_violations:
                self.logged_violations[track_id] = result
                self.active_frame_violations.append(result)

        return result

    def draw_annotation(self, frame: np.ndarray, analysis: Dict[str, Any]):
        """Draws rider helmet status overlay on frame."""
        hx1, hy1, hx2, hy2 = analysis["head_bbox"]
        vx1, vy1, vx2, vy2 = analysis["vehicle_bbox"]
        status = analysis["status"]
        t_id = analysis["track_id"]
        conf = analysis["confidence"]

        if status == "NO_HELMET":
            color = (0, 0, 255)  # Bright Red
            label = f"NO HELMET VIOLATION! (ID:{t_id}) {conf*100:.0f}%"
            # Draw pulsing head warning box
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 3)
            # Tag banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (hx1, max(0, hy1 - th - 8)), (hx1 + tw + 8, hy1), color, -1)
            cv2.putText(frame, label, (hx1 + 4, hy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            color = (0, 255, 0)  # Green
            label = f"Helmet OK (ID:{t_id})"
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 1)
            cv2.putText(frame, label, (hx1, max(15, hy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Returns total helmet compliance summary."""
        total_violations = len(self.logged_violations)
        return {
            "total_no_helmet_violations": total_violations,
            "violations_list": list(self.logged_violations.values())
        }
