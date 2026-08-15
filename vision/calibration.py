"""
================================================================================
ROAD SENSE AI - PERSPECTIVE HOMOGRAPHY CALIBRATION SYSTEM
MODULE: PLANAR PERSPECTIVE RECTIFICATION & REAL-WORLD MAPPING
================================================================================

This module provides a reusable, robust calibration engine that maps 2D camera
pixel coordinates onto a 2D real-world ground coordinate plane (in meters)
using a 4-point planar projective transformation (homography).

MATHEMATICAL FOUNDATION:
1. ROAD PLANE HOMOGRAPHY:
   Let p_img = [x, y, 1]^T be the homogeneous pixel coordinates of a vehicle's
   road-contact point (bottom-center of bounding box).
   Let P_world = [X, Y, 1]^T be the corresponding real-world ground coordinate (meters).
   There exists a 3x3 non-singular projective transformation matrix H such that:
       s * P_world = H * p_img
   where s is a non-zero projective scale factor.

2. COMPUTATION:
   Given 4 non-collinear image points and 4 corresponding real-world points,
   H is uniquely determined up to scale and computed using cv2.getPerspectiveTransform.

3. FORWARD TRANSFORM:
   Given any arbitrary pixel coordinate p = (x, y), its real-world ground position
   P = (X, Y) is evaluated using cv2.perspectiveTransform.

4. STATUS CATEGORIZATION:
   - "calibrated"       : Verified scene-specific physical survey points supplied.
   - "demo_calibration" : Predefined demonstrator points supplied for reference videos.
   - "uncalibrated"     : No homography available; physical speed evaluates to NaN.
================================================================================
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import cv2


# Default path to calibration configuration
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration_config.json"


# ==============================================================================
# 1. POINT VALIDATION
# ==============================================================================

def validate_calibration_points(
    image_points: Union[List, np.ndarray],
    world_points: Union[List, np.ndarray]
) -> Tuple[bool, Optional[str]]:
    """
    Validates that image and world points form a non-degenerate, computable
    4-point projective transformation pair.

    Checks:
    1. Exactly 4 points in each array.
    2. 2D coordinates per point.
    3. Numeric, finite coordinates (no NaN or Inf).
    4. No duplicate points in image or world sets.
    5. Non-degenerate quadrilaterals (non-zero area).
    6. Non-singular homography matrix can be calculated.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    try:
        img_arr = np.float32(image_points)
        wld_arr = np.float32(world_points)
    except Exception as e:
        return False, f"Could not convert points to float32 array: {e}"

    if img_arr.shape != (4, 2):
        return False, f"Image points shape must be (4, 2), got {img_arr.shape}."

    if wld_arr.shape != (4, 2):
        return False, f"World points shape must be (4, 2), got {wld_arr.shape}."

    if not np.all(np.isfinite(img_arr)):
        return False, "Image points contain non-finite values (NaN or Inf)."

    if not np.all(np.isfinite(wld_arr)):
        return False, "World points contain non-finite values (NaN or Inf)."

    # Check for duplicate image points
    unique_img = np.unique(img_arr, axis=0)
    if len(unique_img) < 4:
        return False, "Image points contain duplicate coordinates."

    # Check for duplicate world points
    unique_wld = np.unique(wld_arr, axis=0)
    if len(unique_wld) < 4:
        return False, "World points contain duplicate coordinates."

    # Check non-zero quadrilateral area for image points
    img_area = cv2.contourArea(img_arr)
    if img_area < 1.0:
        return False, f"Image points quadrilateral has near-zero area ({img_area:.2f})."

    # Check non-zero quadrilateral area for world points
    wld_area = cv2.contourArea(wld_arr)
    if wld_area < 0.01:
        return False, f"World points quadrilateral has near-zero area ({wld_area:.4f})."

    # Test perspective matrix computation
    try:
        matrix = cv2.getPerspectiveTransform(img_arr, wld_arr)
        if matrix is None or not np.all(np.isfinite(matrix)):
            return False, "Failed to compute finite perspective transformation matrix."
        det = np.linalg.det(matrix)
        if abs(det) < 1e-12:
            return False, f"Singular homography matrix (determinant = {det})."
    except Exception as e:
        return False, f"cv2.getPerspectiveTransform failed: {e}"

    return True, None


# ==============================================================================
# 2. HOMOGRAPHY CALIBRATOR CLASS
# ==============================================================================

class HomographyCalibrator:
    """
    Encapsulates 4-point perspective transformation, coordinate projection,
    and calibration status management.
    """

    def __init__(
        self,
        image_points: Optional[Union[List, np.ndarray]] = None,
        world_points: Optional[Union[List, np.ndarray]] = None,
        status: str = "uncalibrated",
        description: str = "",
        video_name: str = ""
    ):
        self.status = status
        self.description = description
        self.video_name = video_name
        self.image_points: Optional[np.ndarray] = None
        self.world_points: Optional[np.ndarray] = None
        self.matrix: Optional[np.ndarray] = None
        self.inv_matrix: Optional[np.ndarray] = None

        if image_points is not None and world_points is not None:
            is_valid, err_msg = validate_calibration_points(image_points, world_points)
            if not is_valid:
                raise ValueError(f"Invalid calibration points: {err_msg}")

            self.image_points = np.float32(image_points)
            self.world_points = np.float32(world_points)
            self._compute_matrices()
            if self.status == "uncalibrated":
                self.status = "calibrated"

    def _compute_matrices(self):
        """Computes forward (pixel -> world) and inverse (world -> pixel) matrices."""
        self.matrix = cv2.getPerspectiveTransform(self.image_points, self.world_points)
        self.inv_matrix = cv2.getPerspectiveTransform(self.world_points, self.image_points)

    @property
    def is_calibrated(self) -> bool:
        """Returns True if a valid perspective matrix is available."""
        return self.matrix is not None and self.status in ["calibrated", "demo_calibration"]

    def pixel_to_world(self, points: Union[List, np.ndarray]) -> np.ndarray:
        """
        Transforms 2D pixel coordinates (x, y) into real-world ground coordinates (X, Y) in meters.

        Args:
            points: (N, 2) array-like of [x, y] pixel coordinates.

        Returns:
            (N, 2) numpy array of [X, Y] in meters. Returns NaNs if uncalibrated.
        """
        pts_arr = np.float32(points)
        if pts_arr.ndim == 1:
            pts_arr = pts_arr.reshape(1, -1)

        if pts_arr.shape[1] != 2:
            raise ValueError(f"Input points must have shape (N, 2), got {pts_arr.shape}")

        if not self.is_calibrated:
            return np.full_like(pts_arr, np.nan, dtype=np.float32)

        # Handle rows with NaNs safely
        valid_mask = np.all(np.isfinite(pts_arr), axis=1)
        world_pts = np.full_like(pts_arr, np.nan, dtype=np.float32)

        if np.any(valid_mask):
            valid_pts = pts_arr[valid_mask].reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(valid_pts, self.matrix)
            world_pts[valid_mask] = transformed.reshape(-1, 2)

        return world_pts

    def world_to_pixel(self, world_points: Union[List, np.ndarray]) -> np.ndarray:
        """
        Transforms 2D world coordinates (X, Y) back into pixel coordinates (x, y).
        """
        pts_arr = np.float32(world_points)
        if pts_arr.ndim == 1:
            pts_arr = pts_arr.reshape(1, -1)

        if not self.is_calibrated or self.inv_matrix is None:
            return np.full_like(pts_arr, np.nan, dtype=np.float32)

        valid_mask = np.all(np.isfinite(pts_arr), axis=1)
        pixel_pts = np.full_like(pts_arr, np.nan, dtype=np.float32)

        if np.any(valid_mask):
            valid_pts = pts_arr[valid_mask].reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(valid_pts, self.inv_matrix)
            pixel_pts[valid_mask] = transformed.reshape(-1, 2)

        return pixel_pts

    @classmethod
    def from_config(
        cls,
        config_source: Optional[Union[str, Path, Dict]] = None,
        video_key: Optional[str] = None,
        mode: str = "auto"
    ) -> "HomographyCalibrator":
        """
        Factory method to instantiate a calibrator from configuration file or dict.

        Modes:
        - "none"   : Explicitly creates an uncalibrated calibrator.
        - "demo"   : Forces demo calibration lookup.
        - "custom" : Forces custom calibration lookup.
        - "auto"   : Resolves matching video_key or demo fallback.
        """
        if mode == "none":
            return cls(status="uncalibrated", description="Explicitly uncalibrated mode.")

        # Load config
        config_data = {}
        if config_source is None:
            config_source = DEFAULT_CONFIG_PATH

        if isinstance(config_source, (str, Path)):
            path = Path(config_source)
            if path.exists():
                try:
                    with open(path, "r") as f:
                        config_data = json.load(f)
                except Exception as e:
                    print(f"[!] Warning: Failed to parse calibration config {path}: {e}")
            else:
                print(f"[!] Info: Calibration config not found at {path}. Operating in uncalibrated mode.")
        elif isinstance(config_source, dict):
            config_data = config_source

        if not config_data:
            return cls(status="uncalibrated", description="No calibration config provided.")

        # Normalize video_key
        key_candidates = []
        if video_key:
            basename = Path(video_key).name
            stem = Path(video_key).stem
            key_candidates.extend([video_key, basename, stem])
        key_candidates.extend(["traffic.mp4", "default", "demo"])

        selected_entry = None
        matched_key = None

        for k in key_candidates:
            if k in config_data:
                selected_entry = config_data[k]
                matched_key = k
                break

        if not selected_entry:
            return cls(status="uncalibrated", description=f"No matching calibration found for '{video_key}'.")

        img_pts = selected_entry.get("image_points")
        wld_pts = selected_entry.get("world_points")
        status = selected_entry.get("status", "calibrated")
        desc = selected_entry.get("description", f"Calibration loaded for {matched_key}")

        if mode == "demo" or "demo" in matched_key.lower():
            status = "demo_calibration"
        elif mode == "custom":
            status = "calibrated"

        if img_pts is None or wld_pts is None:
            return cls(status="uncalibrated", description=f"Calibration entry for '{matched_key}' is missing points.")

        try:
            return cls(
                image_points=img_pts,
                world_points=wld_pts,
                status=status,
                description=desc,
                video_name=matched_key
            )
        except Exception as e:
            print(f"[!] Warning: Invalid calibration in config for '{matched_key}': {e}")
            return cls(status="uncalibrated", description=f"Calibration failed: {e}")

    def __repr__(self) -> str:
        return f"<HomographyCalibrator status='{self.status}' video='{self.video_name}' is_calibrated={self.is_calibrated}>"


# ==============================================================================
# 3. VISUAL DEBUGGING UTILITY
# ==============================================================================

def draw_calibration_overlay(
    frame: np.ndarray,
    calibrator: HomographyCalibrator,
    output_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Draws the 4 image calibration points, connecting quadrilateral polygon,
    coordinate labels, and calibration status overlay on the video frame.
    """
    vis_frame = frame.copy()

    if not calibrator.is_calibrated or calibrator.image_points is None:
        cv2.putText(
            vis_frame,
            "STATUS: UNCALIBRATED (No Homography)",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        return vis_frame

    img_pts = calibrator.image_points.astype(int)
    wld_pts = calibrator.world_points

    # 1. Draw semi-transparent quadrilateral overlay on the road plane
    overlay = vis_frame.copy()
    cv2.fillPoly(overlay, [img_pts], (0, 255, 255))
    cv2.addWeighted(overlay, 0.25, vis_frame, 0.75, 0, vis_frame)

    # 2. Draw polygon boundary lines
    cv2.polylines(vis_frame, [img_pts], isClosed=True, color=(0, 255, 255), thickness=2)

    # 3. Draw and label the 4 calibration reference points
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
    labels = ["P1 (TL)", "P2 (TR)", "P3 (BR)", "P4 (BL)"]

    for idx, (pt, label, col) in enumerate(zip(img_pts, labels, colors)):
        x, y = pt
        wx, wy = wld_pts[idx]
        cv2.circle(vis_frame, (x, y), 8, col, -1)
        cv2.circle(vis_frame, (x, y), 10, (255, 255, 255), 2)

        text_label = f"{label}: img=({x},{y}) -> world=({wx:.1f}m,{wy:.1f}m)"
        offset_y = -12 if idx < 2 else 24
        cv2.putText(
            vis_frame,
            text_label,
            (max(x - 80, 20), y + offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2
        )
        cv2.putText(
            vis_frame,
            text_label,
            (max(x - 80, 20), y + offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1
        )

    # 4. Header status banner
    status_text = f"CALIBRATION STATUS: {calibrator.status.upper()} ({calibrator.video_name})"
    cv2.putText(vis_frame, status_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), vis_frame)
        print(f"[+] Calibration debug visualization saved to: {out_p}")

    return vis_frame
