"""
================================================================================
ROAD SENSE AI - PERSPECTIVE HOMOGRAPHY & LENS CALIBRATION SYSTEM
MODULE: PLANAR PERSPECTIVE RECTIFICATION & WIDE-ANGLE CORRECTION
================================================================================

This module provides a reusable, robust calibration engine that maps 2D camera
pixel coordinates onto a 2D real-world ground coordinate plane (in meters)
using planar projective transformation (homography) with optional wide-angle
lens distortion compensation.

MATHEMATICAL FOUNDATION:
1. WIDE-ANGLE / RADIAL LENS UNDISTORTION:
   For wide-angle / fisheye lenses with radial distortion (k1, k2, p1, p2),
   raw distorted pixel coordinates (u_d, v_d) are first mapped to normalized
   rectilinear coordinates via cv2.undistortPoints:
       x_u = x_d * (1 + k1*r^2 + k2*r^4) + [2*p1*x_d*y_d + p2*(r^2 + 2*x_d^2)]
       y_u = y_d * (1 + k1*r^2 + k2*r^4) + [p1*(r^2 + 2*y_d^2) + 2*p2*x_d*y_d]

2. ROAD PLANE HOMOGRAPHY:
   Let p_img = [u, v, 1]^T be the homogeneous rectified pixel coordinates of
   a vehicle's road-contact point (bottom-center of bounding box).
   Let P_world = [X, Y, 1]^T be the corresponding ground coordinate (meters).
   s * P_world = H * p_img

3. COMPUTATION:
   - For N == 4: H is uniquely computed via cv2.getPerspectiveTransform.
   - For N > 4 : H is solved via overdetermined Direct Linear Transform (DLT) with
     least-squares / RANSAC minimization (cv2.findHomography).

4. STATUS CATEGORIZATION:
   - "calibrated"       : Verified scene-specific physical survey points supplied.
   - "demo_calibration" : Predefined demonstrator reference points supplied for test videos.
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
    projective transformation pair (N >= 4 points).
    """
    try:
        img_arr = np.float32(image_points)
        wld_arr = np.float32(world_points)
    except Exception as e:
        return False, f"Could not convert points to float32 array: {e}"

    if img_arr.ndim != 2 or img_arr.shape[1] != 2:
        return False, f"Image points shape must be (N, 2), got {img_arr.shape}."

    if wld_arr.ndim != 2 or wld_arr.shape[1] != 2:
        return False, f"World points shape must be (N, 2), got {wld_arr.shape}."

    if len(img_arr) < 4:
        return False, f"At least 4 point correspondences required, got {len(img_arr)}."

    if len(img_arr) != len(wld_arr):
        return False, f"Point count mismatch: image has {len(img_arr)}, world has {len(wld_arr)}."

    if not np.all(np.isfinite(img_arr)):
        return False, "Image points contain non-finite values (NaN or Inf)."

    if not np.all(np.isfinite(wld_arr)):
        return False, "World points contain non-finite values (NaN or Inf)."

    unique_img = np.unique(img_arr, axis=0)
    if len(unique_img) < 4:
        return False, "Image points contain duplicate coordinates."

    unique_wld = np.unique(wld_arr, axis=0)
    if len(unique_wld) < 4:
        return False, "World points contain duplicate coordinates."

    try:
        if len(img_arr) == 4:
            matrix = cv2.getPerspectiveTransform(img_arr[:4], wld_arr[:4])
        else:
            matrix, _ = cv2.findHomography(img_arr, wld_arr, cv2.RANSAC, 5.0)

        if matrix is None or not np.all(np.isfinite(matrix)):
            return False, "Failed to compute finite perspective transformation matrix."
        det = np.linalg.det(matrix)
        if abs(det) < 1e-15:
            return False, f"Singular homography matrix (determinant = {det})."
    except Exception as e:
        return False, f"Homography computation failed: {e}"

    return True, None


# ==============================================================================
# 2. HOMOGRAPHY & LENS CALIBRATOR CLASS
# ==============================================================================

class HomographyCalibrator:
    """
    Encapsulates N-point perspective transformation, wide-angle lens undistortion,
    coordinate projection, and calibration status management.
    """

    def __init__(
        self,
        image_points: Optional[Union[List, np.ndarray]] = None,
        world_points: Optional[Union[List, np.ndarray]] = None,
        camera_matrix: Optional[Union[List, np.ndarray]] = None,
        dist_coeffs: Optional[Union[List, np.ndarray]] = None,
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
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        if camera_matrix is not None and dist_coeffs is not None:
            self.camera_matrix = np.float32(camera_matrix)
            self.dist_coeffs = np.float32(dist_coeffs)

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
        """Computes homography matrix H mapping image pixels to ground meters."""
        pts_src = self.image_points
        if self.has_lens_distortion:
            pts_src = self.undistort_points(pts_src)

        if len(pts_src) == 4:
            self.matrix = cv2.getPerspectiveTransform(pts_src[:4], self.world_points[:4])
            self.inv_matrix = cv2.getPerspectiveTransform(self.world_points[:4], pts_src[:4])
        else:
            self.matrix, _ = cv2.findHomography(pts_src, self.world_points, cv2.RANSAC, 5.0)
            self.inv_matrix, _ = cv2.findHomography(self.world_points, pts_src, cv2.RANSAC, 5.0)

    @property
    def is_calibrated(self) -> bool:
        """Returns True if a valid perspective matrix is available."""
        return self.matrix is not None and self.status in ["calibrated", "demo_calibration"]

    @property
    def has_lens_distortion(self) -> bool:
        """Returns True if lens distortion coefficients are configured."""
        return self.camera_matrix is not None and self.dist_coeffs is not None

    def undistort_points(self, points: Union[List, np.ndarray]) -> np.ndarray:
        """Applies lens undistortion to 2D pixel coordinates."""
        if not self.has_lens_distortion:
            return np.float32(points)

        pts_arr = np.float32(points).reshape(-1, 1, 2)
        undist = cv2.undistortPoints(
            pts_arr,
            self.camera_matrix,
            self.dist_coeffs,
            P=self.camera_matrix
        )
        return undist.reshape(-1, 2)

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rectifies wide-angle barrel distortion across an entire video frame."""
        if not self.has_lens_distortion or frame is None:
            return frame
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def pixel_to_world(self, points: Union[List, np.ndarray]) -> np.ndarray:
        """
        Transforms 2D pixel coordinates (u, v) into real-world ground coordinates (X, Y) in meters.
        Automatically applies wide-angle lens undistortion if configured.
        """
        pts_arr = np.float32(points)
        if pts_arr.ndim == 1:
            pts_arr = pts_arr.reshape(1, -1)

        if pts_arr.shape[1] != 2:
            raise ValueError(f"Input points must have shape (N, 2), got {pts_arr.shape}")

        if not self.is_calibrated:
            return np.full_like(pts_arr, np.nan, dtype=np.float32)

        # Handle NaNs safely
        valid_mask = np.all(np.isfinite(pts_arr), axis=1)
        world_pts = np.full_like(pts_arr, np.nan, dtype=np.float32)

        if np.any(valid_mask):
            valid_pts = pts_arr[valid_mask]
            if self.has_lens_distortion:
                valid_pts = self.undistort_points(valid_pts)

            transformed = cv2.perspectiveTransform(valid_pts.reshape(-1, 1, 2), self.matrix)
            world_pts[valid_mask] = transformed.reshape(-1, 2)

        return world_pts

    def world_to_pixel(self, world_points: Union[List, np.ndarray]) -> np.ndarray:
        """Transforms 2D world coordinates (X, Y) back into pixel coordinates (u, v)."""
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
        """Factory method to instantiate a calibrator from configuration file or dict."""
        if mode == "none":
            return cls(status="uncalibrated", description="Explicitly uncalibrated mode.")

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

        key_candidates = []
        if video_key:
            basename = Path(video_key).name
            stem = Path(video_key).stem
            key_candidates.extend([video_key, basename, stem])
            
            # Check for substring matches in config keys
            for config_k in config_data.keys():
                if config_k.lower() in video_key.lower() or config_k.lower() in basename.lower():
                    key_candidates.append(config_k)
                if "2min" in video_key.lower() and "2min" in config_k.lower():
                    key_candidates.insert(0, config_k)

        # Default fallback candidates
        key_candidates.extend(["traffic_2min.mp4", "traffic.mp4", "custom_camera_template", "default", "demo"])

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
        cam_mat = selected_entry.get("camera_matrix")
        dist_coef = selected_entry.get("distortion_coefficients")
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
                camera_matrix=cam_mat,
                dist_coeffs=dist_coef,
                status=status,
                description=desc,
                video_name=matched_key
            )
        except Exception as e:
            print(f"[!] Warning: Invalid calibration in config for '{matched_key}': {e}")
            return cls(status="uncalibrated", description=f"Calibration failed: {e}")


# ==============================================================================
# 3. VISUAL DEBUGGING UTILITY
# ==============================================================================

def draw_calibration_overlay(
    frame: np.ndarray,
    calibrator: HomographyCalibrator,
    output_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """Draws calibration polygon and status overlay on the video frame."""
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

    overlay = vis_frame.copy()
    cv2.fillPoly(overlay, [img_pts], (0, 255, 255))
    cv2.addWeighted(overlay, 0.25, vis_frame, 0.75, 0, vis_frame)
    cv2.polylines(vis_frame, [img_pts], isClosed=True, color=(0, 255, 255), thickness=2)

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]

    for idx, pt in enumerate(img_pts):
        x, y = pt
        wx, wy = wld_pts[idx]
        col = colors[idx % len(colors)]
        cv2.circle(vis_frame, (x, y), 8, col, -1)
        cv2.circle(vis_frame, (x, y), 10, (255, 255, 255), 2)

        text_label = f"P{idx+1}: img=({x},{y}) -> world=({wx:.1f}m,{wy:.1f}m)"
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

    lens_txt = " + Lens Undistortion" if calibrator.has_lens_distortion else ""
    status_text = f"CALIBRATION STATUS: {calibrator.status.upper()} ({calibrator.video_name}){lens_txt}"
    cv2.putText(vis_frame, status_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), vis_frame)
        print(f"[+] Calibration debug visualization saved to: {out_p}")

    return vis_frame
