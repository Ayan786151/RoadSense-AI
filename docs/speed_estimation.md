# RoadSense AI - Scientifically Defensible Vehicle Speed Estimation

## Overview
This document describes the computer vision kinematics and perspective homography architecture implemented in RoadSense AI for vehicle speed estimation.

---

## 1. Scientific Foundations

### A. Road-Contact Point Tracking
- **Limitation of Geometric Bounding-Box Center**:
  The 2D geometric center of a vehicle bounding box $((x_1 + x_2)/2, (y_1 + y_2)/2)$ is elevated in 3D space above the road plane. Because cameras view traffic at an oblique pitch angle, the 3D elevation causes severe perspective parallax errors when projected onto a 2D ground plane.
- **Scientific Solution**:
  We track the **bottom-center point** of the bounding box:
  $$x_{\text{contact}} = \frac{x_1 + x_2}{2.0}, \quad y_{\text{contact}} = y_2$$
  The bottom-center point approximates the vehicle's contact point with the road plane and is therefore more appropriate for perspective transformation than the bounding-box center.

### B. Planar Perspective Homography
- **Why Homography is Required**:
  Image coordinates are affected by camera perspective projection. Equal pixel displacements near the camera horizon represent much larger real-world physical distances than pixel displacements near the bottom of the camera frame. A 4-point planar projective transformation maps 2D image coordinates onto a metric real-world ground coordinate system (in meters).
- **Mathematical Transformation**:
  Given 4 non-collinear image points $p_i = [x_i, y_i]^T$ and 4 surveyed real-world landmarks $P_i = [X_i, Y_i]^T$, a $3 \times 3$ non-singular matrix $H$ is computed via `cv2.getPerspectiveTransform()` satisfying:
  $$s \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
  Arbitrary vehicle contact points are projected via `cv2.perspectiveTransform()`.

### C. Scene-Specific Calibration
- A homography is dependent on the camera viewpoint, focal length, mounting height, tilt angle, and the physical geometry represented by the selected reference points.
- Homography cannot be generalized across different camera viewpoints without site-specific metric surveys.

---

## 2. Speed Calculation Pipeline

```
YOLO Detection (Bounding Box)
       ↓
Bottom-Center Road Contact Point: (x, y2)
       ↓
Planar Homography Projection H: (X_world, Y_world) [meters]
       ↓
Temporal Tracking by Track ID: (X_{t-1}, Y_{t-1}) → (X_t, Y_t)
       ↓
Real-World Displacement: Δd = sqrt((X_t - X_{t-1})² + (Y_t - Y_{t-1})²) [m]
       ↓
Elapsed Time: Δt = t - t_{prev} [s]
       ↓
Physical Speed (m/s): speed_mps = Δd / Δt
       ↓
Physical Speed (km/h): speed_kmh = speed_mps × 3.6
```

### First Observation Integrity
The first observation for every tracked vehicle has no previous position ($t_{\text{prev}}$ is `None`).
- **Policy**: `speed = NaN` (strictly excluded from downstream speed aggregations).
- **Rule**: We never insert artificial `0.0` values, ensuring initial track formation does not depress mean or median traffic velocity metrics.

### Defensive Physical Validation
- Non-positive time intervals ($\Delta t \le 0$) $\to$ `NaN` (`invalid_speed_reason = "invalid_time_interval"`).
- Impossible physical jumps ($\Delta d > 50.0\text{ m}$ in $0.167\text{ s} \implies > 1080\text{ km/h}$) caused by tracker ID swaps $\to$ `NaN` (`invalid_speed_reason = "unreasonable_world_displacement"`).
- No arbitrary artificial clamping (e.g. `min(speed, 80)`); physical outliers are flagged observably.

---

## 3. Operational Modes & Fallbacks

| Mode | Calibration State | Speed Output | Calibration Status Label |
| :--- | :--- | :--- | :--- |
| **`CALIBRATED`** | Verified surveyed landmarks supplied | Physical speed ($\text{km/h}$) | `"calibrated"` |
| **`DEMO_CALIBRATION`** | Predefined reference trapezoid for `videos/traffic.mp4` | Estimated physical speed ($\text{km/h}$) | `"demo_calibration"` |
| **`UNCALIBRATED`** | No homography supplied | Physical speed evaluates to `NaN`; camera pixel speed ($\text{px/s}$) retained | `"uncalibrated"` |

---

## 4. Remaining Limitations
1. **Pitch/Elevation Road Slopes**: 2D planar homography assumes a locally flat planar road. Roads with steep vertical grade changes require 3D calibration.
2. **Lens Distortion**: Non-linear radial lens distortion (e.g. wide-angle fisheye) should ideally be rectified prior to planar projection.
3. **Camera Vibration / Wind Shake**: Extreme camera shake can introduce high-frequency displacement noise, which can be mitigated with Kalman filtering or smoothing.
