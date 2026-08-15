# RoadSense AI - Advanced CCTV Speed Estimation & Computer Vision Enhancements

## Overview
This document describes the computer vision kinematics, perspective homography, wide-angle lens compensation, heavy traffic occlusion mitigation, and adaptive night-vision processing in RoadSense AI.

---

## 1. Real-World CCTV Challenges & Engineered Solutions

### A. Wide-Angle & Fisheye Lens Distortion
- **Challenge**: Wide-angle CCTV lenses suffer from radial barrel distortion ($k_1, k_2, p_1, p_2$), curving straight lanes near frame borders and corrupting planar homography coordinates.
- **Solution**:
  - `HomographyCalibrator` in [`vision/calibration.py`](file:///c:/Users/asus/Downloads/traffic_sim-main%20%283%29/traffic_sim-main/vision/calibration.py) incorporates optional camera intrinsics matrix $K$ and distortion coefficients $D = [k_1, k_2, p_1, p_2]$.
  - Distorted image coordinates $(u_d, v_d)$ are rectified via `cv2.undistortPoints()` prior to homography matrix multiplication:
    $$x_u = x_d(1 + k_1 r^2 + k_2 r^4) + 2 p_1 x_d y_d + p_2(r^2 + 2 x_d^2)$$
    $$y_u = y_d(1 + k_1 r^2 + k_2 r^4) + p_1(r^2 + 2 y_d^2) + 2 p_2 x_d y_d$$

### B. Heavy Traffic & Bumper-to-Bumper Occlusions
- **Challenge**: In dense traffic jams, front vehicles occlude the lower tires of rear vehicles, pushing bounding-box bottoms upward and creating false speed spikes.
- **Solution**:
  1. **Occlusion Detection**: [`vision/vehicle_detector.py`](file:///c:/Users/asus/Downloads/traffic_sim-main%20%283%29/traffic_sim-main/vision/vehicle_detector.py) performs perspective overlap checks on adjacent bounding boxes. If a rear vehicle overlaps $> 15\%$ with a front vehicle, `is_occluded = True`.
  2. **2D Ground-Plane Kalman Filter**: [`vision/kalman_tracker.py`](file:///c:/Users/asus/Downloads/traffic_sim-main%20%283%29/traffic_sim-main/vision/kalman_tracker.py) maintains a state vector $\mathbf{x} = [X, Y, v_X, v_Y]^T$. When an occluded or outlier measurement is detected ($> 4\sigma$ innovation gate), the filter **coast-predicts** using learned vehicle momentum, preventing false speed spikes.

### C. Nighttime & Low-Light Contrast Enhancement
- **Challenge**: At night, vehicle bodies blend into dark tarmac while headlamps cause glare and blooming, degrading YOLO detection confidence.
- **Solution**:
  - [`vision/enhancement.py`](file:///c:/Users/asus/Downloads/traffic_sim-main%20%283%29/traffic_sim-main/vision/enhancement.py) analyzes mean luminance $L^*$ in CIELAB color space.
  - If low light ($L^* < 65$), it applies **Contrast Limited Adaptive Histogram Equalization (CLAHE)** on the $L^*$ channel (clipLimit = 2.5, tileGrid = (8, 8)), boosting vehicle contours while avoiding noise amplification.
  - If deep night ($L^* < 40$), it applies power-law gamma correction ($\gamma = 0.75$).
  - Daytime frames ($L^* \ge 65$) bypass enhancement with zero alteration and zero overhead.

---

## 2. Speed Calculation Pipeline

```
Raw CCTV Frame
      ↓
[Adaptive Night / Low-Light Detector] (CLAHE / Gamma if L* < 65)
      ↓
[Lens Distortion Rectification] (cv2.undistortPoints)
      ↓
[YOLOv8 Detection + ByteTrack]
      ↓
[Occlusion Overlap Analysis] (Flags is_occluded)
      ↓
[Bottom-Center Ground Contact Point: ((x1+x2)/2, y2)]
      ↓
[Planar Homography H Projection: (X, Y) meters]
      ↓
[2D Ground-Plane Kalman Filter: State [X, Y, vX, vY]]
      ↓
[Defensible Physical Speed: sqrt(vX² + vY²) × 3.6 km/h]
```

### Initial Observation Integrity
The first observation for every tracked vehicle receives `speed = NaN` (never artificially $0.0$) and is excluded from downstream aggregations.

---

## 3. Operational Calibration Modes

| Mode | Trigger | Speed Metric | Calibration Label |
| :--- | :--- | :--- | :--- |
| **`CALIBRATED`** | 4+ surveyed reference points configured | Physical speed ($\text{km/h}$) | `"calibrated"` |
| **`DEMO_CALIBRATION`** | Predefined reference for `traffic.mp4` | Estimated physical speed ($\text{km/h}$) | `"demo_calibration"` |
| **`UNCALIBRATED`** | Unknown camera / no homography | `speed_kmh = NaN`; pixel speed ($\text{px/s}$) retained | `"uncalibrated"` |
