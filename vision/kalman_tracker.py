"""
================================================================================
ROAD SENSE AI - GROUND-PLANE KINEMATIC KALMAN FILTER
MODULE: 2D GROUND-PLANE KALMAN FILTER & OCCLUSION MITIGATION
================================================================================

This module implements a 2D ground-plane Kalman filter for tracking vehicles in
metric world coordinates (X, Y) and estimating smooth velocity (vX, vY).

MATHEMATICAL FOUNDATION:
1. STATE VECTOR & CONTINUOUS-ACCELERATION MOTION MODEL:
   State: x = [X, Y, vX, vY]^T  (Position in meters, Velocity in m/s)
   Measurement: z = [X_meas, Y_meas]^T

   State Transition:
       x_{t+1} = F(dt) * x_t + w_t
       where F(dt) = [ [1, 0, dt,  0],
                       [0, 1,  0, dt],
                       [0, 0,  1,  0],
                       [0, 0,  0,  1] ]

2. PROCESS & MEASUREMENT COVARIANCES:
   - Process Noise Q(dt): Continuous white-noise acceleration with sigma_a = 1.5 m/s^2.
   - Measurement Noise R: sigma_pos = 0.50 m (ground-plane projection uncertainty).

3. HEAVY TRAFFIC OCCLUSION & OUTLIER GATING:
   In dense traffic, when a vehicle's bounding box is occluded by a vehicle in front,
   its bottom contact point is artificially shifted upward.
   - Gating: If innovation ||z - H*x|| exceeds gate threshold or is_occluded=True,
     the tracker coast-predicts using learned momentum rather than corrupting the state.

4. VELOCITY OUTPUT:
   Physical velocity is directly derived from state:
       speed_kmh = sqrt(vX^2 + vY^2) * 3.6
================================================================================
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any


class VehicleKalmanFilter:
    """
    2D Constant-Velocity Kalman Filter for a single vehicle track on the ground plane.
    """

    def __init__(
        self,
        init_x: float,
        init_y: float,
        init_timestamp: float,
        sigma_a: float = 1.5,
        sigma_pos: float = 0.50
    ):
        self.last_timestamp = init_timestamp
        self.sigma_a = sigma_a
        self.sigma_pos = sigma_pos
        self.observation_count = 1
        self.consecutive_occlusions = 0

        # State vector: [X, Y, vX, vY]
        self.x = np.array([init_x, init_y, 0.0, 0.0], dtype=np.float64)

        # Initial covariance P
        self.P = np.diag([sigma_pos ** 2, sigma_pos ** 2, 4.0 ** 2, 4.0 ** 2]).astype(np.float64)

        # Measurement matrix H (maps state to [X, Y])
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Measurement noise R
        self.R = np.diag([sigma_pos ** 2, sigma_pos ** 2]).astype(np.float64)

    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Performs Kalman time update (prediction) over elapsed time dt."""
        if dt <= 0:
            dt = 1e-3

        # State transition matrix F
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Process noise covariance Q(dt) (discrete continuous-time white noise model)
        dt2 = dt ** 2
        dt3 = dt ** 3 / 2.0
        dt4 = dt ** 4 / 4.0
        q = self.sigma_a ** 2

        Q = q * np.array([
            [dt4, 0.0, dt3, 0.0],
            [0.0, dt4, 0.0, dt3],
            [dt3, 0.0, dt2, 0.0],
            [0.0, dt3, 0.0, dt2]
        ], dtype=np.float64)

        # Update state and covariance predictions
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x, self.P

    def update(
        self,
        meas_x: float,
        meas_y: float,
        timestamp: float,
        is_occluded: bool = False
    ) -> Dict[str, Any]:
        """
        Updates the filter with a new ground-plane measurement at timestamp.
        Applies occlusion gating and innovation bounding.
        """
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        self.observation_count += 1

        # 1. Prediction step
        self.predict(dt)

        z = np.array([meas_x, meas_y], dtype=np.float64)

        # 2. Innovation & Gating
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        inv_S = np.linalg.inv(S)

        # Mahalanobis distance squared
        maha_dist_sq = float(y.T @ inv_S @ y)
        is_gated_outlier = (maha_dist_sq > 16.0) or is_occluded  # 4-sigma gate or active occlusion

        if is_gated_outlier and self.observation_count > 2:
            # Heavy traffic occlusion / anomaly: Coast on prediction, expand covariance
            self.consecutive_occlusions += 1
            self.P[:2, :2] += np.eye(2) * 0.20
            mode = "COAST_PREDICTION (Occluded/Gated)"
        else:
            # Standard measurement update
            self.consecutive_occlusions = 0
            K = self.P @ self.H.T @ inv_S
            self.x = self.x + K @ y
            I_KH = np.eye(4) - K @ self.H
            # Joseph form covariance update for numerical stability
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
            mode = "MEASUREMENT_UPDATE"

        # Speed in km/h from estimated velocity vector
        vx = self.x[2]
        vy = self.x[3]
        speed_mps = float(np.sqrt(vx ** 2 + vy ** 2))
        speed_kmh = round(speed_mps * 3.6, 2)

        return {
            "kalman_x": round(float(self.x[0]), 3),
            "kalman_y": round(float(self.x[1]), 3),
            "kalman_vx": round(float(vx), 3),
            "kalman_vy": round(float(vy), 3),
            "kalman_speed_kmh": speed_kmh,
            "filter_mode": mode
        }


class GroundPlaneKalmanTracker:
    """
    Manages multi-vehicle 2D ground-plane Kalman filters across an observation session.
    """

    def __init__(self, sigma_a: float = 1.5, sigma_pos: float = 0.50):
        self.sigma_a = sigma_a
        self.sigma_pos = sigma_pos
        self.filters: Dict[int, VehicleKalmanFilter] = {}

    def process_observation(
        self,
        track_id: int,
        world_x: float,
        world_y: float,
        timestamp: float,
        is_occluded: bool = False
    ) -> Dict[str, Any]:
        """Processes an observation for a given track_id."""
        if not np.isfinite(world_x) or not np.isfinite(world_y):
            return {
                "kalman_x": np.nan,
                "kalman_y": np.nan,
                "kalman_vx": np.nan,
                "kalman_vy": np.nan,
                "kalman_speed_kmh": np.nan,
                "filter_mode": "INVALID_MEASUREMENT"
            }

        if track_id not in self.filters:
            # Initialize new Kalman filter for newly spawned track
            self.filters[track_id] = VehicleKalmanFilter(
                init_x=world_x,
                init_y=world_y,
                init_timestamp=timestamp,
                sigma_a=self.sigma_a,
                sigma_pos=self.sigma_pos
            )
            return {
                "kalman_x": round(float(world_x), 3),
                "kalman_y": round(float(world_y), 3),
                "kalman_vx": 0.0,
                "kalman_vy": 0.0,
                "kalman_speed_kmh": np.nan,  # First observation speed is strictly NaN
                "filter_mode": "INITIALIZED"
            }

        return self.filters[track_id].update(world_x, world_y, timestamp, is_occluded)
