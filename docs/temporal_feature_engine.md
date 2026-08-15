# RoadSense AI - Temporal Feature Engine Architecture

## Overview & Purpose
The **Temporal Feature Engine** ([`live_data/temporal_feature_engine.py`](file:///c:/Users/asus/Downloads/traffic_sim-main%20%283%29/traffic_sim-main/live_data/temporal_feature_engine.py)) transforms high-frequency camera observation sessions from [`data/temporal_traffic_store.csv`](file:///c:/Users/asus/Downloads/traffic_sim-main%20%283%29/traffic_sim-main/data/temporal_traffic_store.csv) into calendar-week aggregations and engineers temporal intelligence features (Lag-1, Rolling 4-Week, Week-over-Week changes, and 4-Week OLS Linear Trends) while strictly enforcing **zero temporal lookahead leakage** and **zero data fabrication**.

---

## 1. Calendar Week Identification
- High-frequency video observation sessions (e.g. `session_001`, `session_003`) recorded on the same date (`2026-08-15`) share the same municipal observation window (`week_id = 1` / ISO Week 33).
- **Rule**: Multiple video files recorded on the same calendar date/week do **NOT** constitute separate historical weeks.
- The feature engine groups records by `(location_id, camera_id, week_id)` to form unified weekly observations.

---

## 2. Weekly Traffic Signal Aggregation
High-frequency frame-level signals across sessions within the week are aggregated into statistical distributions:
- **`vehicle_density`**: Mean normalized frame occupancy proxy ($0\text{--}100\%$)
- **`congestion`**: Mean fused camera congestion score ($0\text{--}100$)
- **`average_speed`**: Mean pixel speed in pixels/second (camera-relative)
- **`dominant_traffic_state`**: Mode state (`LOW`, `MODERATE`, `HIGH`, `SEVERE`)
- **State Proportions**: Percentage of observations in `LOW`, `MODERATE`, `HIGH`, and `SEVERE` states.

---

## 3. Mathematical Feature Engineering & Leakage Protection

All temporal momentum features strictly reference **past calendar weeks only** ($t-1, t-2, t-3, t-4$). Current week $t$ and future weeks are mathematically excluded from all historical baselines.

```
Time Axis (Weeks):
[  Week t-4  ]  [  Week t-3  ]  [  Week t-2  ]  [  Week t-1  ]  |  [  Week t (Current)  ]
 \___________________________________________________________/   |
               Historical Window (Past Only)                     | Current Observation
```

### A. Lag-1 Features ($t-1$)
- **Formula**: $y_{t-1} = \text{shift}(1)$
- **Features**: `previous_week_vehicle_density`, `previous_week_congestion`, `previous_week_average_speed`, `previous_week_traffic_pressure`.
- **Requirement**: Requires $\ge 2$ distinct calendar weeks. Week 1 evaluates to `NaN` (`WARMUP`).

### B. Rolling 4-Week Historical Averages ($t-4 \dots t-1$)
- **Formula**:
  $$\text{Rolling4W}(y_t) = \frac{1}{4} \sum_{k=1}^{4} y_{t-k} = \frac{y_{t-1} + y_{t-2} + y_{t-3} + y_{t-4}}{4}$$
- **Features**: `rolling_4_week_avg_vehicle_density`, `rolling_4_week_avg_congestion`, `rolling_4_week_avg_speed`, `rolling_4_week_avg_traffic_pressure`.
- **Requirement**: `min_periods = 4` enforced. Weeks 1 to 4 evaluate to `NaN` (`WARMUP`).

### C. Week-over-Week (WoW) Changes
- **Absolute Delta**: $\Delta y_t = y_t - y_{t-1}$
- **Percentage Delta**:
  $$\% \Delta y_t = \left( \frac{y_t - y_{t-1}}{y_{t-1} + 10^{-4}} \right) \times 100$$
- **Features**: `vehicle_density_change`, `congestion_change`, `speed_change`, `traffic_pressure_change`, `vehicle_density_pct_change`, `congestion_pct_change`, `speed_pct_change`.
- **Requirement**: Requires $\ge 2$ distinct calendar weeks. Week 1 evaluates to `NaN` (`WARMUP`).

### D. Four-Week OLS Linear Trend Slopes
- **Closed-Form Formula**:
  Fitting an ordinary least-squares line through $y = [y_{t-4}, y_{t-3}, y_{t-2}, y_{t-1}]$ at $x = [0, 1, 2, 3]$ yields the exact closed-form slope:
  $$\text{Slope} = \frac{3.0 \cdot y_{t-1} + 1.0 \cdot y_{t-2} - 1.0 \cdot y_{t-3} - 3.0 \cdot y_{t-4}}{10.0}$$
- **Features**: `congestion_trend_4w`, `vehicle_density_trend_4w`, `speed_trend_4w`.
- **Requirement**: Requires $\ge 5$ distinct calendar weeks. Weeks 1 to 4 evaluate to `NaN` (`WARMUP`).

---

## 4. Warm-Up Policy & External Feeds

### Temporal Maturity States
- **`WARMUP`**: $< 2$ distinct weeks in temporal store (Lag-1, Rolling, and Trends evaluate to `NaN`).
- **`PARTIAL_HISTORY`**: $2 \dots 4$ distinct weeks in store (Lag-1 and WoW active; 4-Week Rolling and Trends in warmup).
- **`READY`**: $\ge 5$ distinct weeks in store (all camera temporal features fully active).

### External Feeds (Police Incident Logs & Violation Cameras)
The 10 incident/violation features (`previous_week_red_light_violations`, `previous_week_incident_count`, `rolling_4_week_incident_count`, `incident_trend_4w`, etc.) require municipal data feeds and are explicitly tracked as **`MISSING_EXTERNAL_DATA`** (`NaN`) rather than fabricating synthetic incident values.
