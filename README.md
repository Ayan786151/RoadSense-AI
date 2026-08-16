# 🚦 RoadSense AI — Traffic Intelligence & Risk Analytics

An end-to-end **AI-powered traffic risk prediction and priority intelligence system** combining synthetic city simulation, supervised machine learning, real-time computer vision, and interactive analytics dashboards.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RoadSense AI Pipeline                        │
├──────────────┬──────────────┬───────────────┬───────────────────────┤
│  SIMULATION  │   FEATURES   │   ML MODEL    │     DASHBOARD         │
│              │              │               │                       │
│  sim.py      │  temporal_   │  train_       │  app.py               │
│  (50 zones   │  features.py │  model.py     │  simulation_          │
│   52 weeks)  │  (lag, roll, │  (LR vs RF,   │  dashboard.py         │
│              │   OLS trend) │   chrono       │  (Streamlit +         │
│              │              │   split)      │   Plotly + Map)       │
├──────────────┴──────────────┴───────────────┴───────────────────────┤
│                     COMPUTER VISION PIPELINE                        │
│  vehicle_detector.py → movement_analyzer.py → feature_fusion.py     │
│  (YOLO + ByteTrack)   (Homography + Kalman)   (Density + Movement)  │
│                                                                      │
│  enhancement.py        calibration.py          kalman_tracker.py     │
│  (Night/Low-Light)     (Perspective Transform)  (2D Ground Kalman)   │
├──────────────────────────────────────────────────────────────────────┤
│                      LIVE DATA & PRIORITY                            │
│  temporal_store.py → live_inference.py → priority_engine.py          │
│  (Historical Store)   (Safety Adapter)    (Risk × Exposure × Trend) │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

Execute these steps **in order**:

```bash
# Step 1: Generate synthetic traffic simulation (50 zones × 52 weeks)
python sim.py

# Step 2: Engineer temporal features (lag, rolling, OLS trends)
python analysis/temporal_features.py

# Step 3: Train the ML risk model (Logistic Regression vs Random Forest)
python models/train_model.py

# Step 4: Compute priority rankings
python priority/priority_engine.py

# Step 5: Launch the interactive dashboard
streamlit run app.py
```

### 3. Run the Vision Pipeline (Optional)

```bash
# Detect vehicles and track trajectories
python -m vision.vehicle_detector --video videos/traffic.mp4 --session session_001 --show

# Analyze movement and estimate speed
python -m vision.movement_analyzer --session session_001 --show-calibration

# Extract traffic features
python -m vision.traffic_analyzer --session session_001

# Fuse vision and movement features
python -m vision.feature_fusion --session session_001
```

---

## 📁 Project Structure

```
traffic_sim-main/
├── app.py                          # Main Streamlit entrypoint
├── sim.py                          # Synthetic city traffic simulator
├── requirements.txt                # Pinned Python dependencies
│
├── analysis/
│   ├── temporal_features.py        # Temporal feature engineering (lag, rolling, OLS)
│   └── validate_sessions.py        # Session data validator
│
├── dashboard/
│   └── simulation_dashboard.py     # Interactive Streamlit dashboard
│
├── data/                           # Generated datasets (not tracked in git)
│   ├── simulation_temporal_features.csv
│   ├── location_mapping.csv
│   ├── calibration_config.json
│   └── sessions/                   # Per-session CCTV data
│
├── docs/
│   ├── speed_estimation.md
│   ├── temporal_architecture.md
│   └── temporal_feature_engine.md
│
├── live_data/
│   ├── ingest_session.py           # Session ingestion pipeline
│   ├── live_inference.py           # Live ML inference safety adapter
│   ├── temporal_feature_engine.py  # Live temporal feature computation
│   ├── temporal_store.py           # Historical data accumulation
│   └── traffic_observation_builder.py
│
├── models/
│   ├── train_model.py              # ML model training & evaluation
│   └── best_risk_model.pkl         # Trained Random Forest pipeline
│
├── priority/
│   └── priority_engine.py          # Risk × Exposure × Trend priority scoring
│
├── vision/
│   ├── vehicle_detector.py         # YOLO vehicle detection + ByteTrack
│   ├── movement_analyzer.py        # Homography speed estimation + Kalman
│   ├── traffic_analyzer.py         # Traffic feature extraction
│   ├── feature_fusion.py           # Multi-modal vision feature fusion
│   ├── calibration.py              # Perspective homography calibration
│   ├── enhancement.py              # Adaptive night/low-light enhancement
│   └── kalman_tracker.py           # 2D ground-plane Kalman filter
│
└── videos/                         # Input traffic footage
```

---

## 🧠 Core Concepts

### Risk Model
- **Target**: `P(incident_occurred = 1)` — weekly incident probability
- **Features**: 40 features across current conditions, lag-1, 4-week rolling, OLS trends
- **Training**: Chronological split (Weeks 5-40 train, 41-46 val, 47-52 test)
- **Models**: Logistic Regression baseline vs Random Forest (selected)
- **Leakage Prevention**: Strict `shift(1)` for lags, `min_periods=4` for rolling

### Priority Index
```
Priority = 0.40 × Risk + 0.25 × Population Exposure + 0.20 × Vehicle Exposure + 0.15 × Temporal Trend
```

### Vision Pipeline
- **Detection**: YOLOv11n with COCO vehicle classes (car, motorcycle, bus, truck)
- **Tracking**: ByteTrack with persistent IDs across frames
- **Speed**: 4-point perspective homography → ground-plane Kalman filter → km/h
- **Occlusion**: Bottom-contact point gating with Mahalanobis distance

---

## 📊 Dashboard Features

| Feature | Description |
|:--------|:------------|
| **KPI Cards** | Risk probability, congestion, speed, density with WoW deltas |
| **52-Week Timeline** | Interactive Plotly chart with congestion, speed, and risk curves |
| **Leaderboard** | All 50 zones ranked by composite priority score |
| **Geographic Map** | Scatter mapbox with risk-colored zones on dark basemap |
| **What-If Sandbox** | Real-time ML sensitivity testing with parameter sliders |
| **Diagnostics** | Full JSON observation record per zone-week |

---

## 📄 License

This project is provided as-is for educational and research purposes.
