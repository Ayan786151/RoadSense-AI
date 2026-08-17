# 🚦 RoadSense AI — Intelligent Traffic Risk & Civic Social Service Hub

An end-to-end **AI-powered urban traffic intelligence system** built for civic social service, combining computer vision, predictive machine learning, adaptive traffic light optimization, and environmental carbon reduction analytics across 50 municipal zones.

---

## 🌟 Key Capabilities & Social Impact

1. **AI Safety Briefings (Llama-3.3-70B / Gemini-2.5)**: Automated, plain-English executive briefings for municipal traffic controllers focusing on citizen safety, vulnerable pedestrian protection, and proactive risk mitigation.
2. **Adaptive Signal Control (DTSC)**: Real-time calculation of optimal green-light timing based on queue density, vehicle kinematics, and weather conditions.
3. **Civic Eco-Savings & Carbon Reduction**: Quantifies tonnes of CO2 prevented, liters of fuel saved, and tree-equivalent absorption resulting from dynamic traffic optimization.
4. **Supervised ML Risk Forecaster**: Leakage-free panel modeling predicting incident probability `P(incident_occurred = 1)` across 52 weeks and 50 urban zones.
5. **Real-Time CCTV Computer Vision**: YOLOv11 vehicle detection, ByteTrack ID persistence, and 4-point perspective homography ground-plane velocity estimation.
6. **Geographic Priority Heatmaps**: Interactive Mapbox visualizations ranking municipal zones by composite public-risk exposure.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           RoadSense AI — Civic Control Architecture                     │
├──────────────────┬────────────────────┬────────────────────┬────────────────────────────┤
│   SIMULATION     │  TEMPORAL FEATURES │    ML RISK MODEL   │     ADAPTIVE INTELLIGENCE  │
│                  │                    │                    │                            │
│  sim.py          │  temporal_         │  train_model.py    │  intelligence/             │
│  (50 zones,      │  features.py       │  (Random Forest,   │  - signal_co2.py           │
│   52 weeks panel)│  (lag, rolling,    │   chronological    │  - llm_briefing.py         │
│                  │   OLS slopes)      │   split)           │  (Llama-3.3 / Gemini-2.5)  │
├──────────────────┴────────────────────┴────────────────────┴────────────────────────────┤
│                              COMPUTER VISION PIPELINE                                   │
│  vehicle_detector.py   →   movement_analyzer.py   →   feature_fusion.py                 │
│  (YOLO + ByteTrack)        (Homography Velocity)       (Live Congestion & Telemetry)    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                               INTERACTIVE CONTROL HUB                                   │
│  app.py  &  dashboard/simulation_dashboard.py  (Streamlit + Plotly + Mapbox Dark)       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Control Hub

```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📁 Repository Structure

```
traffic_sim/
├── app.py                          # Streamlit master entrypoint & CCTV hub
├── sim.py                          # 50-Zone × 52-Week synthetic urban simulator
├── requirements.txt                # Pinned dependencies
│
├── intelligence/                   # NEW: Adaptive control & LLM engine
│   ├── signal_co2.py               # Adaptive green light & CO2 savings calculator
│   └── llm_briefing.py             # Multi-key Groq / Gemini executive brief engine
│
├── dashboard/
│   └── simulation_dashboard.py     # Multi-tab analytics lab & Mapbox leaderboard
│
├── analysis/
│   ├── temporal_features.py        # Leakage-free lag, rolling, and OLS trend features
│   └── validate_sessions.py        # CCTV session integrity auditor
│
├── models/
│   ├── train_model.py              # ML risk model trainer & evaluator
│   └── best_risk_model.pkl         # Trained Random Forest pipeline
│
├── priority/
│   ├── priority_engine.py          # Risk × Exposure × Trend composite scoring
│   └── intervention_engine.py      # Municipal intervention recommendation rules
│
├── vision/
│   ├── vehicle_detector.py         # YOLO detection & ByteTrack tracking
│   ├── movement_analyzer.py        # Homography perspective speed estimation
│   ├── calibration.py              # 4-point quad homography matrix
│   ├── enhancement.py              # Low-light CLAHE & night vision preprocessor
│   └── feature_fusion.py           # Multi-modal vision feature aggregation
│
├── data/                           # Panel datasets and CCTV telemetry sessions
│   ├── sessions/                   # session_001, session_002, session_003
│   ├── simulation_temporal_features.csv
│   └── location_mapping.csv
│
└── videos/                         # Sample traffic CCTV footage
```

---

## 📄 License
Provided for smart city innovation, civic safety research, and hackathon evaluation.
