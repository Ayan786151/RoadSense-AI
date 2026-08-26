# 📘 RoadSense AI — Comprehensive Data Science & AI Student Handbook
> **A complete, self-paced learning guide to every library, algorithm, mathematical concept, and code snippet in RoadSense AI.**

---

## 📑 Table of Contents
1. [🌟 Introduction & Project Overview](#1-introduction--project-overview)
2. [🧰 Technology & Library Matrix](#2-technology--library-matrix)
3. [👁️ Module 1: Computer Vision & Video Processing (`OpenCV`, `Ultralytics`, `PyTorch`)](#3-module-1-computer-vision--video-processing-opencv-ultralytics-pytorch)
4. [🎲 Module 2: Synthetic Panel Simulation & Probabilities (`NumPy`)](#4-module-2-synthetic-panel-simulation--probabilities-numpy)
5. [📈 Module 3: Temporal Feature Engineering (`Pandas`, `NumPy`)](#5-module-3-temporal-feature-engineering-pandas-numpy)
6. [🤖 Module 4: Supervised Machine Learning & Evaluation (`Scikit-Learn`, `Joblib`)](#6-module-4-supervised-machine-learning--evaluation-scikit-learn-joblib)
7. [🚦 Module 5: Civic Decision Intelligence & Eco-Optimization](#7-module-5-civic-decision-intelligence--eco-optimization)
8. [📊 Module 6: Web Visualization & Generative AI (`Streamlit`, `Plotly`, `OpenAI`)](#8-module-6-web-visualization--generative-ai-streamlit-plotly-openai)
9. [🎓 Student Self-Test & Interview Preparation Q&A](#9-student-self-test--interview-preparation-qa)

---

# 1. 🌟 Introduction & Project Overview

### What is RoadSense AI?
RoadSense AI is an end-to-end **Intelligent Urban Traffic Risk & Civic Social Service Hub**. It takes raw CCTV traffic footage and historical municipal traffic data across 50 urban zones over 52 weeks, processes them using Data Science and Machine Learning, and delivers three core outcomes:
1. **Real-Time Vision Telemetry**: Detects vehicles, tracks their trajectories, and measures real-world speeds in km/h from camera pixels.
2. **Accident Risk Forecasting**: Predicts the weekly probability of a traffic incident $P(\text{incident\_occurred} = 1)$ using supervised machine learning without lookahead data leakage.
3. **Adaptive Signal Control & Carbon Reduction**: Dynamically adjusts traffic light timings to clear queues and computes fuel and $\text{CO}_2$ emissions prevented.

---

# 2. 🧰 Technology & Library Matrix

| Library | Version | Role in this Project | Key Project Files |
| :--- | :--- | :--- | :--- |
| **`numpy`** | `~=2.0` | Matrix math, sigmoid functions, OLS regression slope, coordinate vectorization. | `sim.py`, `analysis/temporal_features.py` |
| **`pandas`** | `~=2.2` | Panel time-series dataframes, lag features (`shift`), rolling aggregations, joins. | `analysis/temporal_features.py`, `models/train_model.py` |
| **`scikit-learn`** | `~=1.5` | Preprocessing (`ColumnTransformer`, `StandardScaler`), ML models (`RandomForest`), evaluation metrics. | `models/train_model.py` |
| **`opencv-python`** (`cv2`) | `~=4.10` | Video frame decoding, CIELAB color space, CLAHE contrast enhancement, Homography matrices. | `vision/enhancement.py`, `vision/calibration.py` |
| **`ultralytics`** | `~=8.3` | State-of-the-art YOLOv11 deep learning object detector and ByteTrack tracker. | `vision/vehicle_detector.py` |
| **`torch`** | `~=2.4` | PyTorch deep learning backend running YOLO neural network inference. | `vision/vehicle_detector.py` |
| **`joblib`** | `~=1.4` | Model serialization (`.pkl` dumping/loading) for fast, cached inference. | `models/train_model.py`, `dashboard/simulation_dashboard.py` |
| **`streamlit`** | `~=1.38` | Interactive web dashboard, state management, reactive user interface. | `app.py`, `dashboard/simulation_dashboard.py` |
| **`plotly`** | `~=5.24` | GPU-accelerated interactive charts, dual-axis telemetry, and Carto-Darkmatter Mapbox heatmaps. | `dashboard/simulation_dashboard.py`, `dashboard/city_map.py` |
| **`openai`** | `>=1.0` | LLM client API (connecting to Groq / Gemini) to generate natural language safety briefings. | `intelligence/llm_briefing.py` |

---

# 3. 👁️ Module 1: Computer Vision & Video Processing (`OpenCV`, `Ultralytics`, `PyTorch`)

## 3.1. Video Ingestion & Decoding (`OpenCV`)
Videos are sequences of static image frames. OpenCV reads them sequentially:
```python
# vision/vehicle_detector.py
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) # e.g. 30 frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    success, frame = cap.read() # frame is a NumPy ndarray of shape (H, W, 3) in BGR format
    if not success:
        break
```

---

## 3.2. Adaptive Lighting Enhancement (`CIELAB` & `CLAHE`)
Standard RGB/BGR images mix color and lightness in every channel. If you simply brighten an RGB image, colors get washed out.
* **Solution**: Convert BGR to **CIELAB** color space.
  * $L^*$: Luminance (lightness/intensity channel).
  * $a^*$: Red-Green chromaticity.
  * $b^*$: Yellow-Blue chromaticity.
* **CLAHE (Contrast Limited Adaptive Histogram Equalization)** is applied **only to the $L^*$ channel**. It divides the image into an $8 \times 8$ grid of tiles and equalizes contrast locally while clipping high peaks to prevent amplifying camera noise.

```python
# vision/enhancement.py
def enhance_low_light_clahe(frame: np.ndarray, clip_limit: float = 2.5, tile_grid_size=(8, 8)) -> np.ndarray:
    # 1. Convert BGR to CIELAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 2. Apply CLAHE only to the Luminance channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # 3. Merge back and convert to BGR
    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr
```

---

## 3.3. Deep Learning Vehicle Detection & ByteTrack (`Ultralytics YOLO` & `PyTorch`)
YOLOv11 uses a single Convolutional Neural Network pass to predict bounding boxes and class probabilities simultaneously.
* **ByteTrack** matches bounding boxes across consecutive frames using Kalman Filters and IoU (Intersection-over-Union) association to assign persistent integer IDs (`track_id`).

```python
# vision/vehicle_detector.py
from ultralytics import YOLO

# Load model weights (PyTorch deep learning model)
model = YOLO("yolo11s.pt")

# Track objects across frames
results = model.track(
    processed_frame,
    persist=True,             # Retain IDs from previous frames
    tracker="bytetrack.yaml", # Use ByteTrack association
    conf=0.35,                # Confidence threshold (35%)
    imgsz=640, 
    verbose=False
)

# Extract detections
result = results[0]
boxes = result.boxes
for i in range(len(boxes)):
    class_id = int(boxes.cls[i].item())
    xyxy = boxes.xyxy[i].cpu().numpy().astype(int) # [x1, y1, x2, y2]
    track_id = int(boxes.id[i].item()) if boxes.id is not None else None
```

---

## 3.4. 4-Point Planar Perspective Homography (`OpenCV` & `NumPy`)
In a CCTV camera, vehicles that are farther away look smaller and move fewer pixels per second than vehicles close to the camera (perspective distortion).
* **Homography Matrix $H$**: A $3 \times 3$ matrix that maps points on a 2D camera image plane $(u, v)$ to real-world ground coordinates $(X, Y)$ in meters.

$$s \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

```python
# vision/calibration.py
# Given 4 known pixel points and their measured real-world coordinates (in meters)
img_pts = np.float32([[200, 450], [1080, 450], [1200, 700], [80, 700]])
wld_pts = np.float32([[0, 30], [15, 30], [15, 0], [0, 0]]) # 15m width x 30m length road patch

# Compute the 3x3 transformation matrix
H_matrix = cv2.getPerspectiveTransform(img_pts, wld_pts)
```

### Speed Calculation (km/h)
```python
# vision/movement_analyzer.py
# 1. Road-contact point is bottom-center of bounding box: ((x1+x2)/2, y2)
# 2. Transform pixel point to real-world meters using H
# 3. Compute Euclidean physical displacement:
delta_d = np.sqrt((X_current - X_prev)**2 + (Y_current - Y_prev)**2) # meters
delta_t = t_current - t_prev                                         # seconds

speed_mps = delta_d / delta_t                                        # meters / second
speed_kmh = speed_mps * 3.6                                          # km/h
```

---

# 4. 🎲 Module 2: Synthetic Panel Simulation & Probabilities (`NumPy`)

In real-world data science, training datasets are structured as **Panel Data** (multiple entities observed over consecutive time periods).

In `sim.py`, 50 urban zones are simulated over 52 weeks (2,600 total observations).

### Converting Latent Risk Scores to Incident Probabilities:
Raw urban variables (congestion, speeding, rain, violations) are combined into a linear log-odds equation and converted to a probability using the **Logistic Sigmoid Function**:

$$\text{logit} = \beta_0 + \beta_1 \cdot \text{Congestion} + \beta_2 \cdot \text{Violations} - \beta_3 \cdot \text{Speed} + \dots$$
$$P(\text{incident\_occurred} = 1) = \frac{1}{1 + e^{-\text{logit}}}$$

```python
# sim.py
def calculate_incident_probability(row):
    logit = -3.2 \
        + 0.035 * row["congestion"] \
        + 0.040 * row["red_light_violations"] \
        - 0.015 * row["average_speed"] \
        + 0.60  * (1 if row["weather"] == "Heavy Rain" else 0)
    
    prob = 1.0 / (1.0 + np.exp(-logit))
    return prob
```

---

# 5. 📈 Module 3: Temporal Feature Engineering (`Pandas`, `NumPy`)

> [!IMPORTANT]
> **The Golden Rule of Time-Series Machine Learning**: **Never allow future or current-week target outcomes to bleed into training features (Data Leakage).**

When forecasting if an accident will happen in Week $t$, the model can **only** know what happened in past weeks ($t-1, t-2, t-3, t-4$).

```
Timeline:    [Week t-4]  →  [Week t-3]  →  [Week t-2]  →  [Week t-1]  ||  [Week t (Current)]
Window:     |──────────────── Historical 4-Week Context ─────────────|  ||  Predict Incident?
```

## 5.1. 1-Week Lag Features ($t-1$)
```python
# analysis/temporal_features.py
grouped = df.groupby("zone_id")

# Pull previous week's traffic state into the current row
df["previous_week_vehicle_density"] = grouped["vehicle_density"].shift(1)
df["previous_week_congestion"] = grouped["congestion"].shift(1)
df["previous_week_average_speed"] = grouped["average_speed"].shift(1)
```

## 5.2. 4-Week Rolling Averages ($t-4$ to $t-1$)
```python
# analysis/temporal_features.py
# First shift by 1 to exclude current week t, then compute rolling mean of past 4 weeks
shifted_veh = grouped["vehicle_density"].shift(1)
df["rolling_4_week_avg_vehicle_density"] = (
    shifted_veh.groupby(df["zone_id"])
    .rolling(window=4, min_periods=4)
    .mean()
    .round(2)
    .values
)
```

## 5.3. 4-Week Linear Trend Slopes (OLS Regression)
Is congestion getting worse or better over the last month?
We calculate the slope of the best-fit line across $[y_{t-4}, y_{t-3}, y_{t-2}, y_{t-1}]$:

$$\text{Slope} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{3y_{t-1} + y_{t-2} - y_{t-3} - 3y_{t-4}}{10}$$

```python
# analysis/temporal_features.py
def calculate_4w_slope(s_lag1, s_lag2, s_lag3, s_lag4):
    return (3.0 * s_lag1 + s_lag2 - s_lag3 - 3.0 * s_lag4) / 10.0

# If slope > 0: Congestion is trending upwards (getting worse)
# If slope < 0: Congestion is improving
df["congestion_trend_4w"] = calculate_4w_slope(cong_lag1, cong_lag2, cong_lag3, cong_lag4)
```

---

# 6. 🤖 Module 4: Supervised Machine Learning & Evaluation (`Scikit-Learn`, `Joblib`)

## 6.1. Chronological Data Split (Time-Series Splitting)
Standard `train_test_split(shuffle=True)` randomly shuffles rows. In temporal data, this causes the model to "cheat" by using future patterns to predict past events.
* **Correct Practice**: Split strictly chronologically by time.

```python
# models/train_model.py
# Weeks 1-4 dropped (warm-up window for 4-week rolling features)
train_df = df[(df["week"] >= 5) & (df["week"] <= 40)]  # 1,800 records (Train)
val_df   = df[(df["week"] >= 41) & (df["week"] <= 46)] # 300 records (Validation)
test_df  = df[(df["week"] >= 47) & (df["week"] <= 52)] # 300 records (Test)
```

---

## 6.2. Preprocessing with `ColumnTransformer` & `Pipeline`
Machine learning models require numbers. Categorical text (like `"Commercial_Downtown"`, `"Rain"`) must be converted to numbers, and numerical features must be normalized.

```python
# models/train_model.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_feature_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_feature_cols)
    ]
)

# Bundle preprocessing and model into an airtight pipeline
risk_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced", # Compensates for rare accident events
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# Train the entire pipeline in one call
risk_pipeline.fit(X_train, y_train)
```

---

## 6.3. Evaluation Metrics Explained for Students

```python
# models/train_model.py
y_probs = risk_pipeline.predict_proba(X_test)[:, 1] # Probability between 0.0 and 1.0
y_preds = (y_probs >= 0.50).astype(int)              # Binary decision (0 or 1)
```

| Metric | What It Measures | Why It Matters in this Project |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Can be misleading when accidents are rare (e.g. 90% accuracy by just guessing "no accident"). |
| **Precision** | $\frac{TP}{TP + FP}$ | When the model warns of an accident risk, how often is it right? |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Out of all real accidents, what percentage did the model catch? |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of Precision and Recall. |
| **ROC-AUC** | Area Under ROC Curve ($0.5 \to 1.0$) | Measures how well the model separates safe zones from dangerous zones across all thresholds. |
| **PR-AUC** | Area Under Precision-Recall Curve | Best metric for imbalanced data. |
| **Brier Score** | $\frac{1}{N}\sum (P_i - y_i)^2$ | Measures **probability calibration**. Lower is better ($0.0 = \text{perfect}$). |

---

## 6.4. Feature Importance
Which variables drive accident risk the most?
```python
clf = risk_pipeline.named_steps["classifier"]
importances = clf.feature_importances_

df_imp = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)
```

---

## 6.5. Saving & Loading Models with `joblib`
```python
import joblib

# Save trained pipeline to disk
joblib.dump(risk_pipeline, "models/best_risk_model.pkl")

# Load in production/app without retraining
loaded_model = joblib.load("models/best_risk_model.pkl")
predictions = loaded_model.predict_proba(new_data)[:, 1]
```

---

# 7. 🚦 Module 5: Civic Decision Intelligence & Eco-Optimization

## 7.1. Dynamic Traffic Signal Control (DTSC)
Instead of static 30-second timers, the green light adapts dynamically to queue length and congestion:

```python
# intelligence/signal_co2.py
def compute_optimal_signal_timing(congestion, vehicle_density, average_speed, base_green=35):
    congestion_factor = 1.0 + (congestion / 100.0) * 0.8
    density_factor    = 1.0 + (min(vehicle_density, 100) / 100.0) * 0.4
    speed_factor      = max(0.7, 1.0 - (average_speed / 80.0) * 0.3)
    
    recommended = base_green * congestion_factor * density_factor * speed_factor
    return int(np.clip(recommended, 15, 120)) # Clamp between 15s and 120s
```

## 7.2. Fuel & $\text{CO}_2$ Savings Estimation
Every hour of stopped idling consumes $\approx 0.8\text{ liters}$ of fuel per vehicle.
Each liter of fuel burned emits $\approx 2.31\text{ kg of }\text{CO}_2$:

$$\text{Delay Saved (hours)} = \frac{\Delta \text{Delay (seconds)} \times \text{Vehicles}}{3600}$$
$$\text{Fuel Saved (liters)} = \text{Delay Saved} \times 0.8$$
$$\text{CO}_2\text{ Prevented (kg)} = \text{Fuel Saved} \times 2.31$$

---

# 8. 📊 Module 6: Web Visualization & Generative AI (`Streamlit`, `Plotly`, `OpenAI`)

## 8.1. Streamlit Interactive Dashboard Architecture
Streamlit turns Python code into a reactive web app.
* `@st.cache_data`: Caches dataframe reading so CSVs are not re-read on every click.
* `@st.cache_resource`: Caches machine learning models and network connections in memory.

```python
# dashboard/simulation_dashboard.py
import streamlit as st

@st.cache_data
def load_data():
    return pd.read_csv("data/simulation_temporal_features.csv")

st.title("🚦 RoadSense AI Control Hub")
selected_zone = st.selectbox("Choose Municipal Zone", ["Zone_01", "Zone_02", ...])
st.metric("Predicted Risk", f"{risk_prob:.1%}", delta=f"{risk_delta:.1%}")
```

---

## 8.2. Interactive Mapbox Heatmap (`Plotly Express`)
```python
# dashboard/city_map.py
import plotly.express as px

fig = px.scatter_mapbox(
    zone_df,
    lat="latitude",
    lon="longitude",
    color="predicted_risk_probability",
    size="vehicle_density",
    color_continuous_scale="RdYlGn_r", # Red = High Risk, Green = Low Risk
    mapbox_style="carto-darkmatter",
    zoom=11
)
st.plotly_chart(fig, use_container_width=True)
```

---

## 8.3. LLM Natural Language Safety Briefings (`OpenAI` API)
```python
# intelligence/llm_briefing.py
from openai import OpenAI

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a concise traffic intelligence analyst for a smart city system."},
        {"role": "user", "content": f"Zone {zone} has congestion {cong}%, risk {risk}%. Give an executive safety brief."}
    ],
    max_tokens=250,
    temperature=0.3
)
briefing_text = response.choices[0].message.content
```

---

# 9. 🎓 Student Self-Test & Interview Preparation Q&A

### Q1: Why did we use CIELAB color space instead of RGB for low-light enhancement?
> **Answer**: RGB couples color chromaticity and luminance together in all 3 channels ($R, G, B$). Modifying pixel intensity in RGB alters colors and causes unnatural tints. In CIELAB, lightness ($L^*$) is completely decoupled from color ($a^*, b^*$). Applying CLAHE exclusively to $L^*$ boosts contrast in dark regions while preserving natural vehicle colors.

### Q2: Why is `train_test_split(shuffle=True)` dangerous for time-series forecasting?
> **Answer**: Shuffling time-series data causes **lookahead leakage**. If the model trains on Week 45 and is tested on Week 20, it learns patterns from the future to predict the past. We must use a strict **Chronological Split** (e.g. Train on Weeks 5–40, Test on Weeks 47–52).

### Q3: How does Homography convert camera pixels to real-world speed (km/h)?
> **Answer**: A camera perspective causes distant objects to appear smaller and move fewer pixels. The 4-point Homography matrix ($H$) transforms 2D image coordinates $(u, v)$ into metric coordinates $(X, Y)$ on the road plane. By measuring Euclidean displacement in meters ($\Delta d$) over elapsed time ($\Delta t$), we compute physical speed: $v = \frac{\Delta d}{\Delta t} \times 3.6\text{ km/h}$.

### Q4: Why is ROC-AUC or PR-AUC preferred over Accuracy for accident prediction?
> **Answer**: Traffic accidents are relatively rare events (class imbalance). A dummy model that predicts "No Incident" 100% of the time could have 85% accuracy but zero practical utility. ROC-AUC and PR-AUC measure the model's ability to rank high-risk events above low-risk events across all probability thresholds.

### Q5: What is the purpose of `ColumnTransformer` and `Pipeline` in Scikit-Learn?
> **Answer**: `ColumnTransformer` applies different transformations to different columns (e.g., `StandardScaler` on numerical features and `OneHotEncoder` on categorical features). Wrapping this inside a `Pipeline` ensures that scaler parameters (mean $\mu$, standard deviation $\sigma$) are learned strictly from the training set and applied consistently to validation/test sets without data leakage.

---

### 🚀 Hands-On Challenge Ideas for You:
1. **Experiment with Model Hyperparameters**: In [models/train_model.py](file:///c:/Users/asus/Downloads/traffic_sim-main%20%283%29/traffic_sim-main/models/train_model.py#L238-L245), try changing `n_estimators=200` or testing `GradientBoostingClassifier` and compare the ROC-AUC score.
2. **Inspect the Generated Features**: Run `python analysis/temporal_features.py` and inspect `data/simulation_temporal_features.csv` in Excel or VS Code to see how the lag columns shift values by row.
3. **Launch the Dashboard**: Run `streamlit run app.py` and interact with the 50-zone mapbox, what-if sliders, and AI briefings!
