# Model Card — RoadSense AI Risk Prediction Model

## Model Details

| Field | Value |
|:------|:------|
| **Model Name** | RoadSense AI Traffic Incident Risk Model |
| **Version** | 1.0 |
| **Type** | Random Forest Classifier (scikit-learn Pipeline) |
| **Artifact** | `models/best_risk_model.pkl` |
| **Training Script** | `models/train_model.py` |
| **Framework** | scikit-learn 1.5.x |
| **Random Seed** | 42 |

## Intended Use

- **Primary**: Estimate relative risk of traffic incidents across urban zones to inform municipal resource allocation and intervention prioritization.
- **NOT intended for**: Legal traffic enforcement, insurance underwriting, real-time autonomous driving decisions, or safety-critical systems.

## Training Data

| Property | Value |
|:---------|:------|
| **Source** | Synthetic simulation (`sim.py`) |
| **Type** | Simulated urban traffic panel data |
| **Size** | 2,400 zone-week observations (50 zones × 48 usable weeks) |
| **Time Range** | Weeks 5–52 (Weeks 1–4 dropped for warm-up) |
| **Geographic Scope** | 50 synthetic urban zones across Mumbai, Delhi, Lucknow, Chennai |

> **⚠️ IMPORTANT**: Training data is entirely synthetic. The incident generation process uses a known sigmoid formula with the same features given to the model. This means high evaluation metrics reflect the model's ability to recover the simulator's formula — not validated real-world predictive power. Real deployment would require actual incident records.

## Target Variable

| Property | Value |
|:---------|:------|
| **Name** | `incident_occurred` |
| **Type** | Binary (0/1) |
| **Construction** | `Bernoulli(sigmoid(risk_logit))` where `risk_logit` is a function of congestion, violations, speed, pressure, weather, road condition, and historical incident memory |
| **Base Rate** | ~25–30% positive class |

## Features (40 total)

### Categorical (3)
- `zone_type` — Urban archetype (Residential, Commercial, etc.)
- `weather` — Normal, Light Rain, Heavy Rain
- `road_condition` — Good, Moderate, Poor

### Numerical (37)
- **Current conditions (10)**: population_density, road_capacity, effective_road_capacity, vehicle_density, traffic_pressure, congestion, average_speed, red_light_violations, special_event, vehicle_population_ratio
- **Lag-1 features (7)**: Previous week values for density, congestion, speed, violations, incidents, pressure
- **Rolling 4-week features (7)**: Rolling means/counts/rates for density, congestion, speed, violations, incidents, pressure
- **Week-over-week changes (6)**: Absolute deltas for density, congestion, speed, violations, pressure, incidents
- **Percentage changes (3)**: Relative changes for density, congestion, speed
- **4-week OLS trends (4)**: Linear regression slopes for congestion, density, speed, incidents

### Preprocessing
- **Numerical**: StandardScaler
- **Categorical**: OneHotEncoder (handle_unknown="ignore")

## Validation

| Property | Value |
|:---------|:------|
| **Split Method** | Chronological (no random shuffling) |
| **Training Set** | Weeks 5–40 (~1,800 observations) |
| **Validation Set** | Weeks 41–46 (~300 observations) |
| **Test Set** | Weeks 47–52 (~300 observations) |

### Temporal Leakage
- **Status**: CLEAN — All lag/rolling features use `shift(1)` and `min_periods=4`. No future information enters current-week predictions.

### Spatial Leakage
- **Status**: PRESENT — The same 50 zones appear in train, validation, and test sets. The model has seen every zone's infrastructure characteristics during training. This evaluates temporal generalization (predicting future weeks for known zones) but NOT spatial generalization (predicting unseen zones). A GroupKFold by zone_id would be needed to evaluate spatial generalization.

### Class Imbalance
- **Handling**: `class_weight="balanced"` in both Logistic Regression and Random Forest
- **Threshold**: Default 0.5 (not optimized for recall)

## Evaluation Metrics (Test Set)

| Metric | Logistic Regression | Random Forest (Selected) |
|:-------|:-------------------|:-------------------------|
| Accuracy | Reported in training output | Reported in training output |
| Precision | Reported | Reported |
| Recall | Reported | Reported |
| F1-Score | Reported | Reported |
| ROC-AUC | Reported | Reported |
| PR-AUC | Reported | Reported |
| Brier Score | Reported | Reported |

> **Note**: Specific metric values are generated at training time by `train_model.py`. Re-run the training script to produce current values.

## Known Limitations

1. **Synthetic target reconstruction**: The model may achieve high metrics by recovering the simulator's incident formula rather than learning genuine risk patterns.
2. **No real-world validation**: The model has never been tested against actual incident data.
3. **Spatial generalization unknown**: Performance on unseen zones has not been evaluated.
4. **Probability calibration**: `predict_proba()` outputs should be treated as relative risk scores, not calibrated probabilities.
5. **Feature availability**: Several features (red_light_violations, incident history) require external data sources not available from CCTV alone.

## Ethical Considerations

- Risk scores should not be used to penalize residents or businesses in flagged zones.
- Resource allocation decisions should incorporate human oversight and local knowledge.
- Synthetic training data may not capture real-world demographic, socioeconomic, or infrastructure disparities.
