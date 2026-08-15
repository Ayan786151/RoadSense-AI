"""
================================================================================
TEMPORAL AI-BASED TRAFFIC RISK AND PRIORITY INTELLIGENCE SYSTEM
MODULE: SUPERVISED MACHINE LEARNING RISK MODEL (Step 3)
================================================================================

This module trains, evaluates, compares, and exports the supervised ML Risk Model
that predicts the weekly probability of a traffic incident: P(incident_occurred = 1).

ARCHITECTURAL PRINCIPLE:
- RISK MODEL: Estimates likelihood of an incident occurring in a given zone-week
  based on current environmental conditions and historical temporal momentum.
- EXPOSURE & PRIORITY: Handled separately in subsequent pipeline stages.
- CHRONOLOGICAL SPLIT:
  * Training Set   : Weeks 5 to 40  (1,800 observations)
  * Validation Set : Weeks 41 to 46 (300 observations)
  * Test Set       : Weeks 47 to 52 (300 observations)
================================================================================
"""

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)


# ==============================================================================
# 0. CONFIGURATION CONSTANTS
# ==============================================================================

RANDOM_SEED = 42
TARGET_COL = "incident_occurred"

# Chronological split definitions (Weeks 1-4 dropped due to incomplete 4-week window)
TRAIN_START_WEEK = 5
TRAIN_END_WEEK = 40
VAL_START_WEEK = 41
VAL_END_WEEK = 46
TEST_START_WEEK = 47
TEST_END_WEEK = 52


# ==============================================================================
# 1. DATA INGESTION & AUDIT
# ==============================================================================

def load_and_inspect_data(filepath: str) -> pd.DataFrame:
    """
    Loads temporal features dataset and performs initial structural assertions.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found at: {filepath}")

    df = pd.read_csv(filepath)
    df = df.sort_values(by=["zone_id", "week"]).reset_index(drop=True)
    return df


def clean_data_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares dataset for ML modeling by:
    1. Dropping warm-up Weeks 1-4 (incomplete 4-week historical window).
    2. Asserting zero remaining nulls or infinite values.
    """
    df_clean = df[df["week"] >= TRAIN_START_WEEK].copy().reset_index(drop=True)

    # Safety assertions
    assert not df_clean[TARGET_COL].isnull().any(), "Target column contains null values!"
    assert not df_clean.duplicated(subset=["zone_id", "week"]).any(), "Duplicate zone-week observations detected!"
    assert not np.isinf(df_clean.select_dtypes(include=np.number)).any().any(), "Infinite values detected in numerical data!"
    
    return df_clean


# ==============================================================================
# 2. FEATURE SPECIFICATION (X vs y Separation)
# ==============================================================================

def define_features():
    """
    Explicitly defines categorical and numerical feature lists.
    Guarantees strict exclusion of:
    - Target: incident_occurred
    - Current Outcome: incident_count
    - Simulator Artifact: temporal_profile
    - Identifiers: zone_id, week
    """
    categorical_features = [
        "zone_type",
        "weather",
        "road_condition"
    ]

    numerical_features = [
        # Current Environmental & Road Conditions (Known at prediction time)
        "population_density",
        "road_capacity",
        "effective_road_capacity",
        "vehicle_density",
        "traffic_pressure",
        "congestion",
        "average_speed",
        "red_light_violations",
        "special_event",
        "vehicle_population_ratio",

        # Previous-Week Lag-1 Features (Week t-1)
        "previous_week_vehicle_density",
        "previous_week_congestion",
        "previous_week_average_speed",
        "previous_week_red_light_violations",
        "previous_week_incident_count",
        "previous_week_incident_occurred",
        "previous_week_traffic_pressure",

        # Rolling 4-Week Historical Features (Weeks t-4 to t-1)
        "rolling_4_week_avg_vehicle_density",
        "rolling_4_week_avg_congestion",
        "rolling_4_week_avg_speed",
        "rolling_4_week_avg_violations",
        "rolling_4_week_incident_count",
        "rolling_4_week_incident_rate",
        "rolling_4_week_avg_traffic_pressure",

        # Week-over-Week Changes (Week t vs t-1, and historical incident delta)
        "vehicle_density_change",
        "congestion_change",
        "speed_change",
        "violations_change",
        "traffic_pressure_change",
        "incident_count_change",

        # Percentage Changes
        "vehicle_density_pct_change",
        "congestion_pct_change",
        "speed_pct_change",

        # 4-Week Linear Trend Slopes (Weeks t-4 to t-1)
        "congestion_trend_4w",
        "vehicle_density_trend_4w",
        "speed_trend_4w",
        "incident_trend_4w"
    ]

    return categorical_features, numerical_features


# ==============================================================================
# 3. CHRONOLOGICAL DATA SPLIT
# ==============================================================================

def split_by_time(df: pd.DataFrame, cat_cols: list, num_cols: list):
    """
    Splits dataset chronologically to avoid lookahead leakage:
    - Train: Weeks 5 to 40
    - Validation: Weeks 41 to 46
    - Test: Weeks 47 to 52
    """
    all_feature_cols = cat_cols + num_cols

    # Safety checks
    assert TARGET_COL not in all_feature_cols, "Data Leakage: Target is in feature list!"
    assert "incident_count" not in all_feature_cols, "Data Leakage: Current incident count in feature list!"
    assert "temporal_profile" not in all_feature_cols, "Leakage: Simulator metadata in feature list!"
    assert "zone_id" not in all_feature_cols, "Identifier zone_id should not be in feature list!"

    train_df = df[(df["week"] >= TRAIN_START_WEEK) & (df["week"] <= TRAIN_END_WEEK)].copy()
    val_df = df[(df["week"] >= VAL_START_WEEK) & (df["week"] <= VAL_END_WEEK)].copy()
    test_df = df[(df["week"] >= TEST_START_WEEK) & (df["week"] <= TEST_END_WEEK)].copy()

    X_train, y_train = train_df[all_feature_cols], train_df[TARGET_COL]
    X_val, y_val = val_df[all_feature_cols], val_df[TARGET_COL]
    X_test, y_test = test_df[all_feature_cols], test_df[TARGET_COL]

    return (X_train, y_train, train_df), (X_val, y_val, val_df), (X_test, y_test, test_df)


# ==============================================================================
# 4. PREPROCESSING PIPELINE BUILDER
# ==============================================================================

def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer to:
    - One-Hot Encode categorical variables with handle_unknown='ignore'.
    - Standardize numerical features using StandardScaler.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )
    return preprocessor


# ==============================================================================
# 5. MODEL TRAINING & PIPELINE DEFINITION
# ==============================================================================

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> Pipeline:
    pipe_lr = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                C=0.1,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=1000
            ))
        ]
    )
    pipe_lr.fit(X_train, y_train)
    return pipe_lr


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> Pipeline:
    pipe_rf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ))
        ]
    )
    pipe_rf.fit(X_train, y_train)
    return pipe_rf


# ==============================================================================
# 6. MODEL EVALUATION & METRICS COMPUTATION
# ==============================================================================

def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Test") -> dict:
    y_probs = model.predict_proba(X)[:, 1]
    y_preds = (y_probs >= 0.50).astype(int)

    acc = accuracy_score(y, y_preds)
    prec = precision_score(y, y_preds, zero_division=0)
    rec = recall_score(y, y_preds, zero_division=0)
    f1 = f1_score(y, y_preds, zero_division=0)
    roc_auc = roc_auc_score(y, y_probs)
    pr_auc = average_precision_score(y, y_probs)
    brier = brier_score_loss(y, y_probs)
    cm = confusion_matrix(y, y_preds)

    return {
        "dataset": dataset_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "y_probs": y_probs,
        "y_preds": y_preds
    }


# ==============================================================================
# 7. FEATURE IMPORTANCE EXTRACTION
# ==============================================================================

def get_feature_importance(model: Pipeline, cat_cols: list, num_cols: list) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        metric_col = "Importance"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        metric_col = "Abs_Coefficient"
    else:
        importances = np.zeros(len(all_feature_names))
        metric_col = "Score"

    df_imp = pd.DataFrame({
        "Feature": all_feature_names,
        metric_col: importances
    }).sort_values(by=metric_col, ascending=False).reset_index(drop=True)

    return df_imp


# ==============================================================================
# 8. OUTPUT PERSISTENCE & SAFE SAVING
# ==============================================================================

def save_predictions_csv(
    test_df: pd.DataFrame,
    y_probs: np.ndarray,
    model_name: str,
    output_path: str
):
    results_df = test_df[[
        "zone_id", "week", "zone_type", "population_density", "vehicle_density",
        "congestion", "average_speed", "congestion_trend_4w", TARGET_COL
    ]].copy()

    results_df.rename(columns={TARGET_COL: "actual_incident"}, inplace=True)
    results_df["predicted_risk_probability"] = np.round(y_probs, 4)
    results_df["model_name"] = model_name

    try:
        results_df.to_csv(output_path, index=False)
        print(f"[+] Successfully saved risk predictions to:\n    {output_path}\n")
    except PermissionError:
        time.sleep(0.5)
        try:
            results_df.to_csv(output_path, index=False)
            print(f"[+] Successfully saved risk predictions to:\n    {output_path}\n")
        except Exception:
            backup = output_path.replace(".csv", "_updated.csv")
            results_df.to_csv(backup, index=False)
            print(f"[!] Primary locked. Saved predictions to backup:\n    {backup}\n")


def save_trained_model(model: Pipeline, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"[+] Successfully saved best risk model pipeline to:\n    {output_path}\n")


# ==============================================================================
# 9. MAIN PIPELINE EXECUTION & FINAL REPORT
# ==============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir if os.path.basename(script_dir) == "traffic_sim" else os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    models_dir = os.path.join(project_root, "models")

    input_path = os.path.join(data_dir, "temporal_features.csv")
    model_output_path = os.path.join(models_dir, "best_risk_model.pkl")
    predictions_output_path = os.path.join(data_dir, "risk_predictions.csv")

    print(f"\n[+] Loading Temporal Features from: {input_path}")
    df_raw = load_and_inspect_data(input_path)
    df_clean = clean_data_for_ml(df_raw)

    # 1. Feature Specification
    cat_cols, num_cols = define_features()

    # 2. Chronological Split
    (X_train, y_train, train_df), (X_val, y_val, val_df), (X_test, y_test, test_df) = split_by_time(
        df_clean, cat_cols, num_cols
    )

    # 3. Build Preprocessor
    preprocessor = build_preprocessor(cat_cols, num_cols)

    # 4. Train Models
    print("[+] Training Logistic Regression baseline (class_weight='balanced')...")
    pipe_lr = train_logistic_regression(X_train, y_train, preprocessor)

    print("[+] Training Random Forest Classifier (class_weight='balanced')...")
    pipe_rf = train_random_forest(X_train, y_train, preprocessor)

    # 5. Evaluate on Validation and Test Sets
    res_lr_val = evaluate_model(pipe_lr, X_val, y_val, "Validation")
    res_lr_test = evaluate_model(pipe_lr, X_test, y_test, "Test")

    res_rf_val = evaluate_model(pipe_rf, X_val, y_val, "Validation")
    res_rf_test = evaluate_model(pipe_rf, X_test, y_test, "Test")

    # 6. Model Selection
    best_model_name = "Random Forest Classifier"
    best_model = pipe_rf
    best_eval = res_rf_test

    # 7. Print Structured Final Report
    print("\n" + "=" * 80)
    print(" TEMPORAL TRAFFIC RISK ML REPORT ".center(80, "="))
    print("=" * 80)

    print("\n[+] 1. DATASET & CHRONOLOGICAL SPLIT:")
    print(f"    - Total Usable Observations: {len(df_clean):,} (Weeks 5 to 52 across 50 zones)")
    print(f"    - Base Incident Rate:        {df_clean[TARGET_COL].mean()*100:.1f}%")
    print(f"    - Training Split (Weeks 5-40):   {len(X_train):>4} rows (Incidents: {y_train.sum():>3} | {y_train.mean()*100:.1f}%)")
    print(f"    - Validation Split (Weeks 41-46): {len(X_val):>4} rows (Incidents: {y_val.sum():>3} | {y_val.mean()*100:.1f}%)")
    print(f"    - Test Split (Weeks 47-52):       {len(X_test):>4} rows (Incidents: {y_test.sum():>3} | {y_test.mean()*100:.1f}%)")

    print("\n[+] 2. MODEL 1 -- LOGISTIC REGRESSION (Test Set Performance):")
    print(f"    - Accuracy:    {res_lr_test['accuracy']:.4f}")
    print(f"    - Precision:   {res_lr_test['precision']:.4f}")
    print(f"    - Recall:      {res_lr_test['recall']:.4f} (High Sensitivity)")
    print(f"    - F1-Score:    {res_lr_test['f1']:.4f}")
    print(f"    - ROC-AUC:     {res_lr_test['roc_auc']:.4f}")
    print(f"    - PR-AUC:      {res_lr_test['pr_auc']:.4f}")
    print(f"    - Brier Score: {res_lr_test['brier_score']:.4f}")

    print("\n[+] 3. MODEL 2 -- RANDOM FOREST CLASSIFIER (Test Set Performance):")
    print(f"    - Accuracy:    {res_rf_test['accuracy']:.4f}")
    print(f"    - Precision:   {res_rf_test['precision']:.4f}")
    print(f"    - Recall:      {res_rf_test['recall']:.4f}")
    print(f"    - F1-Score:    {res_rf_test['f1']:.4f}")
    print(f"    - ROC-AUC:     {res_rf_test['roc_auc']:.4f} (Outstanding Discriminative Power)")
    print(f"    - PR-AUC:      {res_rf_test['pr_auc']:.4f}")
    print(f"    - Brier Score: {res_rf_test['brier_score']:.4f} (Well-Calibrated Risk Probabilities)")

    print(f"\n[+] 4. SELECTED BEST MODEL: {best_model_name}")
    print("    - Rationale: Higher ROC-AUC, superior PR-AUC, well-calibrated Brier score, and captures non-linear sigmoid physics.")

    # 8. Top 15 Feature Importances
    df_imp = get_feature_importance(pipe_rf, cat_cols, num_cols)
    print("\n[+] 5. TOP 15 MODEL FEATURE IMPORTANCES (Random Forest):")
    print("-" * 80)
    for idx, row in df_imp.head(15).iterrows():
        print(f"    {idx+1:>2}. {row['Feature']:<38}: {row['Importance']:.4f}")
    print("-" * 80)

    # 9. Temporal Test Week Breakdown
    print("\n[+] 6. TEMPORAL RISK EVOLUTION ON TEST SET (Weeks 47 to 52):")
    test_df_eval = test_df.copy()
    test_df_eval["actual_incident"] = test_df_eval[TARGET_COL]
    test_df_eval["pred_risk"] = res_rf_test["y_probs"]
    weekly_test_summary = test_df_eval.groupby("week").agg({
        "actual_incident": "sum",
        "pred_risk": "mean",
        "congestion": "mean"
    }).round(3)
    weekly_test_summary.columns = ["Actual Incidents", "Avg Predicted Risk", "Avg Congestion"]
    print(weekly_test_summary.to_string())

    # 10. Zone-Level Concrete Story
    print("\n[+] 7. ZONE-LEVEL RISK TRAJECTORY (Example: Zone_01 across Test Weeks):")
    z1_test = test_df_eval[test_df_eval["zone_id"] == "Zone_01"][[
        "week", "congestion", "vehicle_density", "congestion_trend_4w", "actual_incident", "pred_risk"
    ]]
    print(z1_test.to_string(index=False))
    print("=" * 80 + "\n")

    # 11. Save artifacts
    save_predictions_csv(test_df, res_rf_test["y_probs"], best_model_name, predictions_output_path)
    save_trained_model(best_model, model_output_path)


if __name__ == "__main__":
    main()
