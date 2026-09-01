"""
================================================================================
ROADSENSE AI — MODEL TRAINING & ACCURACY EVALUATION SUITE
MODULE: backend/train_and_evaluate.py
================================================================================

This module trains a supervised machine learning risk forecasting model on the
dataset organized by weeks and days of the week, and evaluates:
1. Overall Model Accuracy (%), Precision, Recall, F1, ROC-AUC, Brier Score
2. Full Confusion Matrix (TP, TN, FP, FN)
3. Day-of-the-Week Accuracy Breakdown (Accuracy specifically on Mondays vs. Weekends)
4. Temporal Week-by-Week Accuracy Trajectory
5. Feature Importance Rankings (impact of day of week, weather, road conditions)
================================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

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

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ORGANIZED_DIR = os.path.join(BACKEND_DIR, "organized_data")
MASTER_CSV_PATH = os.path.join(ORGANIZED_DIR, "master_organized_panel.csv")
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "custom_trained_risk_model.pkl")
PREDICTIONS_OUTPUT_PATH = os.path.join(DATA_DIR, "custom_model_predictions.csv")
REPORT_OUTPUT_PATH = os.path.join(BACKEND_DIR, "accuracy_report.json")

RANDOM_SEED = 42
TARGET_COL = "incident_occurred"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def select_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Intelligently identifies categorical and numerical feature columns,
    excluding IDs, raw timestamps, and targets to prevent data leakage.
    """
    excluded_keywords = [
        "incident_occurred", "target", "label", "crash_date", "rash_date",
        "date", "timestamp", "datetime", "dt_parsed", "_dt_parsed", "id",
        "beat_of_occurrence", "zone_id", "crashes_delta", "_target_count_raw",
        "injuries_total", "injuries", "injury", "fatalities", "deaths", "crash_count"
    ]

    all_cols = list(df.columns)
    candidate_cols = []

    for c in all_cols:
        c_lower = c.lower()
        if c == TARGET_COL:
            continue
        if any(k == c_lower or k in c_lower for k in excluded_keywords):
            continue
        candidate_cols.append(c)

    num_cols = []
    cat_cols = []

    for c in candidate_cols:
        # Keep week and is_weekend as numeric
        if c in ["week", "month", "hour", "is_weekend"]:
            num_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # Always ensure day_name or day_of_week is included
    if "day_name" in df.columns and "day_name" not in cat_cols:
        cat_cols.append("day_name")

    return cat_cols, num_cols


def chronological_split(
    df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    split_ratio: float = 0.70
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset chronologically by calendar week to prevent temporal lookahead leakage.
    Earlier weeks -> Training set, Later weeks -> Test set.
    """
    features = cat_cols + num_cols
    weeks = sorted(df["week"].unique())
    
    if len(weeks) >= 3:
        split_idx = max(1, int(len(weeks) * split_ratio))
        train_weeks = weeks[:split_idx]
        test_weeks = weeks[split_idx:]
        
        train_mask = df["week"].isin(train_weeks)
        test_mask = df["week"].isin(test_weeks)
    else:
        # Fallback to random chronological index split
        split_idx = int(len(df) * split_ratio)
        train_mask = df.index < split_idx
        test_mask = df.index >= split_idx
        train_weeks = [min(weeks)]
        test_weeks = [max(weeks)]

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    X_train = train_df[features]
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df[features]
    y_test = test_df[TARGET_COL].astype(int)

    return X_train, X_test, y_train, y_test, train_df, test_df


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """Constructs a ColumnTransformer with One-Hot encoding and Standard scaling."""
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    return ColumnTransformer(transformers=transformers)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> Dict[str, Any]:
    """Computes comprehensive accuracy and classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    
    # Handle single-class edge cases gracefully
    has_positives = (np.sum(y_true == 1) > 0)
    has_negatives = (np.sum(y_true == 0) > 0)
    
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    roc_auc = float(roc_auc_score(y_true, y_probs)) if (has_positives and has_negatives) else 0.5
    pr_auc = float(average_precision_score(y_true, y_probs)) if has_positives else 0.0
    brier = float(brier_score_loss(y_true, y_probs))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    
    return {
        "accuracy_pct": round(acc * 100.0, 2),
        "precision_pct": round(prec * 100.0, 2),
        "recall_pct": round(rec * 100.0, 2),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "total_tested": int(len(y_true))
        }
    }


def compute_day_of_week_accuracy(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Computes accuracy, precision, and recall broken down by day of the week.
    """
    eval_df = test_df.copy()
    eval_df["y_pred"] = y_pred
    eval_df["y_prob"] = y_probs
    eval_df["is_correct"] = (eval_df[TARGET_COL].astype(int) == eval_df["y_pred"]).astype(int)

    day_col = "day_name" if "day_name" in eval_df.columns else "day_of_week"
    day_metrics = {}

    for d in DAY_ORDER:
        d_df = eval_df[eval_df[day_col] == d]
        if len(d_df) == 0:
            continue

        y_t = d_df[TARGET_COL].astype(int).values
        y_p = d_df["y_pred"].values

        acc = float(accuracy_score(y_t, y_p)) * 100.0
        prec = float(precision_score(y_t, y_p, zero_division=0)) * 100.0
        rec = float(recall_score(y_t, y_p, zero_division=0)) * 100.0
        f1 = float(f1_score(y_t, y_p, zero_division=0))

        day_metrics[d] = {
            "test_records": int(len(d_df)),
            "actual_incidents": int(y_t.sum()),
            "predicted_incidents": int(y_p.sum()),
            "accuracy_pct": round(acc, 2),
            "precision_pct": round(prec, 2),
            "recall_pct": round(rec, 2),
            "f1_score": round(f1, 4),
            "avg_predicted_risk": round(float(d_df["y_prob"].mean()), 3)
        }

    return day_metrics


def compute_weekly_accuracy_trajectory(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Computes accuracy week-by-week across the test split.
    """
    eval_df = test_df.copy()
    eval_df["y_pred"] = y_pred
    eval_df["y_prob"] = y_probs
    eval_df["is_correct"] = (eval_df[TARGET_COL].astype(int) == eval_df["y_pred"]).astype(int)

    weekly_stats = []
    for w, w_df in eval_df.groupby("week"):
        y_t = w_df[TARGET_COL].astype(int).values
        y_p = w_df["y_pred"].values
        weekly_stats.append({
            "week": int(w),
            "test_records": int(len(w_df)),
            "actual_incidents": int(y_t.sum()),
            "predicted_incidents": int(y_p.sum()),
            "accuracy_pct": round(float(accuracy_score(y_t, y_p) * 100.0), 2),
            "avg_predicted_risk": round(float(w_df["y_prob"].mean()), 3)
        })

    return sorted(weekly_stats, key=lambda x: x["week"])


def extract_feature_importances(
    pipeline: Pipeline,
    cat_cols: List[str],
    num_cols: List[str]
) -> List[Dict[str, Any]]:
    """Extracts ranked feature importances from the Random Forest model."""
    try:
        clf = pipeline.named_steps["classifier"]
        prep = pipeline.named_steps["preprocessor"]

        cat_feature_names = []
        if cat_cols and "cat" in prep.named_transformers_:
            ohe = prep.named_transformers_["cat"]
            cat_feature_names = list(ohe.get_feature_names_out(cat_cols))

        feature_names = num_cols + cat_feature_names
        importances = clf.feature_importances_

        imp_list = []
        for name, imp in zip(feature_names, importances):
            imp_list.append({
                "feature": name,
                "importance": round(float(imp), 4)
            })

        imp_list = sorted(imp_list, key=lambda x: x["importance"], reverse=True)
        return imp_list
    except Exception as e:
        print(f"[-] Feature importance extraction note: {e}")
        return []


def train_and_evaluate_model(
    organized_csv_path: Optional[str] = None,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Loads organized dataset, trains models, evaluates accuracy,
    and saves the trained model and structured report.
    """
    csv_path = organized_csv_path or MASTER_CSV_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Organized dataset not found at: {csv_path}. Run backend/temporal_organizer.py first.")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"\n[+] Ingesting Organized Dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    - Total Rows: {len(df):,} | Columns: {len(df.columns)}")

    # 1. Feature Specification
    cat_cols, num_cols = select_feature_columns(df)
    print(f"    - Identified {len(num_cols)} Numerical Features: {num_cols[:6]}...")
    print(f"    - Identified {len(cat_cols)} Categorical Features: {cat_cols}")

    # 2. Chronological Split (Train: 70%, Test: 30%)
    X_train, X_test, y_train, y_test, train_df, test_df = chronological_split(df, cat_cols, num_cols, split_ratio=0.70)
    print(f"\n[+] Chronological Time Split:")
    print(f"    - Training Set: {len(X_train):,} rows (Incidents: {y_train.sum():,} | {y_train.mean()*100:.1f}%)")
    print(f"    - Test Set:     {len(X_test):,} rows (Incidents: {y_test.sum():,} | {y_test.mean()*100:.1f}%)")

    # 3. Build Preprocessing Pipeline
    preprocessor = build_preprocessor(cat_cols, num_cols)

    # 4. Train Model 1: Logistic Regression Baseline
    print("\n[+] Training Model 1: Logistic Regression Baseline (class_weight='balanced')...")
    pipe_lr = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(C=0.5, class_weight="balanced", random_state=RANDOM_SEED, max_iter=1000))
    ])
    pipe_lr.fit(X_train, y_train)
    lr_probs = pipe_lr.predict_proba(X_test)[:, 1] if hasattr(pipe_lr, "predict_proba") else pipe_lr.predict(X_test)
    lr_preds = (lr_probs >= threshold).astype(int)
    lr_metrics = evaluate_predictions(y_test.values, lr_preds, lr_probs)

    # 5. Train Model 2: Random Forest Classifier
    print("[+] Training Model 2: Random Forest Classifier (Non-linear ensemble)...")
    pipe_rf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=7,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])
    pipe_rf.fit(X_train, y_train)
    rf_probs = pipe_rf.predict_proba(X_test)[:, 1] if hasattr(pipe_rf, "predict_proba") else pipe_rf.predict(X_test)
    rf_preds = (rf_probs >= threshold).astype(int)
    rf_metrics = evaluate_predictions(y_test.values, rf_preds, rf_probs)

    # 6. Select Best Model (based on F1 & ROC-AUC)
    best_name = "Random Forest Classifier" if rf_metrics["f1_score"] >= lr_metrics["f1_score"] else "Logistic Regression"
    best_pipe = pipe_rf if best_name == "Random Forest Classifier" else pipe_lr
    best_metrics = rf_metrics if best_name == "Random Forest Classifier" else lr_metrics
    best_probs = rf_probs if best_name == "Random Forest Classifier" else lr_probs
    best_preds = rf_preds if best_name == "Random Forest Classifier" else lr_preds

    # 7. Day-of-the-Week Accuracy Breakdown
    day_metrics = compute_day_of_week_accuracy(test_df, best_preds, best_probs)

    # 8. Weekly Accuracy Trajectory
    weekly_trajectory = compute_weekly_accuracy_trajectory(test_df, best_preds, best_probs)

    # 9. Top Feature Importances
    feature_importances = extract_feature_importances(pipe_rf, cat_cols, num_cols)

    # 10. Save Model Artifacts
    joblib.dump(best_pipe, MODEL_OUTPUT_PATH)
    print(f"\n[+] Saved Trained Best Model to: {MODEL_OUTPUT_PATH}")

    # Export test predictions
    test_export = test_df.copy()
    test_export["predicted_risk_prob"] = best_probs
    test_export["predicted_incident"] = best_preds
    test_export["prediction_correct"] = (test_export[TARGET_COL].astype(int) == best_preds).astype(int)
    test_export.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    print(f"[+] Saved Test Predictions ({len(test_export):,} rows) to: {PREDICTIONS_OUTPUT_PATH}")

    # Compile Final Report
    report = {
        "selected_model": best_name,
        "decision_threshold": threshold,
        "dataset_stats": {
            "total_observations": len(df),
            "train_observations": len(X_train),
            "test_observations": len(X_test),
            "base_incident_rate_pct": round(float(df[TARGET_COL].mean() * 100.0), 2)
        },
        "model_comparison": {
            "logistic_regression": lr_metrics,
            "random_forest": rf_metrics
        },
        "best_model_metrics": best_metrics,
        "day_of_week_accuracy": day_metrics,
        "weekly_trajectory": weekly_trajectory,
        "top_feature_importances": feature_importances[:15],
        "model_file": MODEL_OUTPUT_PATH,
        "predictions_file": PREDICTIONS_OUTPUT_PATH
    }

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Exported Accuracy Report to: {REPORT_OUTPUT_PATH}")

    return report


def print_accuracy_report(report: Dict[str, Any]):
    """Renders a beautiful ASCII report in the console."""
    best = report["best_model_metrics"]
    cm = best["confusion_matrix"]
    
    print("\n" + "=" * 80)
    print(" MODEL TRAINING & ACCURACY BENCHMARK REPORT ".center(80, "="))
    print("=" * 80)
    print(f"Selected Champion Model: {report['selected_model']}")
    print(f"Decision Threshold:      P >= {report['decision_threshold']:.2f} -> Incident Warning")
    print(f"Test Observations:       {cm['total_tested']:,} records")

    print("\n[+] 1. OVERALL TEST ACCURACY & PERFORMANCE:")
    print(f"    - ACCURACY:   {best['accuracy_pct']:>6.2f}% (Total Correct Predictions / Tested)")
    print(f"    - PRECISION:  {best['precision_pct']:>6.2f}% (True Alarms / All Alarms)")
    print(f"    - RECALL:     {best['recall_pct']:>6.2f}% (Hazards Detected / Total Actual Hazards)")
    print(f"    - F1-SCORE:   {best['f1_score']:>6.4f}")
    print(f"    - ROC-AUC:    {best['roc_auc']:>6.4f}")
    print(f"    - BRIER LOSS: {best['brier_score']:>6.4f} (Probability Calibration Error)")

    print("\n[+] 2. CONFUSION MATRIX (TEST SET):")
    print("    " + "-" * 50)
    print(f"    {'Actual Safe (0)':<22} | TN: {cm['true_negatives']:>5} | FP: {cm['false_positives']:>5} (False Alarm)")
    print(f"    {'Actual Incident (1)':<22} | FN: {cm['false_negatives']:>5} | TP: {cm['true_positives']:>5} (Captured)")
    print("    " + "-" * 50)

    print("\n[+] 3. ACCURACY BROKEN DOWN BY DAY OF THE WEEK:")
    print("    " + "-" * 72)
    print(f"    {'Day of Week':<12} | {'Tested':<6} | {'Accuracy':<10} | {'Recall':<10} | {'Avg Risk':<10}")
    print("    " + "-" * 72)
    for day, metrics in report["day_of_week_accuracy"].items():
        print(f"    {day:<12} | {metrics['test_records']:>6} | {metrics['accuracy_pct']:>8.2f}% | {metrics['recall_pct']:>8.2f}% | {metrics['avg_predicted_risk']:>8.3f}")
    print("    " + "-" * 72)

    print("\n[+] 4. TOP 10 FEATURE CONTRIBUTIONS:")
    print("    " + "-" * 50)
    for idx, item in enumerate(report["top_feature_importances"][:10], 1):
        print(f"    {idx:>2}. {item['feature']:<32}: {item['importance']:.4f}")
    print("    " + "-" * 50)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    report = train_and_evaluate_model()
    print_accuracy_report(report)
