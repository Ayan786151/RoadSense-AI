"""
================================================================================
ROADSENSE AI — END-TO-END DATASET ORGANIZER & MODEL ACCURACY BENCHMARK
MODULE: backend/organize_and_train.py
================================================================================

One-Command CLI Runner:
1. Scans backend/input_csv/ for user-dropped CSV datasets.
2. Organizes records chronologically into:
   - Calendar Weeks (Weeks 1 to 52)
   - Day of the Week (Monday to Sunday)
   - Partitions in backend/organized_data/
3. Trains Supervised ML Risk Models (Logistic Regression & Random Forest).
4. Evaluates and benchmarks accuracy metrics:
   - Overall Accuracy (%), Precision, Recall, F1, ROC-AUC, Brier score
   - Confusion Matrix
   - Day-of-the-Week Accuracy comparison
   - Week-by-week performance trajectory
   - Top Feature Importances
5. Exports trained models and predictions for downstream UI consumption.
================================================================================
"""

import os
import sys
import argparse

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from temporal_organizer import find_input_csv_files, organize_csv_dataset
from train_and_evaluate import train_and_evaluate_model, print_accuracy_report


def run_full_pipeline(csv_path: str = None, threshold: float = 0.50):
    print("\n" + "=" * 80)
    print(" ROADSENSE AI — AUTOMATED TEMPORAL ORGANIZER & ML ACCURACY PIPELINE ".center(80, "="))
    print("=" * 80)

    # 1. Discover CSV
    if not csv_path:
        files = find_input_csv_files()
        if not files:
            print("\n[-] Error: No CSV files found in 'backend/input_csv/'.")
            print("    Please drop your CSV file into: backend/input_csv/")
            print("    Supported columns: CRASH_DATE / Date, Speed, Weather, Road, etc.\n")
            return
        csv_path = files[0]
        print(f"[+] Automatically detected CSV: {os.path.basename(csv_path)}")
    else:
        if not os.path.exists(csv_path):
            print(f"[-] Error: File not found at: {csv_path}")
            return
        print(f"[+] User-specified CSV: {csv_path}")

    # 2. Organize Dataset by Weeks and Day of the Week
    print("\n" + "-" * 80)
    print(" STEP 1: ORGANIZING DATASET BY WEEKS & DAY OF THE WEEK ".center(80, "-"))
    print("-" * 80)
    df_organized, org_summary = organize_csv_dataset(csv_path)

    print(f"\n[OK] Organization Successful:")
    print(f"    - Spans {org_summary['weeks_count']} Weeks (Week {org_summary['min_week']} to {org_summary['max_week']})")
    print(f"    - Structured into 7 Day-of-Week partitions in: backend/organized_data/by_day_of_week/")
    print(f"    - Structured into {org_summary['weeks_count']} Weekly partitions in: backend/organized_data/by_week/")
    print(f"    - Master File: {org_summary['master_csv_path']}")

    # 3. Train Model and Benchmark Accuracy
    print("\n" + "-" * 80)
    print(" STEP 2: TRAINING MODEL & EVALUATING ACCURACY ".center(80, "-"))
    print("-" * 80)
    accuracy_report = train_and_evaluate_model(
        organized_csv_path=org_summary['master_csv_path'],
        threshold=threshold
    )

    # 4. Print Structured ASCII Results
    print_accuracy_report(accuracy_report)
    print("[OK] Pipeline execution finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize dataset into weeks and days of week, then train and evaluate ML model.")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input CSV file (defaults to first CSV found in backend/input_csv/)")
    parser.add_argument("--threshold", "-t", type=float, default=0.50, help="Classification decision threshold (default: 0.50)")

    args = parser.parse_args()
    run_full_pipeline(csv_path=args.input, threshold=args.threshold)
