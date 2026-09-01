# Backend CSV Input Directory

Place your raw CSV datasets in this folder (`backend/input_csv/`).

### Supported Formats
Any CSV dataset containing:
- **Timestamp or Date column**: e.g., `CRASH_DATE`, `Date`, `timestamp`, `datetime`, `date_time`, `CRASH_DAY_OF_WEEK`, `CRASH_MONTH`, etc.
- **Traffic / Accident / Safety Factors**: e.g., speed, weather, lighting, road condition, vehicle counts, location/zone/beat.
- **Target Outcome** (optional/auto-detected): e.g., `incident_occurred`, `INJURIES_TOTAL`, `crash`, `accident`, `target`.

### How It Works
1. When you place a CSV file here, the **Temporal Organizer** (`backend/temporal_organizer.py`) parses dates and partitions records into:
   - **Weeks**: Calendar Weeks 1 to 52 (`backend/organized_data/by_week/week_{W}.csv`)
   - **Day of the Week**: Monday through Sunday (`backend/organized_data/by_day_of_week/{day}.csv`)
2. The **Model Trainer** (`backend/train_and_evaluate.py`) uses the organized dataset to train supervised ML models (Logistic Regression & Random Forest) and measures:
   - Overall Accuracy (%)
   - Precision, Recall, F1-Score, ROC-AUC
   - Full Confusion Matrix
   - Day-of-the-Week Accuracy breakdown (how accurate the model is on Mondays vs. Fridays vs. Weekends)
   - Top Feature Importances

### One-Command CLI Run
```powershell
python backend/organize_and_train.py
```
