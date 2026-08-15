# RoadSense AI - Temporal Architecture Documentation
## From Video Observation Sessions to Machine Learning Temporal Intelligence

---

## 1. Core Semantic Hierarchy

A foundational principle of RoadSense AI is the strict semantic distinction between video processing runs, calendar time, and machine learning historical depth:

```
+-------------------------------------------------------------------------+
| LEVEL 1: OBSERVATION SESSION (e.g. session_001, session_002, session_003)|
| - Represents an individual camera video recording feed.                 |
| - High-frequency time steps: timestamp_seconds (e.g. 0.167s to 150.45s).|
| - Local frame measurements (vehicle counts, pixel speeds, congestion).   |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
| LEVEL 2: OBSERVATION DATE (e.g. 2026-08-15)                             |
| - The physical calendar day on which recordings occurred.               |
| - Multiple observation sessions can occur on the same observation date. |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
| LEVEL 3: CALENDAR WEEK (e.g. week_id = 1 / ISO Calendar Week 33)        |
| - A standardized 7-day municipal observation window.                    |
| - Sessions recorded on the same date or within the same 7 days belong    |
|   to the SAME calendar week.                                            |
| - Multiple video recordings on 2026-08-15 do NOT equal multiple weeks.  |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
| LEVEL 4: MACHINE LEARNING HISTORICAL DEPTH (t-1, t-4..t-1)              |
| - Lag-1 Features: Require >= 2 distinct historical weeks.               |
| - 4-Week Rolling & OLS Trends: Require >= 5 distinct historical weeks.  |
| - If distinct weeks < 2: Status = WARMUP / INSUFFICIENT_HISTORICAL_WEEKS|
+-------------------------------------------------------------------------+
```

---

## 2. Ingestion & Deduplication Mechanism

### Composite Identity Key
Idempotency is guaranteed by tracking uniqueness across:
$$\text{Key} = (\text{session\_id}, \text{location\_id}, \text{camera\_id}, \text{observation\_date}, \text{timestamp\_seconds})$$

### Ingestion Flow
1. **Load**: Ingests `data/sessions/{session_id}/live_traffic_observations.csv` and `session_metadata.json`.
2. **Validate**: Asserts schema completeness, non-null values, timestamp consistency, and lack of internal corruption.
3. **Standardize Provenance**: Adds `session_id`, `location_id`, `camera_id`, `observation_date`, `week_id`, `source_type` (`"CAMERA_LIVE"`), and `observation_sequence`.
4. **Idempotent Merge**: Matches incoming keys against `data/temporal_traffic_store.csv`. Only novel records are inserted; existing records are skipped without duplication.
5. **Auditing**: Evaluates temporal maturity across unique calendar weeks and reports status without fabricating historical data.

---

## 3. Temporal Warm-Up Rules

| Distinct Calendar Weeks in Store | Lag-1 Features ($t-1$) | Rolling 4-Week Averages ($t-4 \dots t-1$) | 4-Week OLS Trends ($t-4 \dots t-1$) | Ingestion State |
| :---: | :---: | :---: | :---: | :--- |
| **1 Week** (e.g. Sessions on 2026-08-15) | `NaN` (Warm-up) | `NaN` (Warm-up) | `NaN` (Warm-up) | `WARMUP / INSUFFICIENT_HISTORICAL_WEEKS` |
| **2 to 4 Weeks** | Active | `NaN` (Warm-up) | `NaN` (Warm-up) | `PARTIAL_HISTORY` |
| **$\ge 5$ Weeks** | Active | Active | Active | `READY` |

Zero artificial history is backfilled. The system preserves absolute mathematical alignment with the training dataset constraints.
