import os
import json
import joblib
import pandas as pd
import numpy as np
import math

def clean_for_json(record):
    """Converts NaN and Infinity to None for valid JSON encoding."""
    clean_rec = {}
    for k, v in record.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean_rec[k] = None
        else:
            clean_rec[k] = v
    return clean_rec

def build_static_api():
    # 1. Setup Output Directory structure
    api_dir = "api/v1"
    os.makedirs(api_dir, exist_ok=True)
    
    print("Loading data and model...")
    df = pd.read_csv("data/simulation_temporal_features.csv")
    
    # Merge location metadata if it exists
    if os.path.exists("data/location_mapping.csv"):
        loc_df = pd.read_csv("data/location_mapping.csv")
        df = pd.merge(df, loc_df, on="zone_id", how="left")
    
    model_path = "models/best_risk_model.pkl"
    model = joblib.load(model_path) if os.path.exists(model_path) else None

    # 2. Generate Master Zones List (/api/v1/zones.json)
    zones = sorted(df["zone_id"].unique().tolist())
    with open(f"{api_dir}/zones.json", "w") as f:
        json.dump({"total_zones": len(zones), "zones": zones}, f, indent=2)
    
    print(f"Generating endpoints for {len(zones)} zones...")

    # 3. Generate endpoints for each Zone
    for zone in zones:
        zone_dir = f"{api_dir}/zones/{zone}"
        os.makedirs(zone_dir, exist_ok=True)
        
        zone_df = df[df["zone_id"] == zone].sort_values("week")
        all_weeks_data = []

        # Iterate through every week for this zone
        for _, row in zone_df.iterrows():
            week = int(row["week"])
            
            # Extract row as dataframe for the ML pipeline
            row_df = zone_df[zone_df["week"] == week]
            
            # Pre-calculate ML model output
            risk_prob = None
            if model and week >= 5:
                risk_prob = float(model.predict_proba(row_df)[0, 1])
                
            record = clean_for_json(row.to_dict())
            
            payload = {
                "zone_id": zone,
                "week": week,
                "predicted_risk_probability": risk_prob,
                "metrics": record
            }
            
            # Save single-week endpoint (e.g., /api/v1/zones/Andheri/50.json)
            with open(f"{zone_dir}/{week}.json", "w") as f:
                json.dump(payload, f, indent=2)
                
            all_weeks_data.append(payload)

        # Save all-weeks endpoint for trend charts (e.g., /api/v1/zones/Andheri/all.json)
        with open(f"{zone_dir}/all.json", "w") as f:
            json.dump({"zone_id": zone, "timeline": all_weeks_data}, f, indent=2)
            
    print(f"Success! API generated at ./{api_dir}/")

if __name__ == "__main__":
    build_static_api()