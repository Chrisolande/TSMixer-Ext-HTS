import json
import os

import pandas as pd


def analyze_exogenous_associations(data_dir="m5_data", output_json_path="artifacts/chapter1/covariate_statistics.json"):
    sales_path = os.path.join(data_dir, "sales_train_evaluation.csv")
    cal_path = os.path.join(data_dir, "calendar.csv")
    
    sales = pd.read_csv(sales_path)
    cal = pd.read_csv(cal_path)
    
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    cal_subset = cal[cal["d"].isin(d_cols)][["d", "snap_CA", "snap_TX", "snap_WI"]].copy()
    
    state_sales = sales.groupby("state_id")[d_cols].sum().T.reset_index()
    state_sales.rename(columns={"index": "d"}, inplace=True)
    
    merged = state_sales.merge(cal_subset, on="d")
    
    records = []
    for state in ["CA", "TX", "WI"]:
        snap_col = f"snap_{state}"
        grouped = merged.groupby(snap_col)[state].mean()
        
        records.append({
            "state_id": f"{state} (State)",
            "snap_status": "Non-SNAP Day",
            "mean_sales": float(grouped.get(0, 0.0))
        })
        records.append({
            "state_id": f"{state} (State)",
            "snap_status": "SNAP Active Day",
            "mean_sales": float(grouped.get(1, 0.0))
        })
        
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(records, f, indent=2)
    return records
