import json
import os
import pandas as pd

def run_data_profiling(data_dir="m5_data", output_json_path="artifacts/chapter1/data_profile.json"):
    sales_path = os.path.join(data_dir, "sales_train_evaluation.csv")
    cal_path = os.path.join(data_dir, "calendar.csv")
    prices_path = os.path.join(data_dir, "sell_prices.csv")
    
    sales = pd.read_csv(sales_path)
    cal = pd.read_csv(cal_path)
    prices = pd.read_csv(prices_path)
    
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    
    profile = {
        "num_bottom_series": int(len(sales)),
        "num_days": int(len(d_cols)),
        "num_stores": int(sales["store_id"].nunique()),
        "num_states": int(sales["state_id"].nunique()),
        "num_items": int(sales["item_id"].nunique()),
        "num_categories": int(sales["cat_id"].nunique()),
        "num_departments": int(sales["dept_id"].nunique()),
        "total_hierarchy_nodes": 42840,
        "forecast_horizon": 28
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(profile, f, indent=2)
        
    return profile
