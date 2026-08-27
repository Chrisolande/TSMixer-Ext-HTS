import json
import os

import pandas as pd


def compute_overdispersion_stats(data_dir="m5_data", output_json_path="artifacts/chapter1/overdispersion_statistics.json"):
    sales_path = os.path.join(data_dir, "sales_train_evaluation.csv")
    sales = pd.read_csv(sales_path)
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    
    means = sales[d_cols].mean(axis=1)
    vars_ = sales[d_cols].var(axis=1)
    vmrs = vars_ / (means + 1e-8)
    
    summary = {
        "mean_vmr": float(vmrs.mean()),
        "median_vmr": float(vmrs.median()),
        "prop_vmr_gt_1": float((vmrs > 1.0).mean()),
        "prop_vmr_gt_2": float((vmrs > 2.0).mean()),
        "max_vmr": float(vmrs.max())
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary
