import json
import os

import numpy as np
import pandas as pd


def compute_distribution_statistics(data_dir="m5_data", output_json_path="artifacts/chapter1/distribution_statistics.json"):
    sales_path = os.path.join(data_dir, "sales_train_evaluation.csv")
    sales = pd.read_csv(sales_path)
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    
    l1 = sales[d_cols].sum(axis=0)
    l2 = sales.groupby("state_id")[d_cols].sum()
    l3 = sales.groupby("store_id")[d_cols].sum()
    l4 = sales.groupby("cat_id")[d_cols].sum()
    l5 = sales.groupby("dept_id")[d_cols].sum()
    l12 = sales[d_cols]
    
    levels = [
        ("Level 1 (Total)", l1),
        ("Level 2 (State)", l2),
        ("Level 3 (Store)", l3),
        ("Level 4 (Category)", l4),
        ("Level 5 (Department)", l5),
        ("Level 12 (Item-Store)", l12)
    ]
    
    records = []
    for name, data in levels:
        if isinstance(data, pd.Series):
            arr = data.values
            m = float(arr.mean())
            med = float(np.median(arr))
            v = float(arr.var())
            vmr = float(v / (m + 1e-8))
            zp = float((arr == 0).mean())
            cv = float(arr.std() / (m + 1e-8))
            sk = float(pd.Series(arr).skew())
        else:
            means = data.mean(axis=1)
            meds = data.median(axis=1)
            vars_ = data.var(axis=1)
            zps = (data == 0).mean(axis=1)
            cvs = data.std(axis=1) / (means + 1e-8)
            skews = data.skew(axis=1)
            vmrs = vars_ / (means + 1e-8)
            
            m = float(means.mean())
            med = float(meds.mean())
            v = float(vars_.mean())
            vmr = float(vmrs.mean())
            zp = float(zps.mean())
            cv = float(cvs.mean())
            sk = float(skews.mean())
            
        records.append({
            "level": name,
            "mean": m,
            "median": med,
            "variance": v,
            "vmr": vmr,
            "zero_prop": zp,
            "cv": cv,
            "skewness": sk
        })
        
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(records, f, indent=2)
    return records
