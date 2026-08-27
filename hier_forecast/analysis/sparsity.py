import json
import os

import pandas as pd


def extract_sparsity_and_traces(data_dir="m5_data",
                                output_sparsity_path="artifacts/chapter1/sparsity_statistics.json",
                                output_traces_path="artifacts/chapter1/representative_traces.json"):
    sales_path = os.path.join(data_dir, "sales_train_evaluation.csv")
    sales = pd.read_csv(sales_path)
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    
    zero_props = (sales[d_cols] == 0).mean(axis=1)
    means = sales[d_cols].mean(axis=1)
    
    sparsity_summary = {
        "mean_zero_prop": float(zero_props.mean()),
        "median_zero_prop": float(zero_props.median()),
        "pct_gt_50": float((zero_props > 0.50).mean()),
        "pct_gt_75": float((zero_props > 0.75).mean()),
        "pct_gt_90": float((zero_props > 0.90).mean())
    }
    
    # Representative traces selection
    high_vol_idx = means.idxmax()
    mod_idx = (zero_props - 0.40).abs().idxmin()
    interm_idx = zero_props[means > 0.05].idxmax()
    
    selected = [
        ("High Volume (" + str(sales.loc[high_vol_idx, "id"]) + ")", high_vol_idx),
        ("Moderate Demand (" + str(sales.loc[mod_idx, "id"]) + ")", mod_idx),
        ("Highly Intermittent (" + str(sales.loc[interm_idx, "id"]) + ")", interm_idx)
    ]
    
    traces = []
    days = [int(x) for x in range(1, 181)]
    for label, idx in selected:
        vals = [int(v) for v in sales.loc[idx, d_cols[:180]].values]
        traces.append({
            "label": label,
            "days": days,
            "sales": vals
        })
        
    os.makedirs(os.path.dirname(output_sparsity_path), exist_ok=True)
    with open(output_sparsity_path, "w") as f:
        json.dump(sparsity_summary, f, indent=2)
    with open(output_traces_path, "w") as f:
        json.dump(traces, f, indent=2)
        
    return sparsity_summary, traces
