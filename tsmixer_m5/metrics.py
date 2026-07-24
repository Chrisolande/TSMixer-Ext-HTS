import numpy as np
import torch
from torch.utils.data import DataLoader

from tsmixer_m5.data import M5Dataset
from tsmixer_m5.wrmsse import M5WRMSSEMetric


def evaluate_wrmsse(data_dict, model, device="cpu", split="val"):
    """Evaluate rolling WRMSSE score over historical validation or test windows."""
    L = 35
    T = 28
    val_window_days = T if split == "val" else 84
    stride = T if split == "val" else 14
    val_dataset = M5Dataset(data_dict, L=L, T=T, stride=stride, split=split, val_window_days=val_window_days)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

    model.eval()
    predictions = []
    ground_truths = []

    dev = torch.device(device) if isinstance(device, str) else device

    with torch.no_grad():
        for bx, bx_hist, bz_futr, bs_cat, bs_cont, by_true in val_loader:
            bx, bx_hist, bz_futr = bx.to(dev), bx_hist.to(dev), bz_futr.to(dev)
            bs_cat, bs_cont = bs_cat.to(dev), bs_cont.to(dev)

            if torch.cuda.is_available() and dev.type == "cuda":
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                ):
                    mu, alpha = model(bx, bx_hist, bz_futr, s_cat=bs_cat, s_cont=bs_cont)
            else:
                mu, alpha = model(bx, bx_hist, bz_futr, s_cat=bs_cat, s_cont=bs_cont)

            predictions.append(mu.squeeze(-1))
            ground_truths.append(by_true.to(dev).squeeze(-1))

    y_pred_bottom = torch.cat(predictions, dim=0)
    y_true_bottom = torch.cat(ground_truths, dim=0)
    num_series = data_dict["sales_matrix"].shape[0]

    num_periods = max(1, y_pred_bottom.shape[0] // num_series)

    S = data_dict["S_matrix"]
    weights = data_dict["weights"]
    scaling_factors = data_dict["scaling_factors"]

    cache_key = f"_wrmsse_metric_cache_{dev}"
    if isinstance(data_dict, dict) and cache_key in data_dict:
        wrmsse_metric = data_dict[cache_key]
    else:
        wrmsse_metric = M5WRMSSEMetric(S, weights, scaling_factors, device=dev)
        if isinstance(data_dict, dict):
            data_dict[cache_key] = wrmsse_metric

    period_scores = []
    for p in range(num_periods):
        start_idx = p * num_series
        end_idx = min((p + 1) * num_series, y_pred_bottom.shape[0])
        y_pred_period = y_pred_bottom[start_idx:end_idx]
        y_true_period = y_true_bottom[start_idx:end_idx]

        score = wrmsse_metric.compute(y_true_period, y_pred_period)
        period_scores.append(score)

    return float(np.mean(period_scores))
