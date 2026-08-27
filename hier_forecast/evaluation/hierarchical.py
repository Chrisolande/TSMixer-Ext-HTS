
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from hier_forecast.data_processing.dataset import M5Dataset
from hier_forecast.evaluation.probabilistic import discrete_crps_nb, empirical_coverage
from hier_forecast.models.distribution import NegativeBinomial


class M5WRMSSEMetric:
    """Compute Weighted Root Mean Squared Scaled Error across M5 hierarchy."""

    def __init__(self, aggregation_matrix, weights, scaling_factors, device=None):
        if device is not None and isinstance(device, (str, torch.device)) and str(device) != "cpu":
            self.device = torch.device(device)
            self.use_gpu = True

            if isinstance(aggregation_matrix, torch.Tensor):
                self.S_tensor = aggregation_matrix.to(self.device)
            else:
                if sp.issparse(aggregation_matrix):
                    coo = aggregation_matrix.tocoo()
                    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.int64)
                    values = torch.tensor(coo.data, dtype=torch.float32)
                    shape = torch.Size(coo.shape)
                    self.S_tensor = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
                else:
                    self.S_tensor = torch.tensor(aggregation_matrix, dtype=torch.float32, device=self.device)

            if isinstance(weights, torch.Tensor):
                self.W_tensor = weights.to(self.device)
            else:
                self.W_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

            if isinstance(scaling_factors, torch.Tensor):
                self.C_tensor = scaling_factors.to(self.device)
            else:
                self.C_tensor = torch.tensor(
                    np.clip(scaling_factors, a_min=1e-8, a_max=None), dtype=torch.float32, device=self.device
                )
        else:
            self.use_gpu = False
            if not sp.issparse(aggregation_matrix):
                self.S = sp.csr_matrix(aggregation_matrix)
            else:
                self.S = aggregation_matrix
            self.W = weights
            self.C = np.clip(scaling_factors, a_min=1e-8, a_max=None)

    def compute(self, y_true_bottom, y_pred_bottom):
        """Compute WRMSSE score given bottom-level true and predicted values."""
        if self.use_gpu:
            if not isinstance(y_true_bottom, torch.Tensor):
                y_true_tensor = torch.tensor(y_true_bottom, dtype=torch.float32, device=self.device)
            else:
                y_true_tensor = y_true_bottom.to(dtype=torch.float32, device=self.device)

            if not isinstance(y_pred_bottom, torch.Tensor):
                y_pred_tensor = torch.tensor(y_pred_bottom, dtype=torch.float32, device=self.device)
            else:
                y_pred_tensor = y_pred_bottom.to(dtype=torch.float32, device=self.device)

            if self.S_tensor.is_sparse:
                y_true_all = torch.sparse.mm(self.S_tensor, y_true_tensor)
                y_pred_all = torch.sparse.mm(self.S_tensor, y_pred_tensor)
            else:
                y_true_all = torch.matmul(self.S_tensor, y_true_tensor)
                y_pred_all = torch.matmul(self.S_tensor, y_pred_tensor)

            squared_errors = (y_true_all - y_pred_all) ** 2
            mean_squared_errors = torch.mean(squared_errors, dim=1)
            rmsse = torch.sqrt(mean_squared_errors / self.C_tensor)
            wrmsse = torch.sum(self.W_tensor * rmsse)
            return wrmsse.item()
        else:
            y_true_all = self.S.dot(y_true_bottom)
            y_pred_all = self.S.dot(y_pred_bottom)

            y_true_all = np.asarray(y_true_all)
            y_pred_all = np.asarray(y_pred_all)

            squared_errors = (y_true_all - y_pred_all) ** 2
            mean_squared_errors = np.mean(squared_errors, axis=1)

            rmsse = np.sqrt(mean_squared_errors / self.C)
            wrmsse = np.sum(self.W * rmsse)
            return wrmsse


def compute_m5_scaling_factors(S, train_sales_matrix):
    """Compute scale factors for each hierarchy node over active training sales."""
    aggregated = S.dot(train_sales_matrix)
    y = aggregated.toarray() if sp.issparse(aggregated) else np.asarray(aggregated, dtype=np.float32)

    has_sales = y > 0
    first_idx = np.where(has_sales.any(axis=1), has_sales.argmax(axis=1), 0)

    cols = np.arange(y.shape[1])
    diff_mask = cols[:, None] > first_idx
    diffs_sq = np.diff(y, axis=1).T ** 2

    active_counts = np.maximum(diff_mask.sum(axis=0) - 1, 1)
    scale = (diffs_sq * diff_mask[:-1]).sum(axis=0) / active_counts
    return np.where(scale > 1e-5, scale, 1.0).astype(np.float32)


def evaluate_wrmsse(data_dict: dict, model, device: str = "cpu", split: str = "val") -> float:
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


def evaluate_multi_window_wrmsse(
    data_dict: dict,
    model,
    origins: list[int] | None = None,
    val_window_days: int = 28,
    L: int = 35,
    T: int = 28,
    device: str = "cpu",
) -> dict:
    """Evaluate multi-window rolling-origin WRMSSE and probabilistic metrics across >= 3 origins."""
    if origins is None:
        origins = [1857, 1885, 1913]

    dev = torch.device(device) if isinstance(device, str) else device
    if model is not None:
        model.eval()

    sales_matrix = data_dict["sales_matrix"]
    total_days = sales_matrix.shape[1]

    S = data_dict.get("S_matrix")
    weights = data_dict.get("weights")
    scaling_factors = data_dict.get("scaling_factors")

    wrmsse_metric = M5WRMSSEMetric(S, weights, scaling_factors, device=dev) if S is not None else None

    origin_results = {}
    wrmsse_list = []
    crps_list = []
    coverage_list = []

    for origin in origins:
        eff_origin = min(origin, total_days - T)
        assert eff_origin - L >= 0, f"Origin {eff_origin} is too small for lookback {L}"
        assert eff_origin + T <= total_days, f"Origin {eff_origin} + horizon {T} exceeds total_days {total_days}"

        if model is None:
            continue

        sales_tensor = torch.from_numpy(sales_matrix).float() if isinstance(sales_matrix, np.ndarray) else sales_matrix
        x_sales = sales_tensor[:, eff_origin - L : eff_origin].unsqueeze(-1)
        y_true = sales_tensor[:, eff_origin : eff_origin + T]

        static_cats = data_dict["static_cats"]
        static_cont = data_dict["static_cont"]
        s_cat = torch.from_numpy(static_cats).long() if isinstance(static_cats, np.ndarray) else static_cats
        s_cont = torch.from_numpy(static_cont).float() if isinstance(static_cont, np.ndarray) else static_cont

        exog = data_dict["exog_matrix"]
        exog_tensor = torch.from_numpy(exog).float() if isinstance(exog, np.ndarray) else exog
        hist_exog = exog_tensor[:, eff_origin - L : eff_origin]
        futr_exog = exog_tensor[:, eff_origin : eff_origin + T]

        with torch.no_grad():
            x_sales_d = x_sales.to(dev)
            hist_exog_d = hist_exog.to(dev)
            futr_exog_d = futr_exog.to(dev)
            s_cat_d = s_cat.to(dev)
            s_cont_d = s_cont.to(dev)

            mu, alpha = model(x_sales_d, hist_exog_d, futr_exog_d, s_cat=s_cat_d, s_cont=s_cont_d)
            if mu.ndim == 3 and mu.shape[-1] == 1:
                mu = mu.squeeze(-1)
            if alpha.ndim == 3 and alpha.shape[-1] == 1:
                alpha = alpha.squeeze(-1)

        y_pred = mu.cpu()
        y_true_t = y_true.to(dev)

        wrmsse_val = float(wrmsse_metric.compute(y_true_t, mu)) if wrmsse_metric is not None else 0.0
        y_true_np = y_true.numpy()
        mu_np = y_pred.numpy()
        alpha_np = alpha.cpu().numpy()

        crps_val = discrete_crps_nb(y_true_np, mu_np, alpha_np)

        dist = NegativeBinomial(mu=mu_np, alpha=alpha_np)
        q10 = dist.ppf(0.10)
        q90 = dist.ppf(0.90)
        cov_val = empirical_coverage(y_true_np, q10, q90)

        wrmsse_list.append(wrmsse_val)
        crps_list.append(crps_val)
        coverage_list.append(cov_val)

        origin_results[f"origin_{origin}"] = {
            "wrmsse": wrmsse_val,
            "crps": crps_val,
            "coverage_80": cov_val,
        }

    return {
        "origins": origin_results,
        "wrmsse_mean": float(np.mean(wrmsse_list)) if wrmsse_list else 0.0,
        "wrmsse_std": float(np.std(wrmsse_list)) if wrmsse_list else 0.0,
        "crps_mean": float(np.mean(crps_list)) if crps_list else 0.0,
        "crps_std": float(np.std(crps_list)) if crps_list else 0.0,
        "coverage_80_mean": float(np.mean(coverage_list)) if coverage_list else 0.0,
    }
