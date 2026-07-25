import numpy as np
import scipy.sparse as sp
import torch


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
