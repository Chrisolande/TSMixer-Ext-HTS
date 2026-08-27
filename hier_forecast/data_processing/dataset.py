import numpy as np
import torch
from torch.utils.data import Dataset


class M5Dataset(Dataset):
    """PyTorch Dataset for sliding-window and stochastic sampling over M5 time series."""

    def __init__(
        self,
        data_dict: dict,
        L: int = 35,
        T: int = 28,
        stride: int = 7,
        split: str = "train",
        is_train: bool | None = None,
        val_window_days: int = 84,
        test_days: int = 28,
        val_days: int = 28,
        stochastic: bool = False,
        samples_per_epoch: int | None = None,
    ):
        sales_mat = data_dict["sales_matrix"]
        self.sales = torch.from_numpy(sales_mat).float() if isinstance(sales_mat, np.ndarray) else sales_mat
        self.num_series, self.total_days = self.sales.shape

        static_c = data_dict["static_cats"]
        self.static_cats = torch.from_numpy(static_c).long() if isinstance(static_c, np.ndarray) else static_c

        static_n = data_dict["static_cont"]
        self.static_cont = torch.from_numpy(static_n).float() if isinstance(static_n, np.ndarray) else static_n

        if "exog_matrix" in data_dict:
            exog_mat = data_dict["exog_matrix"]
            self.exog = torch.from_numpy(exog_mat).float() if isinstance(exog_mat, np.ndarray) else exog_mat
        else:
            futr_ex = data_dict["future_exog"][: self.total_days]
            futr_tensor = torch.from_numpy(futr_ex).float() if isinstance(futr_ex, np.ndarray) else futr_ex
            n_prices = data_dict.get("norm_prices")
            if n_prices is None:
                n_prices = np.stack([data_dict["norm_price_item"], data_dict["norm_price_group"]], axis=-1).astype(
                    np.float32
                )
            prices_tensor = torch.from_numpy(n_prices).float() if isinstance(n_prices, np.ndarray) else n_prices
            cal_expanded = futr_tensor.unsqueeze(0).expand(self.num_series, -1, -1)
            self.exog = torch.cat([cal_expanded, prices_tensor], dim=-1)

        self.L = L
        self.T = T

        if is_train is not None:
            split = "train" if is_train else "val"
        self.split = split
        self.stochastic = stochastic
        self.samples_per_epoch = samples_per_epoch

        if split == "train":
            self.day_start = 0
            self.day_end = max(0, self.total_days - test_days - val_days)
        elif split == "val":
            self.day_start = max(0, self.total_days - test_days - val_window_days - L)
            self.day_end = self.total_days - test_days
        elif split == "test":
            self.day_start = max(0, self.total_days - test_days - L)
            self.day_end = self.total_days
        else:
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

        days = list(range(self.day_start, self.day_end - L - T + 1, stride))
        num_days = len(days)
        if num_days > 0:
            d_arr = np.repeat(np.array(days, dtype=np.int32), self.num_series)
            s_arr = np.tile(np.arange(self.num_series, dtype=np.int32), num_days)
            self.window_indices = np.stack([s_arr, d_arr], axis=1)
        else:
            self.window_indices = np.empty((0, 2), dtype=np.int32)

    def __len__(self) -> int:
        if self.stochastic and self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.window_indices)

    def __getitem__(self, idx: int):
        if self.stochastic and self.samples_per_epoch is not None:
            s_idx = np.random.randint(0, self.num_series)
            d_start = np.random.randint(self.day_start, max(self.day_start + 1, self.day_end - self.L - self.T + 1))
        else:
            s_idx, d_start = self.window_indices[idx]
            s_idx, d_start = int(s_idx), int(d_start)

        x_sales_tensor = self.sales[s_idx, d_start : d_start + self.L].unsqueeze(-1)
        y_sales_tensor = self.sales[s_idx, d_start + self.L : d_start + self.L + self.T].unsqueeze(-1)

        s_cat = self.static_cats[s_idx]
        s_cont = self.static_cont[s_idx]

        hist_exog = self.exog[s_idx, d_start : d_start + self.L]
        futr_exog = self.exog[s_idx, d_start + self.L : d_start + self.L + self.T]

        return x_sales_tensor, hist_exog, futr_exog, s_cat, s_cont, y_sales_tensor
