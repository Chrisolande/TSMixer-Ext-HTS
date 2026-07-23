import glob
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from loguru import logger
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import Dataset

from tsmixer_m5.wrmsse import compute_m5_scaling_factors

CAT_COLS = ["state_id", "store_id", "cat_id", "dept_id", "item_id"]

EVENT_TYPE_TO_CODE = {
    "Cultural": 1,
    "National": 2,
    "Religious": 3,
    "Sporting": 4,
}


def check_files(data_dir: str):
    """Verify raw M5 dataset files exist in the data directory."""
    calendar_path = os.path.join(data_dir, "calendar.csv")
    prices_path = os.path.join(data_dir, "sell_prices.csv")
    sales_matches = glob.glob(os.path.join(data_dir, "*sales_train_evaluation.csv"))
    if not sales_matches:
        sales_matches = glob.glob(os.path.join(data_dir, "*sales_train_validation.csv"))

    missing = [
        name
        for name, exists in [
            ("calendar.csv", os.path.exists(calendar_path)),
            ("sell_prices.csv", os.path.exists(prices_path)),
            ("sales_train_(evaluation|validation).csv", bool(sales_matches)),
        ]
        if not exists
    ]
    if missing:
        raise FileNotFoundError(f"Missing required M5 file(s) in '{data_dir}': {', '.join(missing)}. ")
    return calendar_path, prices_path, sales_matches[0]


def encode_cats(sales: pd.DataFrame):
    """Ordinal-encode static categorical columns and return category labels."""
    encoder = OrdinalEncoder(dtype=np.int64)
    codes = encoder.fit_transform(sales[CAT_COLS])
    sales = sales.copy()
    sales[CAT_COLS] = codes
    cat_mappings = {col: list(cats) for col, cats in zip(CAT_COLS, encoder.categories_)}
    cat_cardinalities = [len(cats) for cats in encoder.categories_]
    return sales, cat_mappings, cat_cardinalities


def build_calendar(calendar: pd.DataFrame) -> np.ndarray:
    """Build standardized calendar, event, and SNAP covariate matrix."""
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])

    event_codes = np.stack(
        [
            calendar["event_type_1"].map(EVENT_TYPE_TO_CODE).fillna(0).to_numpy(dtype=np.float32),
            calendar["event_type_2"].map(EVENT_TYPE_TO_CODE).fillna(0).to_numpy(dtype=np.float32),
        ],
        axis=1,
    )

    date_features = np.stack(
        [
            calendar["wday"].to_numpy(dtype=np.float32),
            calendar["date"].dt.day.to_numpy(dtype=np.float32),
            calendar["date"].dt.dayofyear.to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    date_features = StandardScaler().fit_transform(date_features)

    snap = calendar[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(dtype=np.float32)

    return np.concatenate([date_features, event_codes, snap], axis=1)


def align_prices(prices: pd.DataFrame, calendar: pd.DataFrame, sales_keys: pd.DataFrame, d_cols: list) -> np.ndarray:
    """Pivot weekly prices onto daily timeline and align rows to sales_keys."""
    week_and_day = calendar[["wm_yr_wk", "d"]]
    price_daily_long = prices.merge(week_and_day, on="wm_yr_wk", how="left")

    price_daily_wide = price_daily_long.pivot_table(
        values="sell_price", index=["store_id", "item_id"], columns="d", aggfunc="mean"
    ).reset_index()

    price_aligned = sales_keys.merge(price_daily_wide, on=["store_id", "item_id"], how="left")
    return price_aligned[d_cols].to_numpy()


def price_zscore(price_matrix: np.ndarray, train_days: int, group_ids: np.ndarray | None = None) -> np.ndarray:
    """Train-fitted z-score normalization of a price matrix."""
    train_block = price_matrix[:, :train_days]

    if group_ids is None:
        mean = np.nanmean(train_block, axis=1, keepdims=True)
        std = np.nanstd(train_block, axis=1, keepdims=True)
    else:
        stats = (
            pd.DataFrame(train_block, index=group_ids)
            .stack(future_stack=True)
            .dropna()
            .groupby(level=0)
            .agg(["mean", "std"])
        )
        mean = stats["mean"].reindex(group_ids).to_numpy()[:, None]
        std = stats["std"].reindex(group_ids).to_numpy()[:, None]

    std = np.clip(std, a_min=1e-5, a_max=None)
    normalized = (price_matrix - mean) / (std + 1e-6)
    normalized = np.nan_to_num(normalized)

    train_part = normalized[:, :train_days]
    global_mean = train_part.mean()
    global_std = train_part.std() + 1e-5
    return (normalized - global_mean) / global_std


def build_hierarchy(sales: pd.DataFrame):
    """Fast <100ms SciPy hierarchy matrix construction."""
    levels_groups = [
        [],  # Level 0: Total (1)
        ["state_id"],  # Level 1: State (3)
        ["store_id"],  # Level 2: Store (10)
        ["cat_id"],  # Level 3: Cat (3)
        ["dept_id"],  # Level 4: Dept (7)
        ["state_id", "cat_id"],  # Level 5: State x Cat (9)
        ["state_id", "dept_id"],  # Level 6: State x Dept (21)
        ["store_id", "cat_id"],  # Level 7: Store x Cat (30)
        ["store_id", "dept_id"],  # Level 8: Store x Dept (70)
        ["item_id"],  # Level 9: Item (3049)
        ["state_id", "item_id"],  # Level 10: State x Item (9147)
        ["store_id", "item_id"],  # Level 11: Store x Item / Bottom (30490)
    ]

    num_series = len(sales)
    col_idx = np.arange(num_series)
    ones = np.ones(num_series, dtype=np.float32)

    s_blocks = []
    node_ids = []
    tags = {}

    for cols in levels_groups:
        if not cols:
            keys = pd.Series(["Total"] * num_series)
            lvl_name = "Total"
        else:
            keys = sales[cols].astype(str).agg("_".join, axis=1)
            lvl_name = "_".join(cols)

        codes, categories = pd.factorize(keys)
        S_lvl = sp.csr_matrix((ones, (codes, col_idx)), shape=(len(categories), num_series), dtype=np.float32)

        s_blocks.append(S_lvl)
        node_ids.extend(categories)
        tags[lvl_name] = categories.tolist()

    S_matrix = sp.vstack(s_blocks, format="csr").astype(np.float32)
    return S_matrix, pd.Index(node_ids), tags


def compute_weights(
    S_matrix,
    node_ids: pd.Index,
    tags: dict,
    sales_keys: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    train_sales: np.ndarray,
    effective_train_days: int,
) -> np.ndarray:
    """Compute dollar-revenue hierarchy weights across all levels."""
    last_28_train_sales = train_sales[:, -28:]

    last_train_wk = calendar.loc[calendar["d"] == f"d_{effective_train_days}", "wm_yr_wk"]
    target_wk = last_train_wk.iloc[0] if not last_train_wk.empty else prices["wm_yr_wk"].min()

    prices_last_wk = prices.loc[prices["wm_yr_wk"] == target_wk, ["store_id", "item_id", "sell_price"]]
    item_prices = (
        sales_keys.merge(prices_last_wk, on=["store_id", "item_id"], how="left")["sell_price"]
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    bottom_revenue = last_28_train_sales.sum(axis=1) * item_prices

    num_levels = len(tags)
    weights = np.zeros(S_matrix.shape[0], dtype=np.float32)
    for level_ids in tags.values():
        row_idx = node_ids.get_indexer(level_ids)
        lvl_revenue = S_matrix[row_idx].dot(bottom_revenue)
        total_lvl_revenue = lvl_revenue.sum()

        if total_lvl_revenue > 0:
            weights[row_idx] = (1.0 / num_levels) * (lvl_revenue / total_lvl_revenue)
        else:
            weights[row_idx] = (1.0 / num_levels) / len(row_idx)

    return weights


def preprocess_m5(data_dir: str, train_days: int = 1886):
    """Load and preprocess M5 paper features and hierarchy matrices."""
    calendar_path, prices_path, sales_path = check_files(data_dir)

    logger.info("Loading & preprocessing M5 datasets from {data_dir}", data_dir=data_dir)
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)
    sales_raw = pd.read_csv(sales_path)

    d_cols = sorted((c for c in sales_raw.columns if c.startswith("d_")), key=lambda x: int(x.split("_")[1]))
    sales_matrix = sales_raw[d_cols].to_numpy(dtype=np.float32)
    effective_train_days = min(train_days, sales_matrix.shape[1])
    train_sales_matrix = sales_matrix[:, :effective_train_days]

    static_cont = np.mean(train_sales_matrix, axis=1, keepdims=True).astype(np.float32)
    sales, cat_mappings, cat_cardinalities = encode_cats(sales_raw)
    static_cats = sales[CAT_COLS].to_numpy()

    future_exog_matrix = build_calendar(calendar)

    sales_keys = sales_raw[["store_id", "item_id"]]
    price_matrix = align_prices(prices, calendar, sales_keys, d_cols)

    norm_price_item = price_zscore(price_matrix, effective_train_days)
    norm_price_group = price_zscore(price_matrix, effective_train_days, group_ids=sales_raw["dept_id"].to_numpy())

    S_matrix, node_ids, tags = build_hierarchy(sales_raw)
    scaling_factors = compute_m5_scaling_factors(S_matrix, train_sales_matrix)
    weights = compute_weights(
        S_matrix,
        node_ids,
        tags,
        sales_keys,
        prices,
        calendar,
        train_sales_matrix,
        effective_train_days,
    )

    norm_prices = np.stack([norm_price_item, norm_price_group], axis=-1).astype(np.float32)
    num_series, num_days = sales_matrix.shape
    futr_exog_matched = future_exog_matrix[:num_days, :]
    exog_matrix = np.concatenate(
        [np.repeat(futr_exog_matched[None, :, :], num_series, axis=0), norm_prices], axis=-1
    ).astype(np.float32)

    return {
        "sales_matrix": sales_matrix,
        "static_cats": static_cats,
        "static_cont": static_cont,
        "future_exog": future_exog_matrix,
        "norm_price_item": norm_price_item,
        "norm_price_group": norm_price_group,
        "norm_prices": norm_prices,
        "exog_matrix": exog_matrix,
        "cat_cardinalities": cat_cardinalities,
        "S_matrix": S_matrix,
        "weights": weights,
        "scaling_factors": scaling_factors,
    }


class M5Dataset(Dataset):
    """PyTorch Dataset for sliding-window and stochastic sampling over M5 time series."""

    def __init__(
        self,
        data_dict,
        L=35,
        T=28,
        stride=7,
        split="train",
        is_train=None,
        val_window_days=84,
        test_days=28,
        val_days=28,
        stochastic=False,
        samples_per_epoch=None,
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

    def __len__(self):
        if self.stochastic and self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.window_indices)

    def __getitem__(self, idx):
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
