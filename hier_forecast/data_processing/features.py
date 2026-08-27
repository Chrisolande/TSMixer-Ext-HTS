import glob
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from hier_forecast.data_processing.constants import CAT_COLS, EVENT_TYPE_TO_CODE


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


def build_calendar(
    calendar: pd.DataFrame,
    train_days: int | None = None,
    scaler_mean: np.ndarray | None = None,
    scaler_scale: np.ndarray | None = None,
    return_scaler: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if scaler_mean is not None and scaler_scale is not None:
        mean_arr = np.asarray(scaler_mean, dtype=np.float32)[:3]
        scale_arr = np.asarray(scaler_scale, dtype=np.float32)[:3]
        date_features = (date_features - mean_arr) / np.maximum(scale_arr, 1e-5)
    else:
        scaler = StandardScaler()
        effective_days = train_days if train_days is not None else len(date_features)
        effective_days = min(effective_days, len(date_features))
        scaler.fit(date_features[:effective_days])
        date_features = scaler.transform(date_features)
        mean_arr = scaler.mean_.astype(np.float64)
        scale_arr = scaler.scale_.astype(np.float64)

    snap = calendar[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(dtype=np.float32)

    covariates = np.concatenate([date_features, event_codes, snap], axis=1).astype(np.float32)
    if return_scaler:
        return covariates, mean_arr, scale_arr
    return covariates


def align_prices(prices: pd.DataFrame, calendar: pd.DataFrame, sales_keys: pd.DataFrame, d_cols: list) -> np.ndarray:
    """Pivot weekly prices onto daily timeline and align rows to sales_keys."""
    week_and_day = calendar[["wm_yr_wk", "d"]]
    price_daily_long = prices.merge(week_and_day, on="wm_yr_wk", how="left")

    price_daily_wide = price_daily_long.pivot_table(
        values="sell_price", index=["store_id", "item_id"], columns="d", aggfunc="mean"
    ).reset_index()

    price_aligned = sales_keys.merge(price_daily_wide, on=["store_id", "item_id"], how="left")
    return price_aligned.reindex(columns=d_cols).to_numpy()


def price_zscore(
    price_matrix: np.ndarray,
    train_days: int,
    group_ids: np.ndarray | None = None,
    return_stats: bool = False,
):
    """Train-fitted z-score normalization of a price matrix."""
    train_block = price_matrix[:, :train_days]

    if group_ids is None:
        mean = np.nanmean(train_block, axis=1, keepdims=True)
        std = np.nanstd(train_block, axis=1, keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
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
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)

    std = np.clip(std, a_min=1e-5, a_max=None)
    normalized = (price_matrix - mean) / (std + 1e-6)
    normalized = np.nan_to_num(normalized)

    train_part = normalized[:, :train_days]
    global_mean = float(train_part.mean())
    global_std = float(train_part.std() + 1e-5)
    normalized_out = (normalized - global_mean) / global_std
    if return_stats:
        return normalized_out, {
            "mean": mean,
            "std": std,
            "global_mean": global_mean,
            "global_std": global_std,
        }
    return normalized_out


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
