import numpy as np
import pandas as pd
from loguru import logger

from hier_forecast.data_processing.constants import CAT_COLS
from hier_forecast.data_processing.features import (
    align_prices,
    build_calendar,
    build_hierarchy,
    check_files,
    compute_weights,
    encode_cats,
    price_zscore,
)
from hier_forecast.evaluation.hierarchical import compute_m5_scaling_factors


def preprocess_m5(data_dir: str, train_days: int = 1886) -> dict:
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

    static_cont = np.nan_to_num(np.nanmean(train_sales_matrix, axis=1, keepdims=True), nan=0.0).astype(np.float32)
    sales, cat_mappings, cat_cardinalities = encode_cats(sales_raw)
    static_cats = sales[CAT_COLS].to_numpy()

    future_exog_matrix, cal_scaler_mean, cal_scaler_scale = build_calendar(
        calendar, train_days=effective_train_days, return_scaler=True
    )

    sales_keys = sales_raw[["store_id", "item_id"]]
    price_matrix = align_prices(prices, calendar, sales_keys, d_cols)

    norm_price_item, item_pstats = price_zscore(price_matrix, effective_train_days, return_stats=True)
    norm_price_group, group_pstats = price_zscore(
        price_matrix, effective_train_days, group_ids=sales_raw["dept_id"].to_numpy(), return_stats=True
    )

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
    num_series = sales_matrix.shape[0]
    num_days = min(sales_matrix.shape[1], len(future_exog_matrix))
    sales_matrix = sales_matrix[:, :num_days]
    norm_prices = norm_prices[:, :num_days, :]
    futr_exog_matched = future_exog_matrix[:num_days, :]
    exog_matrix = np.concatenate(
        [np.repeat(futr_exog_matched[None, :, :], num_series, axis=0), norm_prices], axis=-1
    ).astype(np.float32)

    canonical_cat_maps = {
        col: {str(label): int(code) for code, label in enumerate(cat_mappings[col])}
        for col in CAT_COLS
    }

    series_price_stats = {}
    for i, row in sales_keys.iterrows():
        key = f"{row['store_id']}_{row['item_id']}"
        series_price_stats[key] = {
            "mean": float(item_pstats["mean"][i, 0]),
            "std": float(item_pstats["std"][i, 0]),
        }
    dept_price_stats = {}
    for _i, row in sales_raw[["store_id", "dept_id"]].drop_duplicates().iterrows():
        key = f"{row['store_id']}_{row['dept_id']}"
        match_rows = sales_raw[(sales_raw["store_id"] == row["store_id"]) & (sales_raw["dept_id"] == row["dept_id"])]
        if not match_rows.empty:
            idx = match_rows.index[0]
            dept_price_stats[key] = {
                "mean": float(group_pstats["mean"][idx, 0]),
                "std": float(group_pstats["std"][idx, 0]),
            }

    price_stats = {
        "series_price_stats": series_price_stats,
        "dept_price_stats": dept_price_stats,
        "global_item_mean": item_pstats["global_mean"],
        "global_item_std": item_pstats["global_std"],
        "global_dept_mean": group_pstats["global_mean"],
        "global_dept_std": group_pstats["global_std"],
    }

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
        "category_maps": canonical_cat_maps,
        "calendar_scaler_mean": cal_scaler_mean,
        "calendar_scaler_scale": cal_scaler_scale,
        "price_stats": price_stats,
        "S_matrix": S_matrix,
        "weights": weights,
        "scaling_factors": scaling_factors,
    }
