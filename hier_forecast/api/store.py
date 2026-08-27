import os

import numpy as np
import pandas as pd
import torch
from loguru import logger

from hier_forecast.api.schemas.request import SeriesKey
from hier_forecast.data_processing.bundle import load_preprocess_bundle
from hier_forecast.data_processing.features import build_calendar


class InferenceStore:
    """Manages string-to-int category encoding and 35-day historical sales window assembly with real exogenous features."""

    def __init__(
        self,
        category_maps: dict[str, dict[str, int]] | None = None,
        snapshot_dir: str = "./data/m5_sample",
        bundle_dir: str | None = None,
    ):
        self.snapshot_dir = snapshot_dir
        self.bundle_dir = bundle_dir
        self.category_maps = category_maps or {}
        self.price_stats: dict = {}
        self.calendar_scaler_mean: np.ndarray | None = None
        self.calendar_scaler_scale: np.ndarray | None = None

        if bundle_dir and os.path.exists(bundle_dir):
            try:
                bundle = load_preprocess_bundle(bundle_dir)
                if not self.category_maps and "category_maps" in bundle:
                    self.category_maps = bundle["category_maps"]
                self.price_stats = bundle.get("price_stats", {})
                self.calendar_scaler_mean = bundle.get("calendar_scaler_mean")
                self.calendar_scaler_scale = bundle.get("calendar_scaler_scale")
                logger.info("Loaded preprocessing bundle from {b}", b=bundle_dir)
            except Exception as e:
                logger.warning("Could not load preprocessing bundle from {b}: {e}", b=bundle_dir, e=e)

        self.sales_data: pd.DataFrame | None = None
        self.calendar_df: pd.DataFrame | None = None
        self.prices_df: pd.DataFrame | None = None
        self.calendar_matrix: np.ndarray | None = None
        self.date_to_idx: dict[str, int] = {}
        self.d_to_idx: dict[str, int] = {}

        self.load_snapshot()

    def load_snapshot(self):
        """Load sales, calendar, and price snapshot files if present on disk."""
        cal_path = os.path.join(self.snapshot_dir, "calendar.csv")
        if os.path.exists(cal_path):
            try:
                self.calendar_df = pd.read_csv(cal_path)
                self.calendar_df["date"] = pd.to_datetime(self.calendar_df["date"]).dt.strftime("%Y-%m-%d")
                self.date_to_idx = {d: i for i, d in enumerate(self.calendar_df["date"])}
                self.d_to_idx = {d: i for i, d in enumerate(self.calendar_df["d"])}
                self.calendar_matrix = build_calendar(
                    self.calendar_df,
                    scaler_mean=self.calendar_scaler_mean,
                    scaler_scale=self.calendar_scaler_scale,
                )
                logger.info("Loaded and processed calendar matrix with shape {s}", s=self.calendar_matrix.shape)
            except Exception as e:
                logger.warning("Could not process calendar data: {e}", e=e)

        prices_path = os.path.join(self.snapshot_dir, "sell_prices.csv")
        if os.path.exists(prices_path):
            try:
                self.prices_df = pd.read_csv(prices_path)
                logger.info("Loaded sell_prices data ({n} rows)", n=len(self.prices_df))
            except Exception as e:
                logger.warning("Could not read sell prices CSV: {e}", e=e)

        csv_path = os.path.join(self.snapshot_dir, "sales_train_evaluation.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(self.snapshot_dir, "sales_train_validation.csv")

        if os.path.exists(csv_path):
            try:
                logger.info("Loading sales snapshot data from {p}", p=csv_path)
                self.sales_data = pd.read_csv(csv_path)
            except Exception as e:
                logger.warning("Could not read sales snapshot CSV: {e}", e=e)

    def encode_key(self, store_id: str, item_id: str) -> dict[str, int]:
        """Convert string identifiers to integer categorical indices."""
        store_map = self.category_maps.get("store_id", {})
        item_map = self.category_maps.get("item_id", {})
        dept_map = self.category_maps.get("dept_id", {})
        cat_map = self.category_maps.get("cat_id", {})
        state_map = self.category_maps.get("state_id", {})

        if store_id not in store_map:
            raise KeyError(f"Unknown store_id '{store_id}'")
        if item_id not in item_map:
            raise KeyError(f"Unknown item_id '{item_id}'")

        parts = item_id.split("_")
        dept_str = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "HOBBIES_1"
        cat_str = parts[0] if len(parts) >= 1 else "HOBBIES"
        state_str = store_id.split("_")[0] if "_" in store_id else "CA"

        return {
            "store_idx": store_map[store_id],
            "item_idx": item_map[item_id],
            "dept_idx": dept_map.get(dept_str, 0),
            "cat_idx": cat_map.get(cat_str, 0),
            "state_idx": state_map.get(state_str, 0),
        }

    def fetch_history(self, store_id: str, item_id: str, lookback_days: int = 35, as_of_date: str | None = None) -> np.ndarray:
        """Fetch 35-day historical sales from local snapshot dataset."""
        if self.sales_data is not None:
            mask = (self.sales_data["store_id"] == store_id) & (self.sales_data["item_id"] == item_id)
            matched = self.sales_data[mask]
            if not matched.empty:
                day_cols = [c for c in matched.columns if c.startswith("d_")]
                if as_of_date and as_of_date in self.date_to_idx:
                    as_of_idx = self.date_to_idx[as_of_date]
                    if as_of_idx >= lookback_days and as_of_idx <= len(day_cols):
                        return matched[day_cols[as_of_idx - lookback_days : as_of_idx]].values.flatten().astype(np.float32)
                if len(day_cols) >= lookback_days:
                    return matched[day_cols[-lookback_days:]].values.flatten().astype(np.float32)
                logger.warning(
                    "Not enough day columns ({n}) for lookback={lb}; falling back to ones.",
                    n=len(day_cols),
                    lb=lookback_days,
                )
            else:
                logger.warning(
                    "No snapshot row found for store={s} item={i}; falling back to ones.",
                    s=store_id,
                    i=item_id,
                )

        return np.ones(lookback_days, dtype=np.float32)

    def _get_price_series(self, store_id: str, item_id: str, start_idx: int, total_days: int) -> np.ndarray:
        """Construct normalized item and department price series for the window."""
        if self.calendar_df is None or self.prices_df is None:
            return np.zeros((total_days, 2), dtype=np.float32)

        window_cal = self.calendar_df.iloc[start_idx : start_idx + total_days]
        wm_weeks = window_cal["wm_yr_wk"].tolist()

        item_prices = self.prices_df[
            (self.prices_df["store_id"] == store_id) & (self.prices_df["item_id"] == item_id)
        ]
        week_to_price = dict(zip(item_prices["wm_yr_wk"], item_prices["sell_price"]))

        daily_prices = np.array([week_to_price.get(wk, np.nan) for wk in wm_weeks], dtype=np.float32)
        if np.isnan(daily_prices).all():
            mean_val = 1.0
            daily_prices = np.ones(total_days, dtype=np.float32)
        else:
            mean_val = float(np.nanmean(daily_prices))
            daily_prices = np.nan_to_num(daily_prices, nan=mean_val)

        # Normalize item price
        series_key = f"{store_id}_{item_id}"
        series_stat = self.price_stats.get("series_price_stats", {}).get(series_key, {})
        s_mean = series_stat.get("mean", mean_val)
        s_std = max(series_stat.get("std", 1.0), 1e-5)
        g_mean = self.price_stats.get("global_item_mean", 0.0)
        g_std = max(self.price_stats.get("global_item_std", 1.0), 1e-5)

        norm_item = (((daily_prices - s_mean) / (s_std + 1e-6)) - g_mean) / g_std

        # Normalize dept price
        parts = item_id.split("_")
        dept_str = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "HOBBIES_1"
        dept_key = f"{store_id}_{dept_str}"
        dept_stat = self.price_stats.get("dept_price_stats", {}).get(dept_key, {})
        d_mean = dept_stat.get("mean", s_mean)
        d_std = max(dept_stat.get("std", 1.0), 1e-5)
        dg_mean = self.price_stats.get("global_dept_mean", 0.0)
        dg_std = max(self.price_stats.get("global_dept_std", 1.0), 1e-5)

        norm_dept = (((daily_prices - d_mean) / (d_std + 1e-6)) - dg_mean) / dg_std

        return np.stack([norm_item, norm_dept], axis=-1).astype(np.float32)

    def build_tensors(self, item: SeriesKey, as_of_date: str = "2016-04-25") -> dict[str, torch.Tensor]:
        """Assemble PyTorch input tensors (x, x_hist, z_futr, s_cat, s_cont) for TSMixerExt."""
        encoded = self.encode_key(item.store_id, item.item_id)

        if item.past_sales is not None:
            if len(item.past_sales) != 35:
                raise ValueError(f"Invalid past_sales length {len(item.past_sales)}; expected 35 values.")
            sales = np.array(item.past_sales, dtype=np.float32)
        else:
            sales = self.fetch_history(item.store_id, item.item_id, lookback_days=35, as_of_date=as_of_date)

        x_tensor = torch.from_numpy(sales).unsqueeze(-1).float()

        # Determine date index in calendar
        if as_of_date in self.date_to_idx:
            as_of_idx = self.date_to_idx[as_of_date]
        else:
            # Fallback to day 1913 (end of training phase / start of validation 2016-04-25)
            as_of_idx = 1913 if self.calendar_matrix is not None and len(self.calendar_matrix) > 1941 else 35

        start_idx = max(0, as_of_idx - 35)
        futr_idx = as_of_idx

        if self.calendar_matrix is not None and len(self.calendar_matrix) >= futr_idx + 28:
            hist_cal = self.calendar_matrix[start_idx : start_idx + 35]
            futr_cal = self.calendar_matrix[futr_idx : futr_idx + 28]
        else:
            hist_cal = np.zeros((35, 8), dtype=np.float32)
            futr_cal = np.zeros((28, 8), dtype=np.float32)

        norm_prices = self._get_price_series(item.store_id, item.item_id, start_idx=start_idx, total_days=63)
        hist_price = norm_prices[:35]
        futr_price = norm_prices[35:63]

        x_hist = torch.cat([torch.from_numpy(hist_cal).float(), torch.from_numpy(hist_price).float()], dim=-1)
        z_futr = torch.cat([torch.from_numpy(futr_cal).float(), torch.from_numpy(futr_price).float()], dim=-1)

        s_cat = torch.tensor(
            [
                encoded["state_idx"],
                encoded["store_idx"],
                encoded["cat_idx"],
                encoded["dept_idx"],
                encoded["item_idx"],
            ],
            dtype=torch.long,
        )
        s_cont_val = float(np.nanmean(sales))
        if self.sales_data is not None:
            mask = (self.sales_data["store_id"] == item.store_id) & (self.sales_data["item_id"] == item.item_id)
            matched = self.sales_data[mask]
            if not matched.empty:
                day_cols = [c for c in matched.columns if c.startswith("d_")]
                train_cols = [c for c in day_cols if int(c.split("_")[1]) <= 1913]
                if not train_cols:
                    train_cols = day_cols
                s_cont_val = float(np.nan_to_num(np.nanmean(matched[train_cols].values.astype(float)), nan=s_cont_val))
        s_cont = torch.tensor([s_cont_val], dtype=torch.float32)

        return {
            "x": x_tensor,
            "x_hist": x_hist,
            "z_futr": z_futr,
            "s_cat": s_cat,
            "s_cont": s_cont,
        }
