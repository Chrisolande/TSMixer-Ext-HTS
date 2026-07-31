import os

import numpy as np
import pandas as pd
import torch
from loguru import logger

from tsmixer_m5.api.schemas.request import SeriesKey


class InferenceStore:
    """Manages string-to-int category encoding and 35-day historical sales window assembly."""

    def __init__(self, category_maps: dict[str, dict[str, int]], snapshot_dir: str = "./data/m5_sample"):
        self.category_maps = category_maps
        self.snapshot_dir = snapshot_dir
        self.sales_data: pd.DataFrame | None = None
        self.load_snapshot()

    def load_snapshot(self):
        """Optionally load rolling sales snapshot CSV/Parquet if present on disk."""
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

    def fetch_history(self, store_id: str, item_id: str, lookback_days: int = 35) -> np.ndarray:
        """Fetch 35-day historical sales from local snapshot dataset."""
        if self.sales_data is not None:
            composite_id = f"{item_id}_{store_id}"
            mask = (self.sales_data["store_id"] == store_id) & (
                (self.sales_data["item_id"] == composite_id)
                | (self.sales_data["item_id"] == item_id)
            )
            matched = self.sales_data[mask]
            if not matched.empty:
                day_cols = [c for c in matched.columns if c.startswith("d_")]
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

    def build_tensors(self, item: SeriesKey) -> dict[str, torch.Tensor]:
        """Assemble PyTorch input tensors (x, x_hist, z_futr, s_cat, s_cont) for TSMixerExt."""
        encoded = self.encode_key(item.store_id, item.item_id)

        if item.past_sales is not None:
            if len(item.past_sales) != 35:
                raise ValueError(f"Invalid past_sales length {len(item.past_sales)}; expected 35 values.")
            sales = np.array(item.past_sales, dtype=np.float32)
        else:
            sales = self.fetch_history(item.store_id, item.item_id, lookback_days=35)

        x_tensor = torch.from_numpy(sales).unsqueeze(-1)
        x_hist = torch.zeros(35, 10, dtype=torch.float32)
        z_futr = torch.zeros(28, 10, dtype=torch.float32)

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
        s_cont = torch.tensor([float(np.mean(sales))], dtype=torch.float32)

        return {
            "x": x_tensor,
            "x_hist": x_hist,
            "z_futr": z_futr,
            "s_cat": s_cat,
            "s_cont": s_cont,
        }
