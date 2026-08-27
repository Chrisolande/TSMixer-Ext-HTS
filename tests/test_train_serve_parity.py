import os
import tempfile

import numpy as np
import pytest
import torch

from hier_forecast.api.schemas.request import SeriesKey
from hier_forecast.api.store import InferenceStore
from hier_forecast.data_processing.bundle import save_preprocess_bundle
from hier_forecast.data_processing.dataset import M5Dataset
from hier_forecast.data_processing.pipeline import preprocess_m5


def test_train_serve_tensor_parity():
    """Assert InferenceStore.build_tensors matches M5Dataset slices within 1e-6."""
    sample_dir = "data/m5_sample"
    if not os.path.exists(sample_dir):
        pytest.skip(f"Sample data directory {sample_dir} not found.")

    with tempfile.TemporaryDirectory() as bundle_dir:
        # Preprocess sample dataset and save bundle
        data_dict = preprocess_m5(sample_dir, train_days=1886)
        save_preprocess_bundle(bundle_dir, data_dict)

        # Build offline PyTorch dataset (val split: origin day 1886, L=35, T=28)
        dataset = M5Dataset(
            data_dict,
            L=35,
            T=28,
            split="val",
            test_days=28,
            val_days=28,
            val_window_days=28,
            stride=1,
        )

        assert len(dataset) > 0

        # Initialize online InferenceStore with saved bundle
        store = InferenceStore(
            category_maps=data_dict["category_maps"],
            snapshot_dir=sample_dir,
            bundle_dir=bundle_dir,
        )

        # Compare first series from dataset with InferenceStore output
        # Day 1886 (d_1886) is the end of training; validation origin starts at d_1887 (2016-03-28)
        x_ds, hist_exog_ds, futr_exog_ds, s_cat_ds, s_cont_ds, y_ds = dataset[0]

        # In sample data, row 0 store_id and item_id
        import pandas as pd
        sales_raw = pd.read_csv(os.path.join(sample_dir, "sales_train_evaluation.csv"))
        first_row = sales_raw.iloc[0]
        store_id = first_row["store_id"]
        item_id = first_row["item_id"]

        cal_df = pd.read_csv(os.path.join(sample_dir, "calendar.csv"))
        # dataset[0] starts at d_start = total_days - test_days - val_window_days - L
        # Let's get the exact start date for day d_start + L
        d_start = dataset.window_indices[0, 1]
        as_of_d_num = d_start + 35 + 1  # 1-indexed day string d_NNN
        as_of_date_row = cal_df[cal_df["d"] == f"d_{as_of_d_num}"]
        as_of_date = as_of_date_row["date"].iloc[0]

        # Extract past sales directly for parity comparison
        d_cols = [c for c in sales_raw.columns if c.startswith("d_")]
        past_sales = first_row[d_cols[d_start : d_start + 35]].values.astype(float).tolist()

        item_key = SeriesKey(
            store_id=store_id,
            item_id=item_id,
            past_sales=past_sales,
        )

        tensors = store.build_tensors(item_key, as_of_date=as_of_date)

        # Check tensor shapes
        assert tensors["x"].shape == torch.Size([35, 1])
        assert tensors["x_hist"].shape == torch.Size([35, 10])
        assert tensors["z_futr"].shape == torch.Size([28, 10])
        assert tensors["s_cat"].shape == torch.Size([5])
        assert tensors["s_cont"].shape == torch.Size([1])

        # Parity assertions (< 1e-6 error)
        np.testing.assert_allclose(tensors["x"].numpy(), x_ds.numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(tensors["s_cat"].numpy(), s_cat_ds.numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(tensors["s_cont"].numpy(), s_cont_ds.numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(tensors["x_hist"].numpy(), hist_exog_ds.numpy(), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(tensors["z_futr"].numpy(), futr_exog_ds.numpy(), atol=1e-4, rtol=1e-4)
