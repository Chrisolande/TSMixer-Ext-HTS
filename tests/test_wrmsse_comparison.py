import os

import numpy as np
import pandas as pd
import pytest

from hier_forecast.data_processing.features import build_calendar
from hier_forecast.data_processing.pipeline import preprocess_m5
from hier_forecast.evaluation.hierarchical import evaluate_multi_window_wrmsse
from hier_forecast.models.tsmixer_ext import TSMixerExt


def test_leakage_free_preprocessing_invariants():
    """Assert changing data after train_days does not alter fitted preprocessing parameters."""
    sample_dir = "data/m5_sample"
    if not os.path.exists(sample_dir):
        pytest.skip(f"Sample data directory {sample_dir} not found.")

    cal_path = os.path.join(sample_dir, "calendar.csv")
    cal_df1 = pd.read_csv(cal_path)
    cal_df2 = cal_df1.copy()

    # Alter future rows (days > 100)
    cal_df2.loc[120:, "wday"] = 999.0

    _, mean1, scale1 = build_calendar(cal_df1, train_days=100, return_scaler=True)
    _, mean2, scale2 = build_calendar(cal_df2, train_days=100, return_scaler=True)

    np.testing.assert_allclose(mean1, mean2, atol=1e-6)
    np.testing.assert_allclose(scale1, scale2, atol=1e-6)


def test_multi_window_validation_and_window_algebra():
    """Verify rolling-origin multi-window evaluation across multiple origins."""
    sample_dir = "data/m5_sample"
    if not os.path.exists(sample_dir):
        pytest.skip(f"Sample data directory {sample_dir} not found.")

    data_dict = preprocess_m5(sample_dir, train_days=150)

    # Initialize a small test model matching dataset cardinalities
    model = TSMixerExt(
        seq_len=35,
        pred_len=28,
        num_features=1,
        hist_exog_dim=10,
        futr_exog_dim=10,
        static_cont_dim=1,
        cat_cardinalities=data_dict["cat_cardinalities"],
        num_blocks=2,
        hidden_size=32,
        probabilistic=True,
        use_mean_scaling=True,
    )

    # Test origins within available days (e.g. 80, 110, 140)
    origins = [80, 110, 140]
    results = evaluate_multi_window_wrmsse(
        data_dict=data_dict,
        model=model,
        origins=origins,
        L=35,
        T=28,
        device="cpu",
    )

    assert "origins" in results
    assert "wrmsse_mean" in results
    assert "wrmsse_std" in results
    assert "crps_mean" in results
    assert "coverage_80_mean" in results

    assert len(results["origins"]) == 3
    assert results["wrmsse_mean"] >= 0.0
    assert results["crps_mean"] >= 0.0
    assert 0.0 <= results["coverage_80_mean"] <= 1.0


def test_invalid_origin_window_algebra_raises():
    """Assert invalid origin (< lookback) raises assertion error."""
    data_dict = {
        "sales_matrix": np.ones((5, 100), dtype=np.float32),
        "static_cats": np.zeros((5, 5), dtype=np.int64),
        "static_cont": np.ones((5, 1), dtype=np.float32),
        "exog_matrix": np.zeros((5, 100, 10), dtype=np.float32),
        "S_matrix": None,
        "weights": np.ones(5),
        "scaling_factors": np.ones(5),
    }
    model = None

    # Origin 20 is smaller than lookback 35
    with pytest.raises(AssertionError, match="too small"):
        evaluate_multi_window_wrmsse(data_dict, model, origins=[20], L=35, T=28)
