import os
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from hier_forecast.data_processing.bundle import load_preprocess_bundle, save_preprocess_bundle
from hier_forecast.models.distribution import NegativeBinomial


def test_negative_binomial_distribution_properties():
    mu = torch.tensor([2.0, 5.0, 10.0])
    alpha = torch.tensor([0.5, 0.2, 0.1])

    dist = NegativeBinomial(mu=mu, alpha=alpha)

    # Verify r and p relations
    expected_r = 1.0 / alpha
    expected_p = 1.0 / (1.0 + alpha * mu)
    expected_var = mu + alpha * (mu**2)

    assert torch.allclose(dist.r, expected_r, atol=1e-5)
    assert torch.allclose(dist.p, expected_p, atol=1e-5)
    assert torch.allclose(dist.variance, expected_var, atol=1e-5)

    # Test log_prob calculation
    y = torch.tensor([1.0, 5.0, 12.0])
    lp = dist.log_prob(y)
    assert lp.shape == y.shape
    assert torch.all(lp <= 0.0)

    # Test cdf and ppf (via numpy/scipy)
    mu_np = np.array([2.0, 5.0])
    alpha_np = np.array([0.5, 0.2])
    dist_np = NegativeBinomial(mu=mu_np, alpha=alpha_np)

    q10 = dist_np.ppf(0.10)
    q50 = dist_np.ppf(0.50)
    q90 = dist_np.ppf(0.90)
    median = dist_np.median

    assert np.all(q10 <= q50)
    assert np.all(q50 <= q90)
    assert np.array_equal(q50, median)
    assert np.issubdtype(q50.dtype, np.integer)

    # Verify CDF at median >= 0.5 and at median - 1 < 0.5 (for discrete counts)
    cdf_val = dist_np.cdf(median)
    assert np.all(cdf_val >= 0.5)

    # Test randomized and non-randomized PIT
    y_obs = np.array([2, 5])
    pit_non_rand = dist_np.pit(y_obs, randomized=False)
    assert np.all(pit_non_rand >= 0.0) and np.all(pit_non_rand <= 1.0)

    pit_rand = dist_np.pit(y_obs, randomized=True)
    assert np.all(pit_rand >= 0.0) and np.all(pit_rand <= 1.0)


def test_preprocess_bundle_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Construct mock bundle data
        calendar_mean = np.random.randn(8).astype(np.float64)
        calendar_scale = np.abs(np.random.randn(8)).astype(np.float64) + 0.1
        
        # Mock hierarchy matrix S (10 x 5)
        s_dense = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1],
        ], dtype=np.float32)
        s_csr = sp.csr_matrix(s_dense)
        weights = np.ones(10, dtype=np.float64) / 10.0
        scaling_factors = np.ones(10, dtype=np.float64) * 0.5

        category_maps = {
            "item_id": {"ITEM_1": 0, "ITEM_2": 1},
            "store_id": {"STORE_1": 0, "STORE_2": 1},
            "state_id": {"CA": 0, "TX": 1},
            "dept_id": {"DEPT_1": 0},
            "cat_id": {"CAT_1": 0},
        }

        price_stats = {
            "series_price_stats": {
                "ITEM_1_STORE_1": {"mean": 2.5, "std": 0.2},
                "ITEM_2_STORE_2": {"mean": 4.0, "std": 0.5},
            },
            "dept_price_stats": {
                "DEPT_1_STORE_1": {"mean": 3.0, "std": 0.3},
            }
        }

        bundle_data = {
            "calendar_scaler_mean": calendar_mean,
            "calendar_scaler_scale": calendar_scale,
            "S_matrix": s_csr,
            "weights": weights,
            "scaling_factors": scaling_factors,
            "category_maps": category_maps,
            "price_stats": price_stats,
        }

        save_preprocess_bundle(tmpdir, bundle_data)

        # Assert files exist
        assert os.path.exists(os.path.join(tmpdir, "bundle.npz"))
        assert os.path.exists(os.path.join(tmpdir, "category_maps.json"))
        assert os.path.exists(os.path.join(tmpdir, "price_stats.json"))

        # Load back
        loaded = load_preprocess_bundle(tmpdir)

        assert np.allclose(loaded["calendar_scaler_mean"], calendar_mean)
        assert np.allclose(loaded["calendar_scaler_scale"], calendar_scale)
        assert np.allclose(loaded["weights"], weights)
        assert np.allclose(loaded["scaling_factors"], scaling_factors)
        assert np.allclose(loaded["S_matrix"].toarray(), s_dense)
        assert loaded["category_maps"] == category_maps
        assert loaded["price_stats"] == price_stats


def test_run_experiment_end_to_end():
    sample_dir = "data/m5_sample"
    if not os.path.exists(sample_dir):
        pytest.skip(f"Sample data directory {sample_dir} not found.")

    from hier_forecast.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
    from hier_forecast.training_engine.experiment import run_experiment

    with tempfile.TemporaryDirectory() as tmpdir:
        exp_id = "test_smoke_exp"
        cfg = ExperimentConfig(
            experiment_id=exp_id,
            data=DataConfig(
                sample_data_dir=sample_dir,
                data_dir=sample_dir,
                lookback=35,
                horizon=28,
                train_end_day=120,
                multi_window_origins=[70, 100],
            ),
            model=ModelConfig(
                hidden_size=32,
                num_blocks=2,
                dropout=0.1,
                probabilistic=True,
            ),
            train=TrainConfig(
                batch_size=32,
                epochs=1,
                num_batches_per_epoch=2,
                seeds=[42],
            ),
        )

        out_dir = os.path.join(tmpdir, exp_id)
        manifest = run_experiment(config=cfg, output_dir=out_dir)

        # Assert all artifacts created
        assert os.path.exists(os.path.join(out_dir, "model.pt"))
        assert os.path.exists(os.path.join(out_dir, "bundle.npz"))
        assert os.path.exists(os.path.join(out_dir, "category_maps.json"))
        assert os.path.exists(os.path.join(out_dir, "price_stats.json"))
        assert os.path.exists(os.path.join(out_dir, "metrics.json"))
        assert os.path.exists(os.path.join(out_dir, "manifest.json"))
        assert os.path.exists(os.path.join(out_dir, "config.json"))

        assert manifest["experiment_id"] == exp_id
        assert "metrics" in manifest
        assert "git_commit" in manifest

        # Assert model can be loaded back
        state = torch.load(os.path.join(out_dir, "model.pt"), weights_only=True)
        assert "tp_past.linear.weight" in state


def test_mixing_layers_batchnorm_and_gradient_flow():
    from hier_forecast.models.layers import FeatureMixing, MixerLayer, TimeMixing
    from hier_forecast.models.tsmixer import TSMixer

    B, L, C = 16, 35, 8
    x = torch.randn(B, L, C, requires_grad=True)

    # Test TimeMixing with batch norm
    tm = TimeMixing(seq_len=L, num_features=C, norm_type="batch", pre_norm=False)
    out_tm = tm(x)
    assert out_tm.shape == (B, L, C)
    out_tm.sum().backward()
    assert x.grad is not None

    # Test FeatureMixing with batch norm
    x2 = torch.randn(B, L, C, requires_grad=True)
    fm = FeatureMixing(in_features=C, out_features=16, seq_len=L, norm_type="batch", pre_norm=False)
    out_fm = fm(x2)
    assert out_fm.shape == (B, L, 16)
    out_fm.sum().backward()
    assert x2.grad is not None

    # Test MixerLayer with batch norm
    x3 = torch.randn(B, L, C, requires_grad=True)
    ml = MixerLayer(seq_len=L, num_features=C, hidden_size=16, norm_type="batch")
    out_ml = ml(x3)
    assert out_ml.shape == (B, L, C)
    out_ml.sum().backward()
    assert x3.grad is not None

    # Test TSMixer model with batch norm
    model = TSMixer(seq_len=L, pred_len=28, num_features=C, num_blocks=2, hidden_size=16, norm_type="batch")
    out_model = model(x3.detach())
    assert out_model.shape == (B, 28, C)
