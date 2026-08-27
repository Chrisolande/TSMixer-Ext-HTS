import numpy as np
import torch

from hier_forecast.api.runner import ModelRunner
from hier_forecast.models.tsmixer_ext import TSMixerExt


def test_model_runner_exact_quantiles_properties():
    mu = np.array([2.5, 0.4, 12.0, 1.0])
    alpha = np.array([0.5, 1.2, 0.1, 0.3])

    q_dict = ModelRunner.quantiles(mu, alpha)

    assert "p10" in q_dict
    assert "p50" in q_dict
    assert "p90" in q_dict

    p10 = np.array(q_dict["p10"])
    p50 = np.array(q_dict["p50"])
    p90 = np.array(q_dict["p90"])

    # Discrete integers & monotonic ordering
    assert np.all(p10 <= p50)
    assert np.all(p50 <= p90)
    assert np.all(p10 >= 0)

    # p50 must be integer median (not floating mean)
    for v in p50:
        assert isinstance(v, (int, np.integer)) or v.is_integer()


def test_model_runner_predict():
    model = TSMixerExt(
        seq_len=35,
        pred_len=28,
        num_features=1,
        hist_exog_dim=10,
        futr_exog_dim=10,
        static_cont_dim=1,
        cat_cardinalities=[10, 30490, 7, 3, 3],
        cat_emb_dims=[8, 8, 8, 8, 16],
        num_blocks=2,
        hidden_size=32,
        probabilistic=True,
        use_mean_scaling=True,
    )
    runner = ModelRunner(
        model=model,
        category_maps={},
        device="cpu",
        use_amp=False,
    )

    B = 2
    x = torch.ones(B, 35, 1)
    x_hist = torch.zeros(B, 35, 10)
    z_futr = torch.zeros(B, 28, 10)
    s_cat = torch.zeros(B, 5, dtype=torch.long)
    s_cont = torch.ones(B, 1)

    mu, alpha = runner.predict(x, x_hist, z_futr, s_cat, s_cont)
    assert mu.shape == (B, 28)
    assert alpha.shape == (B, 28)
    assert torch.all(mu > 0)
    assert torch.all(alpha > 0)


def test_model_runner_missing_artifacts_raises():
    import pytest

    with pytest.raises((RuntimeError, FileNotFoundError)):
        ModelRunner.from_wandb(
            wandb_artifact="nonexistent/fake/artifact:v0",
            local_dir="/tmp/nonexistent_dir_for_test",
        )
