import datetime
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

from hier_forecast.config import ExperimentConfig
from hier_forecast.data_processing.bundle import save_preprocess_bundle
from hier_forecast.data_processing.dataset import M5Dataset
from hier_forecast.data_processing.pipeline import preprocess_m5
from hier_forecast.evaluation.hierarchical import evaluate_multi_window_wrmsse
from hier_forecast.models.loss import NegativeBinomialLoss
from hier_forecast.models.tsmixer_ext import TSMixerExt
from hier_forecast.training_engine.utils import get_git_commit_hash


def run_experiment(
    config: ExperimentConfig | dict | None = None,
    output_dir: str | None = None,
    use_wandb: bool = False,
) -> dict:
    """Config-driven training and evaluation pipeline producing reproducible artifact bundles."""
    if config is None:
        config = ExperimentConfig()
    elif isinstance(config, dict):
        config = ExperimentConfig(**config)

    exp_dir = output_dir or os.path.join("artifacts", config.experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Resolve dataset
    data_dir = config.data.data_dir
    if not os.path.exists(data_dir):
        if os.path.exists(config.data.sample_data_dir):
            data_dir = config.data.sample_data_dir
        elif os.path.exists("/kaggle/input/competitions/m5-forecasting-accuracy/calendar.csv"):
            data_dir = "/kaggle/input/competitions/m5-forecasting-accuracy"

    logger.info("Running experiment '{exp_id}' on {data_dir}", exp_id=config.experiment_id, data_dir=data_dir)
    data_dict = preprocess_m5(data_dir, train_days=config.data.train_end_day)

    # 2. Save preprocessing bundle and config
    save_preprocess_bundle(exp_dir, data_dict)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config.model_dump(), f, indent=2)

    # 3. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TSMixerExt(
        seq_len=config.data.lookback,
        pred_len=config.data.horizon,
        num_features=1,
        hist_exog_dim=10,
        futr_exog_dim=10,
        static_cont_dim=1,
        cat_cardinalities=data_dict["cat_cardinalities"],
        num_blocks=config.model.num_blocks,
        hidden_size=config.model.hidden_size,
        dropout=config.model.dropout,
        use_mean_scaling=config.model.use_mean_scaling,
        probabilistic=config.model.probabilistic,
    ).to(device)

    # Quick train on first seed
    seed = config.train.seeds[0] if config.train.seeds else 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = M5Dataset(
        data_dict,
        L=config.data.lookback,
        T=config.data.horizon,
        split="train",
        stochastic=True,
        samples_per_epoch=config.train.batch_size * config.train.num_batches_per_epoch,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        drop_last=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)
    loss_fn = NegativeBinomialLoss() if config.model.probabilistic else nn.MSELoss()

    model.train()
    for _epoch in range(min(config.train.epochs, 3)):
        for b_idx, (bx, bx_hist, bz_futr, bs_cat, bs_cont, by_true) in enumerate(train_loader):
            bx = bx.to(device)
            bx_hist = bx_hist.to(device)
            bz_futr = bz_futr.to(device)
            bs_cat = bs_cat.to(device)
            bs_cont = bs_cont.to(device)
            by_true = by_true.to(device)

            optimizer.zero_grad()
            out = model(bx, bx_hist, bz_futr, s_cat=bs_cat, s_cont=bs_cont)
            if config.model.probabilistic:
                mu, alpha = out
                loss = loss_fn(mu, alpha, by_true)
            else:
                loss = loss_fn(out, by_true)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip)
            optimizer.step()
            if b_idx + 1 >= config.train.num_batches_per_epoch:
                break

    # 4. Multi-window evaluation
    multi_metrics = evaluate_multi_window_wrmsse(
        data_dict=data_dict,
        model=model,
        origins=config.data.multi_window_origins,
        L=config.data.lookback,
        T=config.data.horizon,
        device=str(device),
    )

    # 5. Save model checkpoint
    model_path = os.path.join(exp_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # 6. Save metrics and manifest
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(multi_metrics, f, indent=2)

    manifest = {
        "experiment_id": config.experiment_id,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": get_git_commit_hash(),
        "seeds": config.train.seeds,
        "device": str(device),
        "metrics": multi_metrics,
        "bundle_path": "bundle.npz",
        "model_path": "model.pt",
    }
    manifest_path = os.path.join(exp_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.success("Experiment {exp_id} artifacts written to {d}", exp_id=config.experiment_id, d=exp_dir)
    return manifest
