import multiprocessing as mp
import os
import sqlite3
import time

import optuna
import torch
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from hier_forecast.data_processing.dataset import M5Dataset
from hier_forecast.data_processing.pipeline import preprocess_m5
from hier_forecast.evaluation.hierarchical import evaluate_wrmsse
from hier_forecast.models.loss import NegativeBinomialLoss
from hier_forecast.models.tsmixer_ext import TSMixerExt


def objective(trial, data_dict, device):
    """Optuna objective function for TSMixer hyperparameter optimization on assigned GPU."""
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    num_blocks = trial.suggest_categorical("num_blocks", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 640, 1024])
    num_batches_per_epoch = trial.suggest_categorical("num_batches_per_epoch", [50, 100, 200])

    L, T = 35, 28
    samples_per_epoch = batch_size * num_batches_per_epoch

    train_dataset = M5Dataset(data_dict, L=L, T=T, split="train", stochastic=True, samples_per_epoch=samples_per_epoch)
    val_dataset = M5Dataset(data_dict, L=L, T=T, split="val", val_window_days=T, stochastic=False)

    loader_kwargs = {"num_workers": 4, "pin_memory": torch.cuda.is_available()}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **loader_kwargs)
    _ = DataLoader(val_dataset, batch_size=6400, shuffle=False, **loader_kwargs)

    model = TSMixerExt(
        seq_len=L,
        pred_len=T,
        num_features=1,
        hist_exog_dim=10,
        futr_exog_dim=10,
        static_cont_dim=1,
        cat_cardinalities=data_dict["cat_cardinalities"],
        cat_emb_dims=[8, 8, 8, 8, 16],
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        dropout=dropout,
        norm_type="layer",
        pre_norm=False,
        probabilistic=True,
        use_mean_scaling=True,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = NegativeBinomialLoss()

    use_amp = torch.cuda.is_available() and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    best_val_wrmsse = float("inf")

    for epoch in range(1, 31):
        model.train()
        train_loss = 0.0

        for batch_idx, (bx, bx_hist, bz_futr, bs_cat, bs_cont, by_true) in enumerate(train_loader):
            bx = bx.to(device, non_blocking=True)
            bx_hist = bx_hist.to(device, non_blocking=True)
            bz_futr = bz_futr.to(device, non_blocking=True)
            bs_cat = bs_cat.to(device, non_blocking=True)
            bs_cont = bs_cont.to(device, non_blocking=True)
            by_true = by_true.to(device, non_blocking=True)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    mu, alpha = model(bx, bx_hist, bz_futr, s_cat=bs_cat, s_cont=bs_cont)
                    loss = loss_fn(mu, alpha, by_true)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                mu, alpha = model(bx, bx_hist, bz_futr, s_cat=bs_cat, s_cont=bs_cont)
                loss = loss_fn(mu, alpha, by_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            train_loss += loss.item()
            if batch_idx + 1 >= num_batches_per_epoch:
                break

        avg_train_loss = train_loss / max(batch_idx + 1, 1)
        scheduler.step(avg_train_loss)

        if epoch % 3 == 0 or epoch == 1:
            val_wrmsse = evaluate_wrmsse(data_dict, model, device=device, split="val")
            if val_wrmsse < best_val_wrmsse:
                best_val_wrmsse = val_wrmsse

            trial.report(val_wrmsse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_wrmsse


def worker(gpu_id, n_trials, study_name, storage_url, data_dict):
    """Entry point for worker process pinned to single GPU."""
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    def objective_with_attr(trial):
        trial.set_user_attr("gpu_id", gpu_id)
        return objective(trial, data_dict, device)

    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": 30.0}},
    )
    study = optuna.load_study(study_name=study_name, storage=storage)

    from hier_forecast.training_engine.utils import setup_wandb_auth
    setup_wandb_auth()



    try:
        from optuna_integration.wandb import WeightsAndBiasesCallback
    except ImportError:
        from optuna.integration.wandb import WeightsAndBiasesCallback


    wandb_cb = WeightsAndBiasesCallback(
        metric_name="wrmsse",
        wandb_kwargs={
            "project": "tsmixer-m5",
            "group": "optuna_sweep",
            "tags": ["hparam_search", f"gpu_{gpu_id}"],
        },
    )
    callbacks = [wandb_cb]



    study.optimize(objective_with_attr, n_trials=n_trials, callbacks=callbacks)


def run_optuna_study(n_trials=100, data_dir="./data/m5", storage_url="sqlite:///m5_optuna.db"):
    """Run parallel Optuna study using TPE sampler with MedianPruner."""
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise RuntimeError(f"Expected 2 GPUs, found {num_gpus}. Use single-process path instead.")

    if n_trials % num_gpus != 0:
        logger.warning(
            "n_trials={n} is not evenly divisible by num_gpus={g}; "
            "{dropped} trial(s) will not run. Consider a multiple of {g}.",
            n=n_trials,
            g=num_gpus,
            dropped=n_trials % num_gpus,
        )

    raw_data_dir = (
        data_dir
        if not os.path.exists("/kaggle/input/competitions/m5-forecasting-accuracy/calendar.csv")
        else "/kaggle/input/competitions/m5-forecasting-accuracy"
    )
    data_dict = preprocess_m5(raw_data_dir, train_days=1886)

    study_name = "m5_tsmixer_study"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if storage_url.startswith("sqlite:///"):
        db_path = storage_url.replace("sqlite:///", "", 1)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()

    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": 30.0}},
    )
    optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
        load_if_exists=True,
    )

    trials_per_gpu = n_trials // num_gpus
    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=worker,
            args=(gpu_id, trials_per_gpu, study_name, storage_url, data_dict),
        )
        p.start()
        procs.append(p)

    pbars = {
        0: tqdm(total=trials_per_gpu, desc="GPU 0 Trials", position=0, leave=True),
        1: tqdm(total=trials_per_gpu, desc="GPU 1 Trials", position=1, leave=True),
    }
    completed_counts = {0: 0, 1: 0}

    poll_study = optuna.load_study(study_name=study_name, storage=storage)

    while any(p.is_alive() for p in procs):
        time.sleep(2)
        try:
            counts = {0: 0, 1: 0}
            for trial in poll_study.get_trials(deepcopy=False):
                if trial.state.is_finished():
                    gid = trial.user_attrs.get("gpu_id")
                    if gid is not None and int(gid) in counts:
                        counts[int(gid)] += 1

            for gid in (0, 1):
                delta = counts[gid] - completed_counts[gid]
                if delta > 0:
                    pbars[gid].update(delta)
                    completed_counts[gid] = counts[gid]
                    best_value = poll_study.best_value if poll_study.best_trial else None
                    if best_value is not None:
                        pbars[gid].set_postfix({"best_wrmsse": f"{best_value:.4f}"})
                    pbars[gid].refresh()
        except Exception:
            logger.debug("Progress poll failed, will retry next tick", exc_info=True)

    for p in procs:
        p.join()

    for pbar in pbars.values():
        pbar.refresh()
        pbar.close()

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    logger.success("Best trial WRMSSE: {score:.6f}", score=study.best_value)
    return study.best_params
