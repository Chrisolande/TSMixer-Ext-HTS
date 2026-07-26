# %%writefile training.py
import os

import numpy as np
import torch
import torch.optim as optim
import wandb
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tsmixer_m5.data import M5Dataset, preprocess_m5
from tsmixer_m5.metrics import evaluate_wrmsse
from tsmixer_m5.modeling import NegativeBinomialLoss, TSMixerExt


def seed_worker(worker_id):
    """Reseed numpy global RNG per DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def train_and_validate(
    seeds=(42, 43, 44),
    lr=0.0005066474601108174,
    hidden_size=128,
    num_blocks=8,
    dropout=0.1,
    batch_size=1024,
    num_batches_per_epoch=200,
    wandb_project="tsmixer-m5",
    wandb_entity=None,
):
    """Train TSMixerExt model across multiple random seeds using optimal hyperparameters."""
    raw_data_dir = (
        "./data/m5"
        if not os.path.exists("/kaggle/input/competitions/m5-forecasting-accuracy/calendar.csv")
        else "/kaggle/input/competitions/m5-forecasting-accuracy"
    )

    data_dict = preprocess_m5(raw_data_dir, train_days=1886)

    L = 35
    T = 28

    BATCH_SIZE = batch_size
    NUM_BATCHES_PER_EPOCH = num_batches_per_epoch
    SAMPLES_PER_EPOCH = BATCH_SIZE * NUM_BATCHES_PER_EPOCH

    GRAD_CLIP = 10.0
    EPOCHS = 60  # ~12.3M samples; paper saw 7.68M → early stopping handles the rest
    PATIENCE = 10  # stop if NLL stagnates for 10 epochs (~2.5hr max wasted compute)

    EVAL_FREQ = 3  # matches HPO eval cadence for finer WRMSSE tracking

    train_dataset = M5Dataset(
        data_dict,
        L=L,
        T=T,
        split="train",
        stochastic=True,
        samples_per_epoch=SAMPLES_PER_EPOCH,
    )

    val_dataset = M5Dataset(data_dict, L=L, T=T, split="val", val_window_days=T)

    test_dataset = M5Dataset(data_dict, L=L, T=T, split="test")

    num_workers = 4 if os.cpu_count() and os.cpu_count() >= 4 else 0
    loader_kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, **loader_kwargs)

    val_loader = DataLoader(val_dataset, batch_size=6400, shuffle=False, **loader_kwargs)

    logger.info(
        "Train dataset: {train_size} | Val dataset: {val_size} | Test dataset: {test_size}",
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    test_wrmsse_scores = []

    hparams = dict(
        lr=lr,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        dropout=dropout,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        eval_freq=EVAL_FREQ,
    )

    from tsmixer_m5.utils import setup_wandb_auth
    setup_wandb_auth()



    seed_bar = tqdm(seeds, desc="Seeds", position=0)

    for seed in seed_bar:
        with wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"final_seed_{seed}",
            group="final_training",
            tags=["final", f"seed_{seed}"],
            config={**hparams, "seed": seed},
        ) as run:
            # Step metrics setup: plots all val_* against 'epoch'
            run.define_metric("epoch")
            run.define_metric("train_*", step_metric="epoch")
            run.define_metric("val_*", step_metric="epoch")

            torch.manual_seed(seed)
            np.random.seed(seed)

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

            # Track model weights & gradients to catch exploding/vanishing grads
            run.watch(model, log="all", log_freq=100)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
            loss_fn = NegativeBinomialLoss()

            best_val_loss = float("inf")
            best_val_wrmsse = float("inf")

            best_nll_state = None
            best_wrmsse_state = None

            best_nll_path = f"best_nll_model_seed_{seed}.pth"
            best_wrmsse_path = f"best_wrmsse_model_seed_{seed}.pth"

            epochs_no_improve = 0

            epoch_bar = tqdm(range(1, EPOCHS + 1), desc=f"Seed {seed}", position=1, leave=True)
            for epoch in epoch_bar:
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
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        mu, alpha = model(bx, bx_hist, bz_futr, s_cat=bs_cat, s_cont=bs_cont)
                        loss = loss_fn(mu, alpha, by_true)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                        optimizer.step()

                    train_loss += loss.item()
                    if batch_idx + 1 >= NUM_BATCHES_PER_EPOCH:
                        break

                avg_train_loss = train_loss / max(batch_idx + 1, 1)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for bx, bx_hist, bz_futr, bs_cat, bs_cont, by_true in val_loader:
                        bx = bx.to(device, non_blocking=True)
                        bx_hist = bx_hist.to(device, non_blocking=True)
                        bz_futr = bz_futr.to(device, non_blocking=True)
                        bs_cat = bs_cat.to(device, non_blocking=True)
                        bs_cont = bs_cont.to(device, non_blocking=True)
                        by_true = by_true.to(device, non_blocking=True)
                        if use_amp:
                            with torch.amp.autocast("cuda", dtype=amp_dtype):
                                mu, alpha = model(bx, bx_hist, bz_futr, s_cat=bs_cat, s_cont=bs_cont)
                                val_loss += loss_fn(mu, alpha, by_true).item()
                        else:
                            mu, alpha = model(bx, bx_hist, bz_futr, s_cat=bs_cat, s_cont=bs_cont)
                            val_loss += loss_fn(mu, alpha, by_true).item()

                avg_val_loss = val_loss / max(len(val_loader), 1)
                scheduler.step(avg_val_loss)
                current_lr = optimizer.param_groups[0]["lr"]

                log_payload = {
                    "epoch": epoch,
                    "train_nll": avg_train_loss,
                    "val_nll": avg_val_loss,
                    "lr": current_lr,
                }

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_nll_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += 1

                if epoch % EVAL_FREQ == 0 or epoch == 1:
                    val_wrmsse = evaluate_wrmsse(data_dict, model, device=device, split="val")
                    if val_wrmsse < best_val_wrmsse:
                        best_val_wrmsse = val_wrmsse
                        best_wrmsse_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    log_payload["val_wrmsse"] = val_wrmsse
                    log_payload["best_val_wrmsse"] = best_val_wrmsse

                run.log(log_payload)
                epoch_bar.set_postfix({"nll": f"{avg_val_loss:.3f}", "wrmsse": f"{best_val_wrmsse:.3f}"})

                if epochs_no_improve >= PATIENCE:
                    logger.info("Early stopping at epoch {e} (patience={p})", e=epoch, p=PATIENCE)
                    break

            best_state = best_wrmsse_state or best_nll_state
            if best_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                if best_nll_state is not None:
                    torch.save(best_nll_state, best_nll_path)
                if best_wrmsse_state is not None:
                    torch.save(best_wrmsse_state, best_wrmsse_path)

            seed_test_wrmsse = evaluate_wrmsse(data_dict, model, device=device, split="test")
            test_wrmsse_scores.append(seed_test_wrmsse)
            logger.success("Seed {seed} test WRMSSE: {score:.4f}", seed=seed, score=seed_test_wrmsse)

            run.summary["test_wrmsse"] = seed_test_wrmsse
            run.summary["best_val_wrmsse"] = best_val_wrmsse
            run.summary["epochs_trained"] = epoch
            run.summary["early_stopped"] = epochs_no_improve >= PATIENCE

            # 3. Native Model Logging — logs checkpoint & registers in W&B model catalog
            if best_wrmsse_state is not None:
                run.log_model(path=best_wrmsse_path, name=f"tsmixer_m5_seed_{seed}", aliases=["best", f"seed_{seed}"])

    mean_wrmsse = np.mean(test_wrmsse_scores)
    std_wrmsse = np.std(test_wrmsse_scores)

    logger.success(
        "Final ({num_seeds} seeds): {mean:.4f} ± {std:.4f}", num_seeds=len(seeds), mean=mean_wrmsse, std=std_wrmsse
    )

    return mean_wrmsse, std_wrmsse
