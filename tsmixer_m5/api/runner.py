import json
import os

import numpy as np
import torch
from loguru import logger

from tsmixer_m5.modeling import TSMixerExt

DEFAULT_CATEGORY_MAPS: dict[str, dict[str, int]] = {
    "store_id": {f"CA_{i}": i - 1 for i in range(1, 11)},
    "item_id": {f"HOBBIES_1_{i:03d}": i - 1 for i in range(1, 30491)},
    "dept_id": {"HOBBIES_1": 0, "HOBBIES_2": 1},
    "cat_id": {"HOBBIES": 0, "FOODS": 1, "HOUSEHOLD": 2},
    "state_id": {"CA": 0, "TX": 1, "WI": 2},
}


class ModelRunner:
    """Manages TSMixerExt model artifact loading from W&B and AMP inference execution."""

    def __init__(
        self,
        model: TSMixerExt,
        category_maps: dict,
        device: str = "cpu",
        use_amp: bool = True,
    ):
        self.model = model
        self.category_maps = category_maps
        self.device = device
        self.use_amp = use_amp

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_wandb(
        cls,
        wandb_artifact: str = "olandechris-/tsmixer-m5/tsmixer_m5_seed_43:v0",
        wandb_api_key: str | None = None,
        local_dir: str = "./artifact",
        device: str = "cpu",
        use_amp: bool = True,
    ) -> "ModelRunner":
        """Download or load model checkpoint and encodings from W&B Model Registry."""
        weights_path = None

        if wandb_api_key or os.environ.get("WANDB_API_KEY"):
            try:
                import wandb

                if wandb_api_key:
                    wandb.login(key=wandb_api_key)
                api = wandb.Api()
                logger.info("Fetching model artifact from W&B Model Catalog: {a}", a=wandb_artifact)
                artifact = api.artifact(wandb_artifact)
                download_path = artifact.download(root=local_dir)
                weights_path = os.path.join(download_path, "best_wrmsse_seed_42.pth")
                if not os.path.exists(weights_path):
                    weights_path = os.path.join(download_path, "model.pth")
            except Exception as e:
                logger.warning("Failed to fetch W&B artifact ({e}); attempting local fallback.", e=e)

        if not weights_path or not os.path.exists(weights_path):
            local_weights = os.path.join(local_dir, "best_wrmsse_seed_42.pth")
            if os.path.exists(local_weights):
                weights_path = local_weights
            elif os.path.exists("best_wrmsse_seed_42.pth"):
                weights_path = "best_wrmsse_seed_42.pth"

        if not weights_path or not os.path.exists(weights_path):
            logger.error("No valid checkpoint found locally or via W&B.")
            raise RuntimeError(f"Model checkpoint not found for {wandb_artifact} or local paths.")

        logger.info("Loading PyTorch model weights from: {p}", p=weights_path)
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)

        hidden_size = 128
        cat_cardinalities = [10, 30490, 7, 3, 3]
        cat_emb_dims = [8, 8, 8, 8, 16]

        if "static_embedder.proj.weight" in state_dict:
            hidden_size = state_dict["static_embedder.proj.weight"].shape[0]

        for i in range(5):
            key = f"static_embedder.embeddings.{i}.weight"
            if key in state_dict:
                cat_cardinalities[i] = state_dict[key].shape[0]
                cat_emb_dims[i] = state_dict[key].shape[1]

        model = TSMixerExt(
            seq_len=35,
            pred_len=28,
            num_features=1,
            hist_exog_dim=10,
            futr_exog_dim=10,
            static_cont_dim=1,
            cat_cardinalities=cat_cardinalities,
            cat_emb_dims=cat_emb_dims,
            num_blocks=8,
            hidden_size=hidden_size,
            dropout=0.1,
            probabilistic=True,
            use_mean_scaling=True,
        )
        model.load_state_dict(state_dict, strict=False)

        category_maps = DEFAULT_CATEGORY_MAPS
        cat_map_path = os.path.join(local_dir, "category_maps.json")
        if os.path.exists(cat_map_path):
            with open(cat_map_path, encoding="utf-8") as f:
                category_maps = json.load(f)

        return cls(
            model=model,
            category_maps=category_maps,
            device=device,
            use_amp=use_amp,
        )

    def predict(
        self,
        x: torch.Tensor,
        x_hist: torch.Tensor,
        z_futr: torch.Tensor,
        s_cat: torch.Tensor,
        s_cont: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute forward pass with torch.inference_mode and optional AMP."""
        x = x.to(self.device, non_blocking=True)
        x_hist = x_hist.to(self.device, non_blocking=True)
        z_futr = z_futr.to(self.device, non_blocking=True)
        s_cat = s_cat.to(self.device, non_blocking=True)
        s_cont = s_cont.to(self.device, non_blocking=True)

        with torch.inference_mode():
            if self.use_amp and self.device.startswith("cuda"):
                with torch.amp.autocast("cuda"):
                    mu, alpha = self.model(x, x_hist, z_futr, s_cat=s_cat, s_cont=s_cont)
            elif self.use_amp and self.device == "cpu":
                with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                    mu, alpha = self.model(x, x_hist, z_futr, s_cat=s_cat, s_cont=s_cont)
            else:
                mu, alpha = self.model(x, x_hist, z_futr, s_cat=s_cat, s_cont=s_cont)

        mu = mu.float()
        alpha = alpha.float()
        if mu.ndim == 3 and mu.shape[-1] == 1:
            mu = mu.squeeze(-1)
        if alpha.ndim == 3 and alpha.shape[-1] == 1:
            alpha = alpha.squeeze(-1)

        return mu, alpha

    @staticmethod
    def quantiles(mu: np.ndarray, alpha: np.ndarray) -> dict[str, list[float]]:
        """Compute p10, p50, p90 quantiles from Negative Binomial parameters mu and alpha."""
        from scipy.stats import norm

        var = mu + (alpha * (mu**2))
        std = np.sqrt(np.maximum(var, 1e-6))

        z10 = float(norm.ppf(0.10))
        z90 = float(norm.ppf(0.90))

        p10 = np.maximum(0.0, mu + z10 * std)
        p50 = np.maximum(0.0, mu)
        p90 = np.maximum(0.0, mu + z90 * std)

        return {
            "p10": np.round(p10, 4).tolist(),
            "p50": np.round(p50, 4).tolist(),
            "p90": np.round(p90, 4).tolist(),
        }
