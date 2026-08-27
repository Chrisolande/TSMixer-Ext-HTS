import os
import subprocess

import numpy as np
import torch
from loguru import logger


def seed_worker(worker_id: int):
    """Reseed numpy global RNG per DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def get_git_commit_hash() -> str:
    """Retrieve current Git commit SHA-1 hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def setup_wandb_auth() -> str | None:
    """Resolve W&B API key from environment variable or Kaggle Secrets, login if found."""
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        try:
            from kaggle_secrets import UserSecretsClient

            api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            pass

    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        try:
            import wandb

            wandb.login(key=api_key)
        except Exception as e:
            logger.warning(f"Failed to authenticate with W&B using resolved API key: {e}")
        return api_key

    return None
