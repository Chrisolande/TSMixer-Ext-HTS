import os
from loguru import logger


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
