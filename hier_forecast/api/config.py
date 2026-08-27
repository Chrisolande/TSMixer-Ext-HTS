from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """FastAPI Inference Service Configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    title: str = "TSMixer M5 Inference API"
    version: str = "1.0.0"
    port: int = 8000
    host: str = "0.0.0.0"
    debug: bool = False

    wandb_api_key: str = ""
    wandb_model_artifact: str = "olandechris-/tsmixer-m5/tsmixer_m5_seed_43:v0"
    model_artifact_local_dir: str = "./artifact"

    device: str = "cpu"
    use_amp: bool = True
    max_batch_size: int = 1000
    default_horizon_days: int = 28
    lookback_window_days: int = 35

    data_snapshot_dir: str = "./data/m5_sample"
    bundle_dir: str = "./artifact"

    allowed_origins: list[str] = ["*"]


config = AppConfig()
