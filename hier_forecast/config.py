from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for data loading, window slicing, and rolling origins."""

    data_dir: str = "data/m5"
    sample_data_dir: str = "data/m5_sample"
    lookback: int = 35
    horizon: int = 28
    train_end_day: int = 1913
    val_window_days: int = 28
    multi_window_origins: list[int] = [1857, 1885, 1913]


class ModelConfig(BaseModel):
    """Configuration for TSMixerExt architecture."""

    hidden_size: int = 128
    num_blocks: int = 8
    dropout: float = 0.1
    use_revin: bool = False
    use_mean_scaling: bool = True
    probabilistic: bool = True


class TrainConfig(BaseModel):
    """Configuration for optimization, training cadence, and seeds."""

    batch_size: int = 512
    learning_rate: float = 1e-3
    epochs: int = 30
    patience: int = 10
    grad_clip: float = 10.0
    seeds: list[int] = [42, 43, 44]
    num_batches_per_epoch: int = 200


class ExperimentConfig(BaseModel):
    """Unified experiment configuration for reproducible training and evaluation runs."""

    experiment_id: str = "exp_default"
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
