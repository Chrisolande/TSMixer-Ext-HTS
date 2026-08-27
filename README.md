# Hierarchical & Probabilistic Time Series Forecasting

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference%20Service-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage%20Slim-2496ED.svg)](https://www.docker.com/)
[![W&B Tracked](https://img.shields.io/badge/Weights_%26_Biases-Tracked-yellow.svg)](https://wandb.ai/olandechris-/tsmixer-m5/reports/TSMixer-M5-Test-WRMSSE-Optimization-Report--VmlldzoxNzY5MzQ3NA?accessToken=n6bpmf6yiadweiscxlo7ggks6p47i209dp6l1h59bmqr2gqxl2qdxufj9arh4uhw)
[![Optuna](https://img.shields.io/badge/Optuna-HPO%20Enabled-blue)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What this does

Walmart sells thousands of products across many stores. This system predicts how many units of each product will sell on each of the next 28 days, and it gives a range of likely outcomes instead of a single guess.

Two things make that harder than it sounds. First, most products barely sell at all on most days: a lot of the data is zeros, with the occasional spike. Second, individual product forecasts need to add up correctly at the store level, the category level, and the company level. A forecast that looks fine for one item but breaks the totals when you roll it up isn't actually useful.

This project handles both problems, then wraps the trained model in a live web service so another piece of software can ask "how many of this item will sell next week?" and get an answer back immediately.

---

## How it works

The dataset covers 30,490 individual product-store combinations, organized into a hierarchy that totals 42,840 series once you include every store, category, and company-wide rollup. A few design choices make the forecasts useful in practice rather than just accurate on paper:

- **It predicts a range, not just a number.** Sales counts are modeled with a Negative Binomial distribution, which is built for exactly this kind of sparse, spiky count data. Instead of a single prediction, the model outputs a full distribution, so you get a low estimate, a likely estimate, and a high estimate for each day.
- **The hierarchy stays consistent.** Aggregating 42,840 series to check that store-level and category-level totals line up would be slow if done naively. This uses sparse matrix multiplication on the GPU to keep that aggregation under 100 milliseconds.
- **Validation reflects how the model will actually be used.** Forecasts are tested against multiple historical cutoff points, with scalers and preprocessing fit only on data that would have been available at the time. That rules out the model accidentally learning from the future.
- **What gets trained is what gets served.** Preprocessing steps, category lookups, and price normalization are packaged into a self-contained bundle so the live API applies the exact same transformations as training, to within a tiny numerical tolerance.
- **It's a real service, not just a notebook.** A FastAPI app serves forecasts in real time, and the whole thing runs in a slim Docker container.
- **Hyperparameters are tuned automatically** using Optuna, and every experiment is logged to Weights & Biases so runs are comparable and reproducible.

---

## Mathematical formulations

<details>
<summary><b>Click to expand: metric derivations and model math</b></summary>

<br>

### 1. Reversible Instance Normalization (RevIN)

To keep the model stable across sliding historical windows with different scales:

$$\hat{x} = \gamma \odot \left( \frac{x - \mu_x}{\sigma_x + \epsilon} \right) + \beta$$

where $\mu_x, \sigma_x$ are the mean and standard deviation over the lookback window, and $\gamma, \beta$ are learnable parameters.

### 2. Negative Binomial likelihood

For overdispersed count data $y \sim \text{NB}(\mu, \alpha)$ with mean $\mu$ and dispersion $\alpha$:

$$r = \frac{1}{\alpha}, \quad p = \frac{1}{1 + \alpha \mu}, \quad \text{Var}(Y) = \mu + \alpha \mu^2$$

$$\mathcal{L}_{\text{NLL}} = -\left[ \log \Gamma(y + r) - \log \Gamma(y + 1) - \log \Gamma(r) + r \log(p + \epsilon) + y \log(1 - p + \epsilon) \right]$$

### 3. Discrete quantiles and CRPS

Quantiles come from the discrete Negative Binomial percent point function:

$$q_\tau = \min \{ k \in \mathbb{N}_0 : F_{\text{NB}}(k; \mu, \alpha) \ge \tau \}$$

and the exact discrete Continuous Ranked Probability Score is:

$$\text{CRPS}(F, y) = \sum_{k=0}^{\infty} \left( F_{\text{NB}}(k; \mu, \alpha) - \mathbb{I}(y \le k) \right)^2$$

### 4. Pinball loss and Weighted Interval Score (WIS)

For quantile level $\tau \in (0, 1)$:

$$\mathcal{L}_{\tau}(y, \hat{y}_\tau) = \max \left( \tau (y - \hat{y}_\tau), (\tau - 1)(y - \hat{y}_\tau) \right)$$

For a central $(1 - \alpha)$ prediction interval $[L, U]$:

$$\text{IS}_\alpha(y, L, U) = (U - L) + \frac{2}{\alpha}(L - y)\mathbb{I}(y < L) + \frac{2}{\alpha}(y - U)\mathbb{I}(y > U)$$

### 5. Weighted Root Mean Squared Scaled Error (WRMSSE)

The official M5 evaluation metric, computed across all $K = 42{,}840$ hierarchy nodes:

$$\text{WRMSSE} = \sum_{i=1}^{K} w_i \sqrt{ \frac{\frac{1}{h} \sum_{t=n+1}^{n+h} (\hat{y}_{i,t} - y_{i,t})^2}{c_i} }$$

where $c_i$ is a scale factor from historical sales:

$$c_i = \frac{1}{n_i - 1} \sum_{t=2}^{n_i} (y_{i,t} - y_{i,t-1})^2$$

and $w_i$ is the dollar-revenue weight assigned to node $i$.

</details>

---

## Project structure

```
TSMixer-Ext-HTS/
├── hier_forecast/
│   ├── models/                  # Architectures, layers, and distributions
│   │   ├── distribution.py      # NegativeBinomial wrapper, exact PPF quantiles, PIT
│   │   ├── loss.py              # NegativeBinomialLoss
│   │   ├── layers.py            # TimeMixing, FeatureMixing, StaticEmbeddingBlock, MeanScaling, RevIN
│   │   ├── tsmixer.py           # Base mixer architecture
│   │   ├── tsmixer_ext.py       # Extended architecture with static/exogenous conditioning
│   │   └── __init__.py
│   ├── data_processing/         # Preprocessing, feature engineering, PyTorch dataset
│   │   ├── constants.py         # Categorical and event definitions
│   │   ├── features.py          # Calendar, price z-scores, hierarchy S-matrix, weights
│   │   ├── dataset.py           # M5Dataset (sliding-window & stochastic sampling)
│   │   ├── bundle.py            # Artifact bundle serialization (bundle.npz, JSON lookups)
│   │   ├── pipeline.py          # preprocess_m5 end-to-end pipeline
│   │   └── __init__.py
│   ├── evaluation/              # Hierarchical, probabilistic, and calibration metrics
│   │   ├── hierarchical.py      # evaluate_wrmsse, evaluate_multi_window_wrmsse
│   │   ├── probabilistic.py     # pinball_loss, WIS, empirical_coverage, sharpness, discrete CRPS
│   │   ├── calibration.py       # PIT uniformity and histogram test
│   │   └── __init__.py
│   ├── training_engine/         # Multi-seed training, config runner, experiment tracking
│   │   ├── utils.py             # seed_worker, git commit helper
│   │   ├── trainer.py           # train_and_validate with W&B logging & checkpoints
│   │   ├── experiment.py        # Config-driven experiment execution & manifest packaging
│   │   └── __init__.py
│   ├── api/                     # FastAPI inference microservice
│   │   ├── app.py               # Application factory and lifespan
│   │   ├── config.py            # AppConfig and environment settings
│   │   ├── runner.py            # ModelRunner for batched inference
│   │   ├── store.py             # InferenceStore for dynamic exogenous feature injection
│   │   ├── dependencies.py      # Dependency injection
│   │   ├── schemas/             # Pydantic request and response models
│   │   └── v1/                  # API v1 routes (forecast, health)
│   ├── config.py                # DataConfig, ModelConfig, TrainConfig, ExperimentConfig
│   └── hparam_search.py         # Optuna hyperparameter study
├── data/
│   ├── m5/                      # Full M5 CSV files
│   └── m5_sample/               # Sample M5 subset for unit tests and local API testing
├── tests/                       # pytest suite (17 tests, 100% pass rate)
├── Dockerfile                   # Multi-stage slim container definition
├── pyproject.toml               # Project metadata, dependencies, and tooling config
└── README.md
```

---

## Getting started

### 1. Installation

Set up the Python 3.13 environment with `uv` (recommended) or `pip`:

```bash
git clone https://github.com/Chrisolande/TSMixer-Ext-HTS.git
cd TSMixer-Ext-HTS

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies and dev tools
uv pip install -e ".[dev]"
```

### 2. Dataset setup

Place the official M5 dataset CSV files in `./data/m5/` (or use the sample in `./data/m5_sample/` for testing):

```
data/
├── m5/
│   ├── calendar.csv
│   ├── sell_prices.csv
│   └── sales_train_evaluation.csv    # or sales_train_validation.csv
└── m5_sample/
    ├── calendar.csv
    ├── sell_prices.csv
    └── sales_train_evaluation.csv
```

### 3. Environment variables

```bash
export WANDB_API_KEY="your_wandb_api_key_here"

# Optional, defaults shown
export WANDB_MODEL_ARTIFACT="olandechris-/tsmixer-m5/tsmixer_m5_seed_43:v0"
export MODEL_ARTIFACT_LOCAL_DIR="./artifact"
export DATA_SNAPSHOT_DIR="./data/m5_sample"
export DEVICE="cpu"
```

---

## Workflows & usage

### 1. Preprocessing & artifact bundle export

Fit scalers, category maps, and price lookups on training data, then export them alongside the hierarchy matrices into a self-contained bundle:

```python
from hier_forecast.data_processing.bundle import save_preprocess_bundle
from hier_forecast.data_processing.pipeline import preprocess_m5

# Fit and export preprocessing bundle
data_dict = preprocess_m5("data/m5_sample", train_days=150)
save_preprocess_bundle("artifact/", data_dict)
```

### 2. Multi-window rolling-origin validation

Evaluate WRMSSE, discrete CRPS, and empirical coverage across multiple forecast origins (e.g. days 70, 100, 130):

```python
from hier_forecast.data_processing.pipeline import preprocess_m5
from hier_forecast.evaluation.hierarchical import evaluate_multi_window_wrmsse
from hier_forecast.models.tsmixer_ext import TSMixerExt

data_dict = preprocess_m5("data/m5_sample", train_days=150)
model = TSMixerExt(
    seq_len=35,
    pred_len=28,
    num_features=1,
    hist_exog_dim=10,
    futr_exog_dim=10,
    static_cont_dim=1,
    cat_cardinalities=data_dict["cat_cardinalities"],
    num_blocks=2,
    hidden_size=32,
    probabilistic=True,
)

metrics = evaluate_multi_window_wrmsse(data_dict, model, origins=[70, 100, 130])
print(metrics)
```

### 3. Config-driven experiment execution

```python
from hier_forecast.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from hier_forecast.training_engine.experiment import run_experiment

cfg = ExperimentConfig(
    experiment_id="exp_tsmixer_nb_prod",
    data=DataConfig(lookback=35, horizon=28, train_end_day=150),
    model=ModelConfig(hidden_size=128, num_blocks=8, probabilistic=True),
    train=TrainConfig(batch_size=1024, learning_rate=5e-4, seeds=[42, 43, 44]),
)

manifest = run_experiment(config=cfg, output_dir="artifacts/exp_tsmixer_nb_prod")
print("Experiment Manifest:", manifest)
```

### 4. Hyperparameter optimization

Automated TPE sampling with median pruning, via Optuna:

```bash
python -m hier_forecast.hparam_search
```

---

## FastAPI inference service

### Running the API server

```bash
uvicorn hier_forecast.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs:
- **Scalar Docs**: `http://localhost:8000/scalar`
- **Swagger UI**: `http://localhost:8000/docs`

### Health & readiness probes

```bash
curl -s http://localhost:8000/health
# {"status": "healthy"}

curl -s http://localhost:8000/readyz
# {"status": "ready", "device": "cpu"}
```

### 1. Automatic on-disk snapshot lookup (recommended)

Forecast 28 days forward for given store and item IDs without providing past history manually (the service extracts the historical sales window and dynamic covariates automatically):

```bash
curl -s -X POST http://localhost:8000/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "as_of_date": "2011-04-01",
    "items": [
      {
        "store_id": "CA_1",
        "item_id": "HOBBIES_1_000_CA_1"
      }
    ],
    "return_quantiles": true
  }' | python -m json.tool
```

### 2. Manual 35-day `past_sales` override

Supply a custom 35-day historical sales array directly:

```bash
curl -s -X POST http://localhost:8000/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "as_of_date": "2011-04-01",
    "items": [
      {
        "store_id": "CA_1",
        "item_id": "HOBBIES_1_000_CA_1",
        "past_sales": [0,1,2,2,0,1,0,1,0,2,0,0,1,2,0,3,3,5,1,2,0,1,0,1,1,2,2,3,2,0,2,1,2,4,2]
      }
    ],
    "return_quantiles": true
  }' | python -m json.tool
```

### Example response

```json
{
  "as_of_date": "2016-04-25",
  "horizon_days": 28,
  "results": [
    {
      "store_id": "CA_1",
      "item_id": "HOBBIES_1_001",
      "status": "success",
      "mean": [0.965, 1.978, 1.172, 1.472, 1.669, 0.905, 1.102, 2.016, 1.172, 1.894, ...],
      "median": [1, 1, 1, 1, 1, 0, 1, 2, 1, 1, ...],
      "dispersion": [0.558, 0.484, 0.396, 0.383, 0.586, 0.891, 0.672, 0.342, ...],
      "quantiles": {
        "p10": [0, 0, 0, 0, 0, 0, 0, 0, ...],
        "p50": [1, 1, 1, 1, 1, 0, 1, 2, ...],
        "p90": [3, 5, 3, 3, 4, 3, 3, 4, ...],
        "median": null
      },
      "error_detail": null
    }
  ]
}
```

---

## Testing & quality assurance

```bash
# Unit, parity, and integration tests
uv run pytest tests/ -v

# Lint checks
uv run ruff check hier_forecast tests
```

---

## Docker deployment

```bash
# Build multi-stage slim image
docker build -t tsmixer-ext-hts:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e WANDB_API_KEY="your_wandb_api_key_here" \
  --name tsmixer-ext-hts \
  tsmixer-ext-hts:latest
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.