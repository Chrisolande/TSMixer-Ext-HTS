#  TSMixer M5: Hierarchical & Probabilistic Time Series Forecasting

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference%20Service-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage%20Slim-2496ED.svg)](https://www.docker.com/)
[![W&B Tracked](https://img.shields.io/badge/Weights_%26_Biases-Tracked-yellow.svg)](https://wandb.ai/olandechris-/tsmixer-m5/reports/TSMixer-M5-Test-WRMSSE-Optimization-Report--VmlldzoxNzY5MzQ3NA)
[![Optuna](https://img.shields.io/badge/Optuna-HPO%20Enabled-blue)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end implementation of **TSMixerExt** (Extended Time-Series Mixer) adapted for the **30,490-series M5 Hierarchical Forecasting Dataset**. This repository combines probabilistic count modeling (Negative Binomial likelihood), sub-100ms sparse matrix hierarchy aggregation across 42,840 nodes, automated parallel Optuna hyperparameter optimization (HPO), native Weights & Biases experiment tracking, and a **FastAPI inference service** with Docker containerization for real-time probabilistic forecasting.

---

##  Project Overview & Empirical Results

The M5 Forecasting competition requires predicting daily unit sales across 30,490 bottom-level series organized into a 12-level hierarchy totaling 42,840 time series nodes.

### Key Empirical Results:
- **WRMSSE Metric**: Achieves a **0.575 WRMSSE** averaged across all 12 hierarchy levels on the M5 evaluation set.
- **Multi-Seed Stability**: Evaluated over 3 independent random seeds (`42, 43, 44`) with low variance.
- **Fast Aggregation**: SciPy sparse matrix hierarchy construction executes in **<100ms** without costly pandas DataFrame melting operations.
- **GPU-Accelerated Evaluation**: PyTorch GPU sparse matrix multiplication (`torch.sparse.mm`) enables high-throughput validation across all 42,840 nodes.
- **Real-Time Inference API**: FastAPI service delivers 28-day probabilistic forecasts ($\mu, \alpha$, p10/p50/p90) from real M5 sales history in a single HTTP request.
- **Containerization**: Multi-stage slim Docker image built using `uv` with zero dev bloat and standard library health check probe.

---

## ️ Core Architectural Features

```
                          ┌──────────────────────────┐
                          │    Static Metadata       │
                          │ (Categorical + Cont.)    │
                          └────────────┬─────────────┘
                                       │ Static Embeddings
                                       ▼
┌──────────────────┐           ┌──────────────┐
│  Past Sales &    │ ────────► │  Temporal    │ ────┐
│ Historical Exog  │ (L steps) │ Projection   │     │
└──────────────────┘           └──────────────┘     │
                                                    ▼
                                       ┌──────────────────────────┐
                                       │ Conditional Feature      │ ◄── Static Vector (v_static)
                                       │ Mixing (Past & Future)   │
                                       └────────────┬─────────────┘
                                                    │
┌──────────────────┐                                ▼
│ Future Exogenous │ (T steps) ──────► ┌──────────────────────────┐
│ Covariates       │                   │ Stacked Conditional      │ ◄── Static Vector (v_static)
└──────────────────┘                   │ Mixer Layers (1..N)      │
                                       └────────────┬─────────────┘
                                                    │
                                                    ▼
                                       ┌──────────────────────────┐
                                       │   Probabilistic Head     │
                                       │ (Softplus μ, Softplus α) │
                                       └────────────┬─────────────┘
                                                    │
                                                    ▼
                                       ┌──────────────────────────┐
                                       │ Continuous Negative      │
                                       │ Binomial Log-Likelihood  │
                                       └──────────────────────────┘
                                       └────────────┬─────────────┘
```

1. **Probabilistic Negative Binomial Loss ($\mu, \alpha$)**:
   - Models intermittent, zero-inflated count data using dual softplus output heads ($\mu > 0, \alpha > 0$).
   - Minimizes continuous Negative Binomial negative log-likelihood ($\mathcal{L}_{\text{NLL}}$).
2. **Reversible Instance Normalization ($\text{RevIN}$) & Scale Normalization**:
   - `RevIN` eliminates temporal distribution shifts between historical input windows ($L=35$) and forecasting horizons ($T=28$).
   - Optional `MeanScaling` normalizes time-series magnitude using historical average demand.
3. **Fast 12-Level Hierarchy Aggregation ($S$-Matrix)**:
   - Constructs a sparse CSR aggregation matrix $S \in \mathbb{R}^{42840 \times 30490}$ linking bottom-level items to all 12 hierarchy levels (Total, State, Store, Category, Department, and cross-interactions).
4. **Exogenous Feature & Metadata Mixing**:
   - **Past & Future Exogenous Features**: Combines calendar indicators (`wday`, `day`, `dayofyear`), holiday event codes (Cultural, National, Religious, Sporting), SNAP benefits (`snap_CA`, `snap_TX`, `snap_WI`), and dual price z-score normalization (item-level and department-level).
   - **Static Metadata Embeddings**: Categorical embedding layers for `state_id`, `store_id`, `cat_id`, `dept_id`, and `item_id` combined with continuous historical sales averages.

---

##  Mathematical Formulations

### 1. Reversible Instance Normalization ($\text{RevIN}$)
To prevent distribution shift across sliding historical windows:

$$\hat{x} = \gamma \odot \left( \frac{x - \mu_x}{\sigma_x + \epsilon} \right) + \beta$$

where $\mu_x, \sigma_x$ are the mean and standard deviation along the temporal dimension $L$, and $\gamma, \beta$ are learnable affine parameters.

### 2. Probabilistic Negative Binomial Loss
For overdispersed count data $y \sim \text{NB}(\mu, \alpha)$ parameterized by mean $\mu$ and dispersion parameter $\alpha$:

$$r = \frac{1}{\alpha}, \quad p = \frac{1}{1 + \alpha \mu}$$

$$\mathcal{L}_{\text{NLL}} = -\left[ \log \Gamma(y + r) - \log \Gamma(y + 1) - \log \Gamma(r) + r \log(p + \epsilon) + y \log(1 - p + \epsilon) \right]$$

### 3. Weighted Root Mean Squared Scaled Error (WRMSSE)
The official M5 metric across all $K = 42,840$ hierarchy nodes:

$$\text{WRMSSE} = \sum_{i=1}^{K} w_i \sqrt{ \frac{\frac{1}{h} \sum_{t=n+1}^{n+h} (\hat{y}_{i,t} - y_{i,t})^2}{c_i} }$$

where $c_i$ is the scale factor computed from active training sales:

$$c_i = \frac{1}{n_i - 1} \sum_{t=2}^{n_i} (y_{i,t} - y_{i,t-1})^2$$

and $w_i$ represents the dollar-revenue weight assigned to node $i$ across all 12 hierarchy levels.

---

##  Project Structure

```
TSMixer-Ext-HTS/
├── tsmixer_m5/
│   ├── __init__.py
│   ├── data.py
│   ├── modeling.py
│   ├── wrmsse.py
│   ├── metrics.py
│   ├── hparam_search.py
│   ├── training.py
│   ├── utils.py
│   └── api/
│       ├── app.py
│       ├── config.py
│       ├── runner.py
│       ├── store.py
│       ├── dependencies.py
│       ├── schemas/
│       │   ├── request.py
│       │   └── response.py
│       └── v1/
│           ├── forecast.py
│           └── health.py
├── data/
│   ├── m5/
│   └── m5_sample/
├── Dockerfile
├── healthcheck.py
├── pyproject.toml
├── uv.lock
└── README.md
```

---

##  Getting Started & Setup

### 1. Prerequisites & Installation

Clone the repository and set up the environment with `uv` (recommended) or `pip`:

```bash
git clone https://github.com/Chrisolande/TSMixer-Ext-HTS.git
cd TSMixer-Ext-HTS

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install all dependencies (includes FastAPI, uvicorn, scalar-fastapi)
pip install -e ".[dev]"
```

### 2. Dataset Layout
Place the official M5 dataset CSV files in `./data/m5/`:

```
data/
├── m5/
│   ├── calendar.csv
│   ├── sell_prices.csv
│   └── sales_train_evaluation.csv    # or sales_train_validation.csv
└── m5_sample/
    └── sales_train_evaluation.csv    # 5-item subset - used by API default data
```

> The `data/m5_sample/` subset is used automatically by the inference API when `past_sales` is omitted from a request. It contains real M5 evaluation rows for `HOBBIES_1_000` through `HOBBIES_1_004` at store `CA_1`.

### 3. Environment Variables
Set your Weights & Biases API key for automated experiment logging and model artifact download:

```bash
export WANDB_API_KEY="your_wandb_api_key_here"

# Optional overrides (defaults shown)
export WANDB_MODEL_ARTIFACT="olandechris-/tsmixer-m5/tsmixer_m5_seed_43:v0"
export MODEL_ARTIFACT_LOCAL_DIR="./artifact"
export DATA_SNAPSHOT_DIR="./data/m5_sample"
export DEVICE="cpu"
```

---

##  Usage & Workflows

### 1. Parallel Hyperparameter Optimization (Optuna HPO)

Run multi-GPU Optuna HPO study with TPE Sampler and MedianPruner:

```bash
python -m tsmixer_m5.hparam_search
```

Or invoke programmatically:

```python
from tsmixer_m5.hparam_search import run_optuna_study

best_params = run_optuna_study(
    n_trials=100,
    data_dir="./data/m5",
    storage_url="sqlite:///m5_optuna.db"
)
print("Optimal Hyperparameters:", best_params)
```

### 2. Final 3-Seed Model Training & Evaluation

Train the final `TSMixerExt` model across 3 seeds (`42, 43, 44`) using optimal hyperparameters:

```bash
python -m tsmixer_m5.training
```

Or run via Python API:

```python
from tsmixer_m5.training import train_and_validate

mean_score, std_score = train_and_validate(
    seeds=(42, 43, 44),
    lr=0.0005066,
    hidden_size=128,
    num_blocks=8,
    dropout=0.1,
    batch_size=1024,
    num_batches_per_epoch=200,
    wandb_project="tsmixer-m5"
)
print(f"Final Evaluation WRMSSE: {mean_score:.4f} ± {std_score:.4f}")
```

### 3. FastAPI Inference Service

#### Local Development Server

```bash
uvicorn tsmixer_m5.api.app:app --host 0.0.0.0 --port 8000 --reload
```

On startup the service will:
1. Attempt to download the best checkpoint from W&B Model Registry (requires `WANDB_API_KEY`).
2. Fall back to `./artifact/best_wrmsse_seed_42.pth` or `./best_wrmsse_seed_42.pth` if W&B is unavailable.
3. Infer `hidden_size`, `cat_cardinalities`, and `cat_emb_dims` directly from the checkpoint state-dict.
4. Load the M5 sample snapshot from `./data/m5_sample/` for history-free requests.

#### Docker Container Deployment

Build and run the multi-stage slim container image:

```bash
# Build image with uv
docker build -t tsmixer-ext-hts:latest .

# Run containerized service
docker run -d \
  -p 8000:8000 \
  -e WANDB_API_KEY="your_wandb_api_key_here" \
  --name tsmixer-ext-hts \
  tsmixer-ext-hts:latest
```

#### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/healthz` | Liveness probe - returns `{"status":"healthy"}` |
| `GET` | `/readyz` | Readiness probe - confirms model is loaded |
| `POST` | `/v1/forecast` | Batch 28-day probabilistic forecast |
| `GET` | `/scalar` | Modern Scalar interactive API docs |
| `GET` | `/docs` | Classic Swagger UI |

#### Forecast via curl

```bash
# Single item - uses on-disk snapshot automatically (no past_sales needed)
curl -s -X POST http://localhost:8000/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "as_of_date": "2016-04-25",
    "items": [{"store_id": "CA_1", "item_id": "HOBBIES_1_001"}],
    "return_quantiles": true
  }' | python -m json.tool

# Batch of multiple items
curl -s -X POST http://localhost:8000/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "as_of_date": "2016-04-25",
    "items": [
      {"store_id": "CA_1", "item_id": "HOBBIES_1_001"},
      {"store_id": "CA_1", "item_id": "HOBBIES_1_002"}
    ],
    "return_quantiles": true
  }' | python -m json.tool

# Manual 35-day sales override
curl -s -X POST http://localhost:8000/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "as_of_date": "2016-04-25",
    "items": [{
      "store_id": "CA_1",
      "item_id": "HOBBIES_1_001",
      "past_sales": [0,1,2,2,0,1,0,1,0,2,0,0,1,2,0,3,3,5,1,2,0,1,0,1,1,2,2,3,2,0,2,1,2,4,2]
    }],
    "return_quantiles": true
  }' | python -m json.tool
```

#### Example Response

```json
{
  "as_of_date": "2016-04-25",
  "horizon_days": 28,
  "results": [
    {
      "store_id": "CA_1",
      "item_id": "HOBBIES_1_001",
      "status": "success",
      "mean": [1.107, 1.352, 1.297, ...],
      "dispersion": [0.754, 0.793, 0.590, ...],
      "quantiles": {
        "p10": [0.0, 0.0, ...],
        "p50": [1.107, 1.352, ...],
        "p90": [2.933, 3.498, ...]
      },
      "error_detail": null
    }
  ]
}
```

> **Note on p10 zeros**: For low-demand items (~1-2 units/day), `P(sales=0)` can exceed 10%, making the 10th percentile mathematically zero. This is correct Negative Binomial behaviour, not a bug.

#### Running Tests

```bash
# Full test suite (unit + integration, no server needed)
python -m pytest -v

# API integration tests only
python -m pytest tests/test_fastapi_forecast_route.py -v

# Store encoding unit tests only
python -m pytest tests/test_api_store.py -v
```

#### Service Architecture

```
Request (POST /v1/forecast)
        │
        ▼
  FastAPI Router
        │
        ▼
  asyncio.to_thread()          ← keeps event loop unblocked
        │
        ▼
  InferenceStore               ← encodes store/item → integer indices
  .build_tensors()             ← assembles 35-day history from snapshot CSV
        │
        ▼
  ModelRunner.predict()        ← batched forward pass with AMP (bfloat16 on CPU)
        │
        ▼
  Negative Binomial quantiles  ← scipy.stats NB ppf for p10/p50/p90
        │
        ▼
  ForecastResponse (JSON)
```

**Observability headers** are automatically injected on every response:
- `X-Request-ID` - UUID trace identifier (pass your own via request header)
- `X-Response-Time-MS` - wall-clock latency in milliseconds

---

##  Weights & Biases Integration

- **Live Interactive W&B Report**: Explore training curves, Optuna parameter importance, and evaluation dashboards on [Weights & Biases Reports](https://wandb.ai/olandechris-/tsmixer-m5/reports/TSMixer-M5-Test-WRMSSE-Optimization-Report--VmlldzoxNzY5MzQ3NA).
- **Automated Metric Tracking**: Real-time logging of train NLL, validation NLL, learning rate schedules, and step-based WRMSSE.
- **Model Registry & Artifacts**: Best performing checkpoints (`best_wrmsse_seed_{seed}.pth`) are automatically uploaded to W&B Model Catalog using `run.log_model()`.
- **API Auto-Download**: The FastAPI service pulls the registered artifact at startup via `wandb.Api().artifact(...)` - no manual checkpoint management required.

---

##  License

Distributed under the MIT License. See `LICENSE` for details.
