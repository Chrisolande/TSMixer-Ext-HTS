from contextlib import asynccontextmanager
import time
import uuid
from loguru import logger

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from scalar_fastapi import get_scalar_api_reference

from tsmixer_m5.api.config import config
from tsmixer_m5.api.runner import ModelRunner, DEFAULT_CATEGORY_MAPS
from tsmixer_m5.api.schemas.response import ErrorResponse
from tsmixer_m5.api.store import InferenceStore
from tsmixer_m5.api.v1.forecast import router as forecast_router
from tsmixer_m5.api.v1.health import router as health_router
from tsmixer_m5.modeling import TSMixerExt


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ModelRunner and InferenceStore on startup."""
    logger.info("Initializing FastAPI Inference Service lifespan...")

    try:
        runner = ModelRunner.from_wandb(
            wandb_artifact=config.wandb_model_artifact,
            wandb_api_key=config.wandb_api_key,
            local_dir=config.model_artifact_local_dir,
            device=config.device,
            use_amp=config.use_amp,
        )
    except Exception as e:
        logger.warning("Could not load trained weights ({e}); falling back to initialized model for API demo.", e=e)
        model = TSMixerExt(
            seq_len=35,
            pred_len=28,
            num_features=1,
            hist_exog_dim=10,
            futr_exog_dim=10,
            static_cont_dim=1,
            cat_cardinalities=[10, 30490, 7, 3, 3],
            cat_emb_dims=[8, 8, 8, 8, 16],
            num_blocks=2,
            hidden_size=32,
            probabilistic=True,
            use_mean_scaling=True,
        )
        runner = ModelRunner(model=model, category_maps=DEFAULT_CATEGORY_MAPS, device=config.device, use_amp=False)

    store = InferenceStore(category_maps=runner.category_maps, snapshot_dir=config.data_snapshot_dir)

    app.state.model_runner = runner
    app.state.inference_store = store

    logger.success("ModelRunner and InferenceStore initialized successfully.")
    yield
    logger.info("Tearing down FastAPI Inference Service lifespan.")


def create_app() -> FastAPI:
    """Create and configure main FastAPI application instance."""
    app = FastAPI(
        title=config.title,
        version=config.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_obs_headers(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-MS"] = str(duration_ms)
        return response

    @app.exception_handler(RequestValidationError)
    async def handle_val_err(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                code="VALIDATION_ERROR",
                message="Request payload validation failed",
                details={"errors": exc.errors()},
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_global_err(request: Request, exc: Exception):
        logger.error("Unhandled exception: {e}", e=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected server error occurred",
                details={"error": str(exc)},
            ).model_dump(),
        )

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/scalar")

    @app.get("/scalar", include_in_schema=False)
    async def scalar_html():
        return get_scalar_api_reference(
            openapi_url=app.openapi_url,
            title=app.title,
        )

    app.include_router(health_router)
    app.include_router(forecast_router)

    return app


app = create_app()
