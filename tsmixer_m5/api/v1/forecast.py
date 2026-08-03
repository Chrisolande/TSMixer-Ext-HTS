import asyncio

import numpy as np
import torch
from fastapi import APIRouter, BackgroundTasks, Depends
from loguru import logger

from tsmixer_m5.api.dependencies import get_inference_store, get_model_runner
from tsmixer_m5.api.runner import ModelRunner
from tsmixer_m5.api.schemas.request import ForecastRequest
from tsmixer_m5.api.schemas.response import (
    ErrorDetail,
    ForecastResponse,
    ItemForecastResult,
    Quantiles,
)
from tsmixer_m5.api.store import InferenceStore

router = APIRouter(prefix="/v1", tags=["Forecast"])


def run_batch_forecast(
    request_data: ForecastRequest,
    store: InferenceStore,
    runner: ModelRunner,
) -> ForecastResponse:
    """Synchronous CPU/GPU batch tensor assembly and model prediction executed in threadpool."""
    num_items = len(request_data.items)
    results_by_idx: list[ItemForecastResult | None] = [None] * num_items

    valid_indices = []
    valid_items = []
    valid_tensors = []

    for idx, item in enumerate(request_data.items):
        try:
            tensors = store.build_tensors(item)
            valid_indices.append(idx)
            valid_items.append(item)
            valid_tensors.append(tensors)
        except KeyError as e:
            results_by_idx[idx] = ItemForecastResult(
                store_id=item.store_id,
                item_id=item.item_id,
                status="error",
                error_detail=ErrorDetail(
                    code="CATEGORY_NOT_FOUND",
                    message=str(e),
                ),
            )
        except ValueError as e:
            results_by_idx[idx] = ItemForecastResult(
                store_id=item.store_id,
                item_id=item.item_id,
                status="error",
                error_detail=ErrorDetail(
                    code="INVALID_WINDOW_LENGTH",
                    message=str(e),
                ),
            )
        except Exception as e:
            results_by_idx[idx] = ItemForecastResult(
                store_id=item.store_id,
                item_id=item.item_id,
                status="error",
                error_detail=ErrorDetail(
                    code="DATA_SNAPSHOT_MISSING",
                    message=str(e),
                ),
            )

    if valid_tensors:
        x_batch = torch.stack([t["x"] for t in valid_tensors], dim=0)
        x_hist_batch = torch.stack([t["x_hist"] for t in valid_tensors], dim=0)
        z_futr_batch = torch.stack([t["z_futr"] for t in valid_tensors], dim=0)
        s_cat_batch = torch.stack([t["s_cat"] for t in valid_tensors], dim=0)
        s_cont_batch = torch.stack([t["s_cont"] for t in valid_tensors], dim=0)

        mu_tensor, alpha_tensor = runner.predict(x_batch, x_hist_batch, z_futr_batch, s_cat_batch, s_cont_batch)

        mu_np = mu_tensor.numpy()
        alpha_np = alpha_tensor.numpy()

        for batch_idx, original_idx in enumerate(valid_indices):
            item = valid_items[batch_idx]
            item_mu = mu_np[batch_idx]
            item_alpha = alpha_np[batch_idx]

            quantiles_obj = None
            if request_data.return_quantiles:
                q_dict = runner.quantiles(item_mu, item_alpha)
                quantiles_obj = Quantiles(
                    p10=q_dict["p10"],
                    p50=q_dict["p50"],
                    p90=q_dict["p90"],
                )

            results_by_idx[original_idx] = ItemForecastResult(
                store_id=item.store_id,
                item_id=item.item_id,
                status="success",
                mean=np.round(item_mu, 4).tolist(),
                dispersion=np.round(item_alpha, 4).tolist(),
                quantiles=quantiles_obj,
            )

    final_results = [r for r in results_by_idx if r is not None]
    return ForecastResponse(
        as_of_date=request_data.as_of_date,
        horizon_days=28,
        results=final_results,
    )


def log_audit(request_data: ForecastRequest, success_count: int, error_count: int):
    """Background task to log post-response forecast audit metrics without blocking response."""
    logger.info(
        "Forecast Audit | as_of={as_of} | total={total} | success={success} | errors={errors}",
        as_of=request_data.as_of_date,
        total=len(request_data.items),
        success=success_count,
        errors=error_count,
    )


@router.post("/forecast", response_model=ForecastResponse, summary="Batch Forecast Endpoint")
async def post_forecast(
    request_data: ForecastRequest,
    background_tasks: BackgroundTasks,
    runner: ModelRunner = Depends(get_model_runner),
    store: InferenceStore = Depends(get_inference_store),
) -> ForecastResponse:
    """Generate 28-day probabilistic sales predictions for a batch of series keys."""
    response = await asyncio.to_thread(run_batch_forecast, request_data, store, runner)

    success_count = sum(1 for r in response.results if r.status == "success")
    error_count = len(response.results) - success_count

    background_tasks.add_task(log_audit, request_data, success_count, error_count)
    return response
