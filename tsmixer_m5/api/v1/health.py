from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from tsmixer_m5.api.dependencies import get_model_runner
from tsmixer_m5.api.runner import ModelRunner

router = APIRouter(tags=["Health"])


@router.get("/healthz", summary="Liveness Probe")
async def get_healthz():
    """Returns HTTP 200 if process is running."""
    return {"status": "healthy"}


@router.get("/readyz", summary="Readiness Probe")
async def get_readyz(runner: ModelRunner = Depends(get_model_runner)):
    """Verifies model is loaded and ready to serve traffic."""
    if not runner:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unready", "reason": "ModelRunner not initialized"},
        )
    return {"status": "ready", "device": runner.device}
