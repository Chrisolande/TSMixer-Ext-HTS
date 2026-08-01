from fastapi import Request

from tsmixer_m5.api.runner import ModelRunner
from tsmixer_m5.api.store import InferenceStore


def get_model_runner(request: Request) -> ModelRunner:
    """Dependency provider for ModelRunner stored on app.state."""
    runner = getattr(request.app.state, "model_runner", None)
    if runner is None:
        raise RuntimeError("ModelRunner is not initialized on app.state")
    return runner


def get_inference_store(request: Request) -> InferenceStore:
    """Dependency provider for InferenceStore stored on app.state."""
    store = getattr(request.app.state, "inference_store", None)
    if store is None:
        raise RuntimeError("InferenceStore is not initialized on app.state")
    return store
