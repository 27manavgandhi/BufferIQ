"""Model management endpoints."""

from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException

from bufferiq.api.dependencies import get_model_loader
from bufferiq.api.services.model_loader import ModelLoader
from bufferiq.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/models")
async def list_models(
    model_loader: ModelLoader = Depends(get_model_loader),
) -> Dict[str, List[str]]:
    """
    List all available models.

    Returns:
        Dictionary with available models
    """
    return {
        "models": list(model_loader.model_paths.keys()),
        "loaded": list(model_loader.models.keys()),
    }


@router.get("/models/{model_name}")
async def get_model_info(
    model_name: str,
    model_loader: ModelLoader = Depends(get_model_loader),
) -> Dict[str, any]:
    """
    Get information about a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Model metadata
    """
    if model_name not in model_loader.model_paths:
        raise HTTPException(status_code=404, detail="Model not found")

    model_path = model_loader.model_paths[model_name]
    is_loaded = model_name in model_loader.models

    return {
        "name": model_name,
        "path": str(model_path),
        "loaded": is_loaded,
        "exists": model_path.exists(),
    }


@router.post("/models/{model_name}/reload")
async def reload_model(
    model_name: str,
    model_loader: ModelLoader = Depends(get_model_loader),
) -> Dict[str, str]:
    """
    Reload a specific model.

    Args:
        model_name: Name of the model to reload

    Returns:
        Status message
    """
    try:
        model_loader.reload(model_name)
        return {"status": "success", "message": f"Model {model_name} reloaded"}
    except Exception as e:
        logger.error(f"Failed to reload model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))