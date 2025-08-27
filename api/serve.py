"""
FastAPI server for rainfall prediction model inference.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model architecture from installed package
from Hyperparameter_Tuning.pytorch_model import create_model_from_hyperparams

app = FastAPI(
    title="Rainfall Prediction API",
    description="PyTorch model for predicting rainfall in American Samoa",
    version="1.0.0"
)

# Global model and preprocessing artifacts
model = None
preprocessing_stats = None
hyperparams = None
loss_name: Optional[str] = None
loss_params: Optional[Dict[str, Any]] = None

class PredictRequest(BaseModel):
    """Request schema for rainfall prediction."""
    climate_data: List[List[List[float]]] = Field(
        ..., 
        description="Climate reanalysis data, shape [16, 3, 3]",
        min_items=16, max_items=16
    )
    local_dem: List[List[float]] = Field(
        ..., 
        description="Local DEM data, shape [3, 3]",
        min_items=3, max_items=3
    )
    regional_dem: List[List[float]] = Field(
        ..., 
        description="Regional DEM data, shape [3, 3]",
        min_items=3, max_items=3
    )
    month: int = Field(
        ..., 
        description="Month (1-12)",
        ge=1, le=12
    )

    class Config:
        schema_extra = {
            "example": {
                "climate_data": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]] for _ in range(16)],
                "local_dem": [[100.0, 110.0, 120.0], [105.0, 115.0, 125.0], [108.0, 118.0, 128.0]],
                "regional_dem": [[200.0, 210.0, 220.0], [205.0, 215.0, 225.0], [208.0, 218.0, 228.0]],
                "month": 7
            }
        }

class PredictResponse(BaseModel):
    """Response schema for rainfall prediction."""
    prediction: float = Field(..., description="Predicted rainfall in mm")
    model_version: str = Field(..., description="Model version/trial info")

def load_preprocessing_stats() -> Dict[str, Any]:
    """Load preprocessing statistics from training."""
    stats_path = os.environ.get("PREPROCESSING_PATH", "/app/artifacts/preprocessing.json")
    
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Preprocessing stats not found at {stats_path}, using defaults")
        # Return default stats - you'll need to extract these from your training
        return {
            "dem_local_min": 0.0,
            "dem_local_max": 1000.0,
            "dem_regional_min": 0.0,
            "dem_regional_max": 1000.0,
            "climate_mean": 0.0,
            "climate_std": 1.0
        }

def preprocess_input(
    climate_data: np.ndarray,
    local_dem: np.ndarray, 
    regional_dem: np.ndarray,
    month: int
) -> torch.Tensor:
    """
    Preprocess input data to match training pipeline.
    
    Args:
        climate_data: Shape [16, 3, 3]
        local_dem: Shape [3, 3] 
        regional_dem: Shape [3, 3]
        month: Integer 1-12
        
    Returns:
        Preprocessed tensors ready for model
    """
    device = next(model.parameters()).device
    
    # Convert to tensors and add batch dimension
    climate_tensor = torch.from_numpy(climate_data).float().unsqueeze(0)  # [1, 16, 3, 3]
    local_dem_tensor = torch.from_numpy(local_dem).float().unsqueeze(0)   # [1, 3, 3]
    regional_dem_tensor = torch.from_numpy(regional_dem).float().unsqueeze(0)  # [1, 3, 3]
    
    # Create month one-hot encoding
    month_onehot = torch.zeros(1, 12)
    month_onehot[0, month - 1] = 1.0  # month is 1-indexed
    
    # Apply preprocessing (normalize based on training stats)
    if preprocessing_stats:
        # Normalize DEMs if stats available
        if "dem_local_min" in preprocessing_stats:
            local_min = preprocessing_stats["dem_local_min"]
            local_max = preprocessing_stats["dem_local_max"]
            local_dem_tensor = (local_dem_tensor - local_min) / (local_max - local_min)
            
        if "dem_regional_min" in preprocessing_stats:
            regional_min = preprocessing_stats["dem_regional_min"]
            regional_max = preprocessing_stats["dem_regional_max"]
            regional_dem_tensor = (regional_dem_tensor - regional_min) / (regional_max - regional_min)
    
    # Move to device
    climate_tensor = climate_tensor.to(device)
    local_dem_tensor = local_dem_tensor.to(device)
    regional_dem_tensor = regional_dem_tensor.to(device)
    month_onehot = month_onehot.to(device)
    
    return {
        'climate': climate_tensor,
        'local_dem': local_dem_tensor,
        'regional_dem': regional_dem_tensor,
        'month': month_onehot
    }

def load_model_artifacts():
    """Load model and preprocessing artifacts at startup."""
    global model, preprocessing_stats, hyperparams, loss_name, loss_params
    
    # Paths (can be overridden with environment variables)
    model_path = os.environ.get("MODEL_PATH", "/app/artifacts/best_model.pth")
    hyperparams_path = os.environ.get("HYPERPARAMS_PATH", "/app/artifacts/hyperparams.json")
    
    logger.info(f"Loading model from {model_path}")
    logger.info(f"Loading hyperparams from {hyperparams_path}")
    
    # Load hyperparameters
    with open(hyperparams_path, 'r') as f:
        hyperparams_data = json.load(f)
        
    # Handle both old and new hyperparameter formats
    if 'hyperparameters' in hyperparams_data:
        hyperparams = hyperparams_data['hyperparameters']
        trial_info = f"trial_{hyperparams_data.get('trial_number', 'unknown')}"
    else:
        hyperparams = hyperparams_data
        trial_info = "legacy"
    
    logger.info(f"Loaded hyperparams: {hyperparams}")

    # Try to extract loss info if present in hyperparams
    loss_name = hyperparams.get('loss_name') if isinstance(hyperparams, dict) else None
    # loss_params could be nested or absent
    lp = hyperparams.get('loss_params') if isinstance(hyperparams, dict) else None
    loss_params = lp if isinstance(lp, dict) else None
    
    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Using device: {device}")
    
    # Create model architecture with required metadata
    metadata = {
        'climate_shape': (16, 3, 3),
        'local_dem_shape': (3, 3),
        'regional_dem_shape': (3, 3),
        'num_month_encodings': 12
    }
    model = create_model_from_hyperparams(hyperparams, metadata)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load preprocessing stats
    preprocessing_stats = load_preprocessing_stats()
    
    logger.info("Model loaded successfully!")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

@app.on_event("startup")
async def startup_event():
    """Load model artifacts when the server starts."""
    load_model_artifacts()

@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(next(model.parameters()).device) if model else None
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict rainfall based on input features.
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Validate input shapes
        climate_array = np.array(request.climate_data, dtype=np.float32)
        local_dem_array = np.array(request.local_dem, dtype=np.float32)
        regional_dem_array = np.array(request.regional_dem, dtype=np.float32)
        
        if climate_array.shape != (16, 3, 3):
            raise HTTPException(
                status_code=400, 
                detail=f"climate_data must be shape [16, 3, 3], got {climate_array.shape}"
            )
        if local_dem_array.shape != (3, 3):
            raise HTTPException(
                status_code=400,
                detail=f"local_dem must be shape [3, 3], got {local_dem_array.shape}"
            )
        if regional_dem_array.shape != (3, 3):
            raise HTTPException(
                status_code=400,
                detail=f"regional_dem must be shape [3, 3], got {regional_dem_array.shape}"
            )
        
        # Preprocess inputs
        inputs = preprocess_input(
            climate_array, 
            local_dem_array, 
            regional_dem_array, 
            request.month
        )
        
        # Run inference
        with torch.no_grad():
            prediction = model(inputs)
            
        # Convert to scalar
        pred_value = float(prediction.cpu().item())
        
        # Get model version info
        trial_info = "unknown"
        if hyperparams and isinstance(hyperparams, dict):
            # Try to get trial info from loaded hyperparams
            trial_info = "loaded_hyperparams"
        
        return PredictResponse(
            prediction=pred_value,
            model_version=trial_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "hyperparameters": hyperparams,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(next(model.parameters()).device),
        "preprocessing_stats_loaded": preprocessing_stats is not None,
        "loss_name": loss_name,
        "loss_params": loss_params
    }

if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        reload=False
    )
