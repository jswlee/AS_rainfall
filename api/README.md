# Rainfall Prediction API

A containerized FastAPI service for predicting rainfall in American Samoa using a PyTorch deep learning model.

## Quick Start

### 1. Build the Docker image
```bash
cd api/
./build.sh
```

### 2. Run the container locally
```bash
docker run --rm -p 8080:8080 rainfall-inference:latest
```

### 3. Test the API
```bash
# Health check
curl http://localhost:8080/healthz

# Make a prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

## API Endpoints

### GET /healthz
Health check endpoint that returns server status and model loading status.

### POST /predict
Main prediction endpoint. Expects JSON with:
- `climate_data`: 16x3x3 array of climate reanalysis data
- `local_dem`: 3x3 array of local DEM data  
- `regional_dem`: 3x3 array of regional DEM data
- `month`: Integer 1-12 representing the month

Returns:
- `prediction`: Predicted rainfall in mm
- `model_version`: Model version/trial information

### GET /model/info
Returns information about the loaded model including hyperparameters and parameter count.

## Cloud Deployment

### Google Cloud Run
```bash
# Build and push to Google Container Registry
docker tag rainfall-inference:latest gcr.io/YOUR_PROJECT/rainfall-inference:latest
docker push gcr.io/YOUR_PROJECT/rainfall-inference:latest

# Deploy to Cloud Run
gcloud run deploy rainfall-api \
  --image gcr.io/YOUR_PROJECT/rainfall-inference:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### AWS ECS/Fargate
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker tag rainfall-inference:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/rainfall-inference:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/rainfall-inference:latest

# Deploy using ECS task definition with Fargate
```

### Azure Container Apps
```bash
# Push to Azure Container Registry
az acr login --name YOUR_REGISTRY
docker tag rainfall-inference:latest YOUR_REGISTRY.azurecr.io/rainfall-inference:latest
docker push YOUR_REGISTRY.azurecr.io/rainfall-inference:latest

# Deploy to Container Apps
az containerapp create \
  --name rainfall-api \
  --resource-group YOUR_RG \
  --environment YOUR_ENV \
  --image YOUR_REGISTRY.azurecr.io/rainfall-inference:latest \
  --target-port 8080 \
  --ingress external \
  --cpu 1.0 \
  --memory 2.0Gi
```

## Files Structure

```
api/
├── serve.py                    # FastAPI application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition
├── build.sh                    # Build script
├── test_request.json          # Example API request
├── artifacts/                  # Model artifacts
│   ├── best_model.pth         # Trained model weights
│   ├── hyperparams.json       # Model hyperparameters
│   └── preprocessing.json     # Data preprocessing stats
├── Hyperparameter_Tuning/     # Model architecture code
│   ├── __init__.py
│   └── pytorch_model.py
├── extract_preprocessing_stats.py  # Utility to extract stats
└── prepare_artifacts.py       # Utility to prepare artifacts
```

## Environment Variables

- `MODEL_PATH`: Path to model weights file (default: `/app/artifacts/best_model.pth`)
- `HYPERPARAMS_PATH`: Path to hyperparameters file (default: `/app/artifacts/hyperparams.json`)
- `PREPROCESSING_PATH`: Path to preprocessing stats (default: `/app/artifacts/preprocessing.json`)
- `PORT`: Server port (default: `8080`)

## Model Information

The model uses a custom PyTorch architecture that processes:
- Climate reanalysis data (16 variables in 3x3 patches)
- Local and regional DEM data (3x3 patches)
- Month information (one-hot encoded)

The model was trained using hyperparameter tuning with Optuna and cross-validation.
