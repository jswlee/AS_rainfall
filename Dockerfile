# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only what we need for the API
COPY api/requirements.txt ./api/
COPY Hyperparameter_Tuning/ ./Hyperparameter_Tuning/

# Install API dependencies only
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r api/requirements.txt

# Copy API code and artifacts
COPY api/serve.py .
COPY api/artifacts/ ./artifacts/

# Set environment variables for artifact paths
ENV MODEL_PATH=/app/artifacts/best_model.pth
ENV HYPERPARAMS_PATH=/app/artifacts/hyperparams.json
ENV PREPROCESSING_PATH=/app/artifacts/preprocessing.json

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Run the application
CMD ["python", "serve.py"]
