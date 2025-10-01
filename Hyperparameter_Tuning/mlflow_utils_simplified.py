# mlflow_utils_simplified.py

import os
import logging
import tempfile
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Experiment tracking will be disabled.")

def _safe_mlflow_op(func):
    """
    Decorator to safely execute MLflow operations.
    - Checks if MLflow is enabled.
    - Ensures an active run exists for logging operations.
    - Catches and logs any exceptions without crashing.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.enabled:
            return False
        # For logging methods, ensure there is an active run
        if "log" in func.__name__ or "set_tag" in func.__name__:
            if not self.active_run:
                self.error("No active MLflow run. Call start_run() first.")
                return False
        try:
            func(self, *args, **kwargs)
            return True
        except Exception as e:
            self.error(f"Failed during '{func.__name__}': {e}")
            return False
    return wrapper

class MLflowLogger:
    """
    Robust MLflow logger with centralized error handling via decorators.
    Provides safe MLflow operations that won't crash your training script.
    """
    def __init__(self, experiment_name: str, run_name: Optional[str] = None, tracking_uri: Optional[str] = None, enabled: bool = True):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.enabled = enabled and MLFLOW_AVAILABLE
        self.active_run = None
        self._run_started_here = False

        if not self.enabled:
            logger.info("MLflow logging is disabled.")
            return

        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.info(f"MLflow experiment set to: {experiment_name}")
        except Exception as e:
            self.error(f"Failed to initialize MLflow: {e}")
            self.enabled = False

    def _ctx(self) -> str:
        run_id = self.active_run.info.run_id if self.active_run else "NoRun"
        return f"[MLflowLogger, exp={self.experiment_name}, run={run_id}]"

    def info(self, msg: str): logger.info(f"{self._ctx()} {msg}")
    def warning(self, msg: str): logger.warning(f"{self._ctx()} {msg}")
    def error(self, msg: str): logger.error(f"{self._ctx()} {msg}")

    @contextmanager
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """Context manager for starting and automatically ending an MLflow run."""
        if not self.enabled:
            yield self
            return

        if mlflow.active_run() and not nested:
            self.warning("MLflow run already active, using existing run.")
            self.active_run = mlflow.active_run()
            self._run_started_here = False
            yield self
        else:
            try:
                self.active_run = mlflow.start_run(run_name=(run_name or self.run_name), nested=nested)
                self._run_started_here = True
                self.info(f"Started MLflow run: {self.active_run.info.run_id}")
                yield self
            except Exception as e:
                self.error(f"Error starting MLflow run: {e}")
                yield self # Yield self even on failure to prevent crashes
            finally:
                if self._run_started_here and self.active_run:
                    try:
                        mlflow.end_run()
                        self.info("MLflow run ended successfully.")
                    except Exception as e:
                        self.error(f"Error ending MLflow run: {e}")
                    finally:
                        self.active_run = None
                        self._run_started_here = False

    @_safe_mlflow_op
    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)
        self.info(f"Logged {len(params)} parameters.")

    def log_param(self, key: str, value: Any):
        return self.log_params({key: value})
        
    @_safe_mlflow_op
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        safe_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
        if len(safe_metrics) != len(metrics):
            self.warning("Some non-numeric or NaN metrics were skipped.")
        if safe_metrics:
            mlflow.log_metrics(safe_metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        return self.log_metrics({key: value}, step=step)

    @_safe_mlflow_op
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        if not os.path.exists(local_path):
            self.warning(f"Artifact path does not exist: {local_path}")
            return # Return here to avoid exception in decorator
        mlflow.log_artifact(local_path, artifact_path)

    @_safe_mlflow_op
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        if not os.path.isdir(local_dir):
            self.warning(f"Artifact directory does not exist: {local_dir}")
            return
        mlflow.log_artifacts(local_dir, artifact_path)

    @_safe_mlflow_op
    def set_tags(self, tags: Dict[str, str]):
        mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def set_tag(self, key: str, value: str):
        return self.set_tags({key: value})

    @_safe_mlflow_op
    def log_text(self, text: str, artifact_file: str):
        """Logs text to a file. Uses modern API with a fallback for older MLflow versions."""
        try:
            mlflow.log_text(text, artifact_file)
        except AttributeError: # Fallback for older mlflow
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(text)
                temp_path = f.name
            try:
                mlflow.log_artifact(temp_path, artifact_path=artifact_file)
            finally:
                os.unlink(temp_path)

    @_safe_mlflow_op
    def log_model_summary(self, model, filename: str = "model_summary.txt"):
        """Logs a text summary of a PyTorch model's architecture and parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        summary = (
            f"Model Architecture Summary\n{'=' * 50}\n\n{model}\n\n"
            f"Total parameters: {total_params:,}\n"
            f"Trainable parameters: {trainable_params:,}\n"
            f"Non-trainable parameters: {total_params - trainable_params:,}\n"
        )
        self.log_text(summary, filename)

    @_safe_mlflow_op
    def log_training_curves(self, history: Dict[str, List[float]], start_epoch: int = 1):
        """Logs metrics from a training history dictionary, epoch by epoch."""
        if not history: return
        num_epochs = len(next(iter(history.values()), []))
        for i in range(num_epochs):
            epoch_metrics = {name: values[i] for name, values in history.items() if i < len(values)}
            if epoch_metrics:
                self.log_metrics(epoch_metrics, step=start_epoch + i)
        self.info(f"Logged training curves for {num_epochs} epochs.")

    def get_run_id(self) -> Optional[str]:
        return self.active_run.info.run_id if self.active_run else None

    @staticmethod
    def log_preview(experiment_name: str, run_name: str, params: Dict[str, Any], enabled: bool = True):
        """Starts a short run to log initial parameters and tags, then ends it."""
        logger = MLflowLogger(experiment_name=experiment_name, run_name=run_name, enabled=enabled)
        if not logger.enabled:
            return True
        with logger.start_run():
            logger.log_params(params)
            logger.set_tags({"phase": "pre_training", "framework": "pytorch"})
        return True

        # Add this method inside the MLflowLogger class

    @_safe_mlflow_op
    def log_pytorch_model(self, model, name: str, input_example=None, **kwargs):
        """Safely logs a PyTorch model."""
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=name,  # Use name as the artifact path for clarity
            input_example=input_example,
            **kwargs
        )
        self.info(f"PyTorch model '{name}' logged successfully.")