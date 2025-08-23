#!/usr/bin/env python3
"""
MLflow utilities for robust experiment tracking in rainfall prediction models.

This module provides:
- Safe MLflow logging with comprehensive error handling
- Standardized logging patterns for hyperparameter tuning, training, and ensemble methods
- Clear documentation for MLOps beginners
- Graceful degradation when MLflow is unavailable

MLOps Best Practices Implemented:
1. Experiment organization: Group related runs under meaningful experiment names
2. Parameter tracking: Log all hyperparameters for reproducibility
3. Metric tracking: Log training curves and evaluation metrics with proper steps
4. Artifact management: Store models, plots, and results files
5. Error resilience: Continue execution even if logging fails
6. Documentation: Clear comments explaining what and why we log
"""

import os
import logging
import tempfile
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager

# Configure logging to show MLflow issues without breaking execution
logger = logging.getLogger(__name__)

# Gracefully handle if not available
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Experiment tracking will be disabled.")


class MLflowLogger:
    """
    Robust MLflow logger with comprehensive error handling.
    
    This class provides safe MLflow logging operations that won't crash your training
    if MLflow encounters issues. All operations are wrapped in try-catch blocks.
    
    Key Features:
    - Automatic experiment creation and management
    - Safe parameter, metric, and artifact logging
    - Graceful degradation when MLflow is unavailable
    - Clear error reporting without stopping execution
    - Context manager support for run lifecycle
    """
    
    def __init__(self, 
                 experiment_name: str,
                 run_name: Optional[str] = None,
                 tracking_uri: Optional[str] = None,
                 enabled: bool = True):
        """
        Initialize MLflow logger.
        
        Args:
            experiment_name: Name of the MLflow experiment (groups related runs)
            run_name: Optional name for this specific run
            tracking_uri: Optional MLflow tracking server URI (defaults to local ./mlruns)
            enabled: Whether to enable MLflow logging (allows easy disable for debugging)
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.enabled = enabled and MLFLOW_AVAILABLE
        self.active_run = None
        self._run_started_here = False
        
        if not self.enabled:
            if not MLFLOW_AVAILABLE:
                logger.info("MLflow logging disabled: MLflow not available")
            else:
                logger.info("MLflow logging disabled by user")
            return
            
        try:
            # Set tracking URI if provided (useful for remote MLflow servers)
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                
            # Create or get experiment
            # Experiments help organize runs by project/model type
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment set to: {experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.enabled = False

    # ------------------------------------------------------------------
    # Instance logging helpers to avoid AttributeErrors and add run context
    # ------------------------------------------------------------------
    def _ctx(self) -> str:
        """Build a short context string with experiment and run id."""
        run_id = None
        try:
            if self.active_run is not None:
                run_id = self.active_run.info.run_id
        except Exception:
            run_id = None
        parts = ["MLflowLogger"]
        if self.experiment_name:
            parts.append(f"exp={self.experiment_name}")
        if run_id:
            parts.append(f"run={run_id}")
        return "[" + ", ".join(parts) + "]"

    def debug(self, msg: str):
        logger.debug(f"{self._ctx()} {msg}")

    def info(self, msg: str):
        logger.info(f"{self._ctx()} {msg}")

    def warning(self, msg: str):
        logger.warning(f"{self._ctx()} {msg}")

    def error(self, msg: str):
        logger.error(f"{self._ctx()} {msg}")
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """
        Context manager for MLflow runs.
        
        Usage:
            with logger.start_run("my_training_run"):
                logger.log_param("learning_rate", 0.001)
                # ... training code ...
                logger.log_metric("accuracy", 0.95)
        
        Args:
            run_name: Optional name for this run
            nested: Whether this is a nested run (for hierarchical experiments)
        """
        if not self.enabled:
            yield self
            return
            
        try:
            # Use provided run_name or fall back to instance default
            name = run_name or self.run_name
            
            # Check if we're already in an active run
            if mlflow.active_run() is not None and not nested:
                logger.warning("MLflow run already active, using existing run")
                self.active_run = mlflow.active_run()
                self._run_started_here = False
            else:
                # Start new run
                self.active_run = mlflow.start_run(run_name=name, nested=nested)
                self._run_started_here = True
                logger.info(f"Started MLflow run: {self.active_run.info.run_id}")
                
            yield self
            
        except Exception as e:
            logger.error(f"Error in MLflow run context: {e}")
            yield self
            
        finally:
            # Clean up run if we started it
            if self._run_started_here and self.enabled:
                try:
                    mlflow.end_run()
                    logger.info("MLflow run ended successfully")
                except Exception as e:
                    logger.error(f"Error ending MLflow run: {e}")
                finally:
                    self.active_run = None
                    self._run_started_here = False
    
    def log_params(self, params: Dict[str, Any]) -> bool:
        """
        Log parameters (hyperparameters, configuration values).
        
        Parameters are immutable values that define your experiment setup.
        Examples: learning_rate, batch_size, model_architecture, etc.
        
        Args:
            params: Dictionary of parameter name -> value pairs
            
        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.active_run:
            self.error("No active MLflow run. Call start_run() first.")
            return False
            
        try:
            # Get existing parameters to avoid duplicates
            existing_params = {}
            try:
                run_data = mlflow.get_run(self.active_run.info.run_id)
                existing_params = run_data.data.params
            except Exception:
                # If we can't get existing params, proceed with caution
                pass
            
            # Filter out parameters that already exist with the same value
            new_params = {}
            skipped_params = []
            
            for key, value in params.items():
                str_value = str(value)
                if key in existing_params:
                    if existing_params[key] == str_value:
                        skipped_params.append(key)
                        continue
                    else:
                        self.warning(f"Parameter '{key}' already exists with different value. Skipping.")
                        skipped_params.append(key)
                        continue
                new_params[key] = value
            
            # Log only new parameters
            if new_params:
                mlflow.log_params(new_params)
                self.debug(f"Logged {len(new_params)} new parameters")
            
            if skipped_params:
                self.debug(f"Skipped {len(skipped_params)} duplicate parameters: {skipped_params}")
            
            return True
            
        except Exception as e:
            self.error(f"Failed to log parameters: {e}")
            return False
    
    def log_param(self, key: str, value: Any) -> bool:
        """Log a single parameter."""
        return self.log_params({key: value})
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> bool:
        """
        Log metrics (performance measurements that change over time).
        
        Metrics are numerical values that track model performance.
        Examples: loss, accuracy, RMSE, R², etc.
        
        Args:
            metrics: Dictionary of metric name -> value pairs
            step: Optional step number (epoch, iteration) for time-series plotting
            
        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            # Ensure all values are numeric and finite
            safe_metrics = {}
            for key, value in metrics.items():
                try:
                    # Convert to float and check for valid numbers
                    float_val = float(value)
                    if not (float_val != float_val):  # Check for NaN (NaN != NaN is True)
                        safe_metrics[key] = float_val
                    else:
                        logger.warning(f"Skipping NaN metric: {key}")
                except (ValueError, TypeError):
                    logger.warning(f"Skipping non-numeric metric: {key}={value}")
                    
            if safe_metrics:
                mlflow.log_metrics(safe_metrics, step=step)
                logger.debug(f"Logged {len(safe_metrics)} metrics at step {step}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            return False
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> bool:
        """Log a single metric."""
        return self.log_metrics({key: value}, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> bool:
        """
        Log an artifact (file or directory).
        
        Artifacts are files produced by your experiment: models, plots, data files, etc.
        They're stored in MLflow's artifact store and can be downloaded later.
        
        Args:
            local_path: Path to the local file/directory to log
            artifact_path: Optional path within the run's artifact directory
            
        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            if not os.path.exists(local_path):
                logger.warning(f"Artifact path does not exist: {local_path}")
                return False
                
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log artifact {local_path}: {e}")
            return False
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> bool:
        """Log all files in a directory as artifacts."""
        if not self.enabled:
            return False
            
        try:
            if not os.path.exists(local_dir):
                logger.warning(f"Artifact directory does not exist: {local_dir}")
                return False
                
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.debug(f"Logged artifacts from directory: {local_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log artifacts from {local_dir}: {e}")
            return False
    
    def log_model(self, model, name: str = None, artifact_path: str = None, 
                  input_example=None, signature=None, **kwargs) -> bool:
        """
        Log a PyTorch model with robust error handling and modern MLflow API.
        
        Args:
            model: PyTorch model to log
            name: Model name (preferred over artifact_path)
            artifact_path: Legacy path parameter (deprecated, use name instead)
            input_example: Example input for model signature inference
            signature: Model signature (auto-inferred if input_example provided)
            **kwargs: Additional arguments for mlflow.pytorch.log_model
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.active_run:
            self.error("No active MLflow run. Call start_run() first.")
            return False
            
        try:
            import mlflow.pytorch
            import torch
            
            # Use name parameter instead of deprecated artifact_path
            model_name = name or artifact_path or "model"
            
            # Auto-infer signature if input_example is provided
            if input_example is not None and signature is None:
                try:
                    from mlflow.models.signature import infer_signature
                    # For PyTorch models, we need to get model output for signature
                    model.eval()
                    with torch.no_grad():
                        if isinstance(input_example, dict):
                            # Handle dict inputs (our rainfall model case)
                            model_output = model(input_example)
                        else:
                            model_output = model(input_example)
                    signature = infer_signature(input_example, model_output.cpu().numpy())
                except Exception as sig_error:
                    self.warning(f"Could not infer model signature: {sig_error}")
                    signature = None
            
            # Log model with modern API
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,  # Still required internally
                signature=signature,
                input_example=input_example,
                **kwargs
            )
            
            self.info(f"Model logged successfully as '{model_name}'")
            if signature:
                self.info("Model signature auto-inferred from input example")
            return True
            
        except Exception as e:
            self.error(f"Failed to log model: {e}")
            return False
    
    def log_text(self, text: str, artifact_file: str) -> bool:
        """
        Log text content as an artifact file.
        
        Useful for logging model summaries, configuration files, etc.
        
        Args:
            text: Text content to log
            artifact_file: Filename for the artifact
            
        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            # Try modern MLflow API first
            try:
                mlflow.log_text(text, artifact_file)
                logger.debug(f"Logged text to: {artifact_file}")
                return True
            except AttributeError:
                # Fallback for older MLflow versions
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                    f.write(text)
                    temp_path = f.name
                    
                try:
                    mlflow.log_artifact(temp_path, artifact_path=artifact_file)
                    logger.debug(f"Logged text to: {artifact_file} (via temp file)")
                    return True
                finally:
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Failed to log text to {artifact_file}: {e}")
            return False
    
    def set_tag(self, key: str, value: str) -> bool:
        """
        Set a tag on the current run.
        
        Tags are key-value pairs for organizing and filtering runs.
        Examples: "model_type", "data_version", "experiment_phase", etc.
        
        Args:
            key: Tag name
            value: Tag value
            
        Returns:
            True if tagging succeeded, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            mlflow.set_tag(key, str(value))
            logger.debug(f"Set tag: {key}={value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set tag {key}={value}: {e}")
            return False
    
    def set_tags(self, tags: Dict[str, str]) -> bool:
        """Set multiple tags."""
        if not self.enabled:
            return False
            
        success = True
        for key, value in tags.items():
            if not self.set_tag(key, value):
                success = False
        return success
    
    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if not self.enabled or not self.active_run:
            return None
        return self.active_run.info.run_id
    
    def log_training_curves(self, history: Dict[str, List[float]], 
                          start_epoch: int = 1) -> bool:
        """
        Log training curves (metrics over epochs).
        
        This creates time-series plots in MLflow UI showing how metrics
        change during training. Essential for debugging training dynamics.
        
        Args:
            history: Dictionary with metric names as keys and lists of values
            start_epoch: Starting epoch number (usually 1)
            
        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.enabled or not history:
            return False
            
        try:
            # Determine number of epochs from the first metric
            first_key = next(iter(history.keys()))
            n_epochs = len(history[first_key])
            
            # Log each epoch's metrics
            for epoch_idx in range(n_epochs):
                epoch_metrics = {}
                for metric_name, values in history.items():
                    if epoch_idx < len(values):
                        epoch_metrics[metric_name] = values[epoch_idx]
                
                if epoch_metrics:
                    self.log_metrics(epoch_metrics, step=start_epoch + epoch_idx)
            
            logger.debug(f"Logged training curves for {n_epochs} epochs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log training curves: {e}")
            return False


def create_mlflow_logger(experiment_name: str, 
                        run_name: Optional[str] = None,
                        enabled: bool = True) -> MLflowLogger:
    """
    Factory function to create an MLflow logger.
    
    This is the main entry point for MLflow logging in your scripts.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Optional name for the specific run
        enabled: Whether to enable MLflow logging
        
    Returns:
        Configured MLflowLogger instance
    """
    return MLflowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        enabled=enabled
    )


def start_pretraining_preview_run(
    experiment_name: str,
    run_name: Optional[str],
    hyperparams: Dict[str, Any],
    training_config_preview: Dict[str, Any],
    enabled: bool = True,
) -> bool:
    """
    Start a short MLflow run to make the experiment visible early and log a minimal
    preview of configuration (e.g., epochs, folds, device) along with hyperparameters.

    This is intended for quick, non-blocking logging before the main training run.

    Args:
        experiment_name: MLflow experiment name
        run_name: Optional custom run name (e.g., "pre_training_<ts>")
        hyperparams: Dictionary of model hyperparameters to log (prefixed by caller if desired)
        training_config_preview: Dictionary of lightweight training config to log as params
        enabled: Whether to enable MLflow logging

    Returns:
        True if the logging succeeded (or MLflow disabled), False otherwise.
    """
    try:
        logger = create_mlflow_logger(
            experiment_name=experiment_name,
            run_name=run_name,
            enabled=enabled,
        )
        if not logger.enabled:
            return True

        success = True
        with logger.start_run():
            # Log HPs and basic config
            success = log_hyperparameters(logger, hyperparams, prefix="hp") and success
            success = logger.log_params(training_config_preview) and success
            success = logger.set_tags({
                "phase": "pre_training",
                "framework": "pytorch",
            }) and success
        return success
    except Exception as e:
        # Do not crash callers; return False to indicate preview logging failed
        logging.getLogger(__name__).error(f"Pre-training MLflow preview failed: {e}")
        return False


# Convenience functions for common logging patterns
def log_hyperparameters(logger: MLflowLogger, hyperparams: Dict[str, Any], 
                       prefix: str = "") -> bool:
    """
    Log hyperparameters with optional prefix.
    
    Args:
        logger: MLflow logger instance
        hyperparams: Dictionary of hyperparameters
        prefix: Optional prefix for parameter names
        
    Returns:
        True if logging succeeded
    """
    if prefix:
        prefixed_params = {f"{prefix}_{k}": v for k, v in hyperparams.items()}
    else:
        prefixed_params = hyperparams
        
    return logger.log_params(prefixed_params)


def log_model_summary(logger: MLflowLogger, model, 
                     filename: str = "model_summary.txt") -> bool:
    """
    Log a text summary of the model architecture.
    
    Args:
        logger: MLflow logger instance
        model: PyTorch model
        filename: Name for the summary file
        
    Returns:
        True if logging succeeded
    """
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Create summary
        summary_lines = [
            "Model Architecture Summary",
            "=" * 50,
            "",
            str(model),
            "",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Non-trainable parameters: {total_params - trainable_params:,}",
        ]
        
        summary_text = "\n".join(summary_lines)
        return logger.log_text(summary_text, filename)
        
    except Exception as e:
        logger.error(f"Failed to create model summary: {e}")
        return False


def log_evaluation_results(logger: MLflowLogger, metrics: Dict[str, float],
                          prefix: str = "test") -> bool:
    """
    Log evaluation metrics with consistent naming.
    
    Args:
        logger: MLflow logger instance
        metrics: Dictionary of evaluation metrics
        prefix: Prefix for metric names (e.g., "test", "val")
        
    Returns:
        True if logging succeeded
    """
    prefixed_metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return logger.log_metrics(prefixed_metrics)
