import mlflow
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    r2_score,
)


class ModelEvaluator:
    def __init__(self, predicting_gameweek: int):
        self.predicting_gameweek = predicting_gameweek

    @staticmethod
    def _compute_metrics(y_true, y_pred, prefix: str):
        # Optional: guard against MAPE explosion
        y_true_safe = np.clip(y_true, 1e-6, None)

        return {
            f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
            f"{prefix}_r2": r2_score(y_true, y_pred),
            f"{prefix}_mape": mean_absolute_percentage_error(y_true_safe, y_pred),
            f"{prefix}_rmse": root_mean_squared_error(y_true, y_pred),
        }

    def get_model_metrics(
        self,
        y_train,
        y_test,
        y_pred_train,
        y_pred_test,
        mlflow_run_id = None
    ):
        # Compute metrics
        train_metrics = self._compute_metrics(y_train, y_pred_train, "train")
        test_metrics = self._compute_metrics(y_test, y_pred_test, "test")

        all_metrics = {**train_metrics, **test_metrics}

        # Log to MLflow
        if mlflow_run_id:
            mlflow.start_run(mlflow_run_id)
            mlflow.log_metrics(all_metrics)

            # Also log gameweek as a param (important for filtering later)
            mlflow.log_param("predicting_gameweek", self.predicting_gameweek)

        return all_metrics