import numpy as np
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive metrics for angle prediction."""
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    correlation = float(np.corrcoef(predictions, targets)[0, 1])
    r2 = correlation ** 2

    return {"mae": float(mae), "mse": float(mse), "rmse": float(rmse), "r2": float(r2)}
