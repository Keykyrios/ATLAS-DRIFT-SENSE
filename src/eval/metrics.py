import numpy as np

def euclidean_error(x_pred: float, y_pred: float, x_true: float, y_true: float) -> float:
    return np.sqrt((x_pred - x_true)**2 + (y_pred - y_true)**2)

def pass_rate_at_threshold(errors: list, threshold: float) -> float:
    if not errors:
        return 0.0
    passed = [e for e in errors if e <= threshold]
    return len(passed) / len(errors)
