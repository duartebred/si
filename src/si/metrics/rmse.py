import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between actual and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Array with the actual target values.
    y_pred : np.ndarray
        Array with the predicted target values.

    Returns
    -------
    float
        The computed RMSE value, indicating the average prediction error.
    """
    errors = y_true - y_pred
    mean_squared_error = np.mean(errors ** 2)
    root_mean_squared_error = np.sqrt(mean_squared_error)
    return root_mean_squared_error
