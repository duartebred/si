from typing import Callable, Union
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    A K-Nearest Neighbors (KNN) regressor that predicts target values
    by averaging the values of the k-nearest samples in the dataset.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN regressor with the number of neighbors (k) and distance function.

        Parameters
        ----------
        k : int
            Number of nearest neighbors to consider for prediction.
        distance : Callable
            Function to compute the distance between samples.
        """
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> "KNNRegressor":
        """
        Store the training dataset for future predictions.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        KNNRegressor
            The fitted model instance.
        """
        self.dataset = dataset
        return self

    def _get_average_value(self, sample: np.ndarray) -> Union[int, float]:
        """
        Calculate the average target value of the k-nearest neighbors.

        Parameters
        ----------
        sample : np.ndarray
            The sample to predict the target value for.

        Returns
        -------
        Union[int, float]
            The predicted target value based on neighbors' average.
        """
        # Compute distances from the sample to all training samples
        distances = self.distance(sample, self.dataset.X)
        
        # Identify the indices of the k closest neighbors
        nearest_indices = np.argsort(distances)[:self.k]
        
        # Retrieve their target values
        neighbor_values = self.dataset.y[nearest_indices]
        
        # Return the mean of the target values
        return np.mean(neighbor_values)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict target values for a dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict target values for.

        Returns
        -------
        np.ndarray
            Array of predicted target values.
        """
        return np.apply_along_axis(self._get_average_value, axis=1, arr=dataset.X)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Evaluate the model using Root Mean Squared Error (RMSE).

        Parameters
        ----------
        dataset : Dataset
            The dataset with true target values.
        predictions : np.ndarray
            Predicted target values.

        Returns
        -------
        float
            RMSE value indicating the prediction error.
        """
        return rmse(dataset.y, predictions)
