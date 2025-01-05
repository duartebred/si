import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class LassoRegression(Model):
    """
    Lasso Regression applies L1 regularization to shrink coefficients, reducing less significant features to zero. 
    This technique helps improve model simplicity and prevents overfitting.
    """

    def __init__(self, l1_penalty: float = 1.0, max_iter: int = 1000, patience: int = 5, scale: bool = True, **kwargs):
        """
        Initialize the Lasso Regression model with regularization and optimization parameters.

        Parameters
        ----------
        l1_penalty : float
            Regularization strength (L1 penalty).
        max_iter : int
            Maximum number of iterations for optimization.
        patience : int
            Number of iterations to wait without improvement before stopping early.
        scale : bool
            Whether to normalize the features.
        """
        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # Model attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> "LassoRegression":
        """
        Train the model using coordinate descent and update coefficients based on L1 regularization.

        Parameters
        ----------
        dataset : Dataset
            Training dataset containing features and target values.

        Returns
        -------
        LassoRegression
            Trained model instance.
        """
        # Feature scaling
        if self.scale:
            self.mean = np.mean(dataset.X, axis=0)
            self.std = np.std(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = dataset.shape()
        self.theta = np.zeros(n)
        self.theta_zero = 0

        early_stopping = 0
        for iteration in range(self.max_iter):
            if early_stopping >= self.patience:
                break

            # Update each coefficient
            for feature in range(n):
                residual = np.dot(X[:, feature], (dataset.y - (np.dot(X, self.theta) + self.theta_zero)))
                self.theta[feature] = self._soft_threshold(residual, self.l1_penalty) / np.sum(X[:, feature] ** 2)

            # Update intercept
            self.theta_zero = np.mean(dataset.y - np.dot(X, self.theta))

            # Compute and track cost
            self.cost_history[iteration] = self._compute_cost(dataset)
            if iteration > 0 and self.cost_history[iteration] > self.cost_history[iteration - 1]:
                early_stopping += 1
            else:
                early_stopping = 0

        return self

    def _compute_cost(self, dataset: Dataset) -> float:
        """
        Calculate the cost function for Lasso Regression.

        Parameters
        ----------
        dataset : Dataset
            The dataset used to evaluate the model.

        Returns
        -------
        float
            Cost value that includes L1 regularization.
        """
        y_pred = self._predict(dataset)
        return np.mean((dataset.y - y_pred) ** 2) / 2 + self.l1_penalty * np.sum(np.abs(self.theta))

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the target values for the given dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing input features.

        Returns
        -------
        np.ndarray
            Predicted values for the dataset.
        """
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Evaluate the model's performance using Mean Squared Error (MSE).

        Parameters
        ----------
        dataset : Dataset
            Dataset containing true target values.
        predictions : np.ndarray
            Model predictions.

        Returns
        -------
        float
            MSE value indicating the error.
        """
        return mse(dataset.y, predictions)

    @staticmethod
    def _soft_threshold(residual: float, l1_penalty: float) -> float:
        """
        Apply the soft-thresholding function to compute the L1-regularized coefficient.

        Parameters
        ----------
        residual : float
            Residual value for a specific feature.
        l1_penalty : float
            Regularization strength.

        Returns
        -------
        float
            Regularized coefficient value.
        """
        if residual > l1_penalty:
            return residual - l1_penalty
        elif residual < -l1_penalty:
            return residual + l1_penalty
        else:
            return 0.0
