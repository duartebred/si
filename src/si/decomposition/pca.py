import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA(Transformer):
    """
    PCA (Principal Component Analysis) is a linear algebra technique
    to reduce the dimensionality of a data set.
    """

    def __init__(self, n_components, **kwargs):
        """
        Initializes the PCA with the desired number of principal components.

        Parameters
        ----------
        n_components : int
            Number of principal components to be calculated.
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.is_fitted = False

    def _fit(self, dataset: Dataset) -> "PCA":
        """
        Fits the PCA to the data set, calculating the principal components.

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing the input data.

        Return
        -------
        self : PCA
            Adjusted instance of the PCA class.

        Raises
        -----
        ValueError
            If n_components is greater than the number of variables in the dataset.
        """
        if self.n_components <= 0 or self.n_components > dataset.shape()[1]:
            raise ValueError("The number of components must be between 1 and the number of variables in the data set.")

        # Centralizar os dados
        self.mean = dataset.get_mean()
        X_centered = dataset.X - self.mean

        # Calcular a matriz de covariância e realizar a decomposição em autovalores e autovetores
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Selecionar os componentes principais
        sorted_indices = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.components = eigenvectors[:, sorted_indices].T
        self.explained_variance = eigenvalues[sorted_indices] / np.sum(eigenvalues)

        self.is_fitted = True
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Applies the PCA transformation to the data set.

        Parameters
        ----------
        dataset : Dataset
            Data set to be transformed.

        Return
        -------
        Dataset
            Reduced data set with the new main features.
        """
        if not self.is_fitted:
            raise ValueError("The PCA needs to be adjusted before it can be used to transform the data.")

        # Centralizar os dados e calcular a projeção nos componentes principais
        X_centered = dataset.X - self.mean
        X_reduced = np.dot(X_centered, self.components.T)

        return Dataset(
            X=X_reduced,
            y=dataset.y,
            features=[f"PC{i + 1}" for i in range(self.n_components)],
            label=dataset.label
        )

    def get_covariance(self) -> np.ndarray:
        """
        Returns the covariance matrix of the centralized data.

        Return
        -------
        np.ndarray
            Data covariance matrix.

        Raises
        -----
        ValueError
            If the PCA has not yet been adjusted.
        """
        if not self.is_fitted:
            raise ValueError("The PCA has not yet been fitted to the data.")
        return np.cov(self.mean, rowvar=False)
