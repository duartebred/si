from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
import os
from si.decomposition.pca import PCA

from unittest import TestCase
import numpy as np
import os
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA
from datasets import DATASETS_PATH

class TestPCA(TestCase):
    """
    Unit tests for the PCA class, verifying its behavior in fitting and transforming data.
    """

    def setUp(self):
        """
        Set up the test environment by loading the dataset and initializing PCA with a specified number of components.
        """
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.n_components = 2
        self.pca = PCA(n_components=self.n_components)

    def test_fit(self):
        """
        Test the fitting process of PCA to ensure the mean, components, and explained variance are computed correctly.
        """
        # Fit PCA to the dataset
        self.pca.fit(self.dataset)

        # Verify the computed mean matches the dataset's feature-wise mean
        self.assertTrue(np.allclose(self.pca.mean, np.mean(self.dataset.X, axis=0)))

        # Check the shape of the principal components matrix
        self.assertEqual(self.pca.components.shape, (self.dataset.X.shape[1], self.n_components))

        # Ensure the number of explained variance entries matches the number of components
        self.assertEqual(len(self.pca.explained_variance), self.n_components)

        # Verify the sum of explained variance is approximately 1.0
        self.assertAlmostEqual(np.sum(self.pca.explained_variance), 1.0, delta=0.05)

    def test_transform(self):
        """
        Test the transformation process of PCA to ensure the data is correctly reduced while retaining variance.
        """
        # Fit PCA and transform the dataset
        self.pca.fit(self.dataset)
        X_reduced = self.pca.transform(self.dataset)

        # Check that the reduced dataset has the correct number of samples
        self.assertEqual(X_reduced.shape[0], self.dataset.X.shape[0])

        # Verify that the total variance is approximately preserved
        original_variance = np.var(self.dataset.X, axis=0).sum()
        reduced_variance = np.var(X_reduced, axis=0).sum()
        self.assertAlmostEqual(original_variance, reduced_variance, delta=0.2)
