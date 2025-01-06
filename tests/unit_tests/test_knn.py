from unittest import TestCase

import numpy as np


from datasets import DATASETS_PATH
from si.metrics.rmse import rmse
from si.models.knn_regressor import KNNRegressor
import os
from si.io.csv_file import read_csv
from si.models.knn_classifier import KNNClassifier
from si.model_selection.split import train_test_split

class TestKNN(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNClassifier(k=3)

        knn.fit(self.dataset)

        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        knn = KNNClassifier(k=1)

        train_dataset, test_dataset = train_test_split(self.dataset)

        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])
        self.assertTrue(np.all(predictions == test_dataset.y))

    def test_score(self):
        knn = KNNClassifier(k=3)

        train_dataset, test_dataset = train_test_split(self.dataset)

        knn.fit(train_dataset)
        score = knn.score(test_dataset)
        self.assertEqual(score, 1)

class TestKNNRegressor(TestCase):
    """
    Unit tests for the KNNRegressor class, verifying its ability to fit, predict, and score datasets.
    """

    def setUp(self):
        """
        Prepare the testing environment by loading the dataset.
        """
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        """
        Test the fitting process of the KNNRegressor to ensure 
        it correctly stores the training data.
        """
        knn = KNNRegressor(k=3)
        knn.fit(self.dataset)

        # Verify that the training data is correctly stored in the model
        self.assertTrue(np.all(self.dataset.features == knn.dataset.features), "Features were not stored correctly.")
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y), "Labels were not stored correctly.")

    def test_predict(self):
        """
        Test the prediction functionality of the KNNRegressor 
        to ensure it returns predictions of the correct shape.
        """
        knn = KNNClassifier(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset)

        # Fit the model and make predictions
        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)

        # Validate the number of predictions matches the number of test samples
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0], "Prediction count does not match test samples.")

    def test_score(self):
        """
        Test the scoring functionality of the KNNRegressor by comparing 
        the calculated RMSE to the expected RMSE.
        """
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset)

        # Fit the model, make predictions, and calculate the score
        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        score = knn.score(test_dataset)

        # Verify that the computed RMSE matches the expected RMSE
        expected_score = rmse(test_dataset.y, predictions)
        self.assertAlmostEqual(score, expected_score, places=5, msg="Computed RMSE does not match the expected RMSE.")