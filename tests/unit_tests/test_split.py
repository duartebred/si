from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv

from si.model_selection.split import train_test_split
from si.model_selection.split import stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

    def test_stratified_train_test_split(self):
        """
        Test the stratified_train_test_split function to ensure the data is split 
        into training and testing sets while maintaining the class distribution.

        This test verifies:
        - The size of the training and testing sets.
        - The proportions of class labels in the training and testing sets match the original dataset.
        """

        # Perform stratified train-test split
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)

        # Calculate class proportions in the original dataset
        _, labels_counts = np.unique(self.dataset.y, return_counts=True)
        total_labels = np.sum(labels_counts)
        original_proportion = labels_counts / total_labels * 100

        # Calculate class proportions in the training set
        _, labels_counts_train = np.unique(train.y, return_counts=True)
        total_labels_train = np.sum(labels_counts_train)
        train_proportion = labels_counts_train / total_labels_train * 100

        # Calculate class proportions in the testing set
        _, labels_counts_test = np.unique(test.y, return_counts=True)
        total_labels_test = np.sum(labels_counts_test)
        test_proportion = labels_counts_test / total_labels_test * 100

        # Expected size of the test set
        expected_test_size = int(self.dataset.shape()[0] * 0.2)

        # Validate the sizes of the train and test sets
        self.assertEqual(test.shape()[0], expected_test_size, "Test set size is incorrect.")
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - expected_test_size, "Train set size is incorrect.")

        # Validate class proportions in the train and test sets
        self.assertTrue(np.allclose(original_proportion, train_proportion, rtol=1e-03), 
                        "Class proportions in the training set do not match the original dataset.")
        self.assertTrue(np.allclose(original_proportion, test_proportion, rtol=1e-03), 
                        "Class proportions in the testing set do not match the original dataset.")
