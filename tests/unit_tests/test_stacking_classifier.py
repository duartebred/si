from unittest import TestCase
import os
from si.ensemble.stacking_classifier import StackingClassifier
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from datasets import DATASETS_PATH

class TestStackingClassifier(TestCase):
    """
    Unit tests for the StackingClassifier class, verifying its fit, predict, and score methods.
    """

    def setUp(self):
        """
        Prepare the dataset and split it into training and testing sets for use in the tests.
        """
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_fit(self):
        """
        Test the fit method to ensure that the base models and the final model are properly configured.
        """
        # Initialize base models and final model
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()
        knn_final = KNNClassifier()

        # Create the stacking classifier
        stacking_classifier = StackingClassifier(models=[knn, logistic_regression, decision_tree], final_model=knn_final)

        # Validate default hyperparameters for one of the base models (decision tree)
        self.assertEqual(stacking_classifier.models[2].min_sample_split, 2, "DecisionTreeClassifier min_sample_split is incorrect.")
        self.assertEqual(stacking_classifier.models[2].max_depth, 10, "DecisionTreeClassifier max_depth is incorrect.")

    def test_predict(self):
        """
        Test the predict method to ensure it returns predictions with the correct shape.
        """
        # Initialize base models and final model
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()
        knn_final = KNNClassifier()

        # Create and train the stacking classifier
        stacking_classifier = StackingClassifier(models=[knn, logistic_regression, decision_tree], final_model=knn_final)
        stacking_classifier.fit(self.train_dataset)

        # Generate predictions for the test dataset
        predictions = stacking_classifier.predict(self.test_dataset)

        # Validate that the number of predictions matches the number of test samples
        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0], 
                         "The number of predictions does not match the number of test samples.")

    def test_score(self):
        """
        Test the score method to verify that the calculated accuracy matches the expected value.
        """
        # Initialize base models and final model
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()
        knn_final = KNNClassifier()

        # Create and train the stacking classifier
        stacking_classifier = StackingClassifier(models=[knn, logistic_regression, decision_tree], final_model=knn_final)
        stacking_classifier.fit(self.train_dataset)

        # Calculate the accuracy of the stacking classifier
        accuracy_ = stacking_classifier.score(self.test_dataset)

        # Validate the accuracy score
        self.assertEqual(round(accuracy_, 2), 0.95, 
                         "The accuracy of the StackingClassifier does not match the expected value (0.95).")
