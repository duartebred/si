from unittest import TestCase

from datasets import DATASETS_PATH

import os
import numpy as np

from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification


class TestSelectPercentile(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        selector = SelectPercentile(percentile=50, score_func=f_classification)
        selector._fit(self.dataset)
        
        self.assertIsNotNone(selector.F)
        self.assertIsNotNone(selector.p)
        self.assertEqual(len(selector.F), self.dataset.X.shape[1])
        self.assertEqual(len(selector.p), self.dataset.X.shape[1])


    def test_transform(self):
        selector = SelectPercentile(percentile=50, score_func=f_classification)
        selector._fit(self.dataset)
        dataset = selector._transform(self.dataset)
        
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.X.shape[1], 2)
        self.assertEqual(dataset.X.shape[0], self.dataset.X.shape[0])
        self.assertEqual(dataset.y.shape[0], self.dataset.y.shape[0])
        self.assertEqual(dataset.y.shape[1], 1)
        self.assertEqual(dataset.features, ['sepal_length', 'sepal_width'])
        self.assertEqual(dataset.label, 'species')
        self.assertTrue(np.all(dataset.X >= 0))
        self.assertTrue(np.all(dataset.y >= 0))
        self.assertTrue(np.all(dataset.X <= 1))
        self.assertTrue(np.all(dataset.y <= 1))
