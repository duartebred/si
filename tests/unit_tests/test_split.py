from unittest import TestCase
from datasets import DATASETS_PATH

import os
from si.feature_selection.select_k_best import SelectKBest
from si.io.csv_file import read_csv

from si.statistics.f_classification import f_classification

class TestSelectKBest(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
