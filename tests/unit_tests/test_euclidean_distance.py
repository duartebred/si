from unittest import TestCase
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv
from si.statistics.euclidean_distance import euclidean_distance



class TestSelectKBest(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, sep=",", features=True, label=True)