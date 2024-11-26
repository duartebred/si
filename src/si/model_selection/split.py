<<<<<<< HEAD
from si.data.dataset import Dataset

import numpy as np

def train_test_split(dataset: Dataset, test_size: float, random_state = 123):
    """
    Splits the dataset into training and testing datasets.

    Arguments:
    - dataset: the Dataset object to split into training and testing data.
    - test_size: the proportion of the dataset to be used as test data (e.g., 0.2 for 20%).
    - random_state: seed for generating permutations.

    Returns:
    - A tuple containing the train and test datasets.
    """
    
    np.random.seed(random_state)
    
    
    permutations = np.random.permutation(dataset.X.shape()[0])
    
    test_sample_size = int(dataset.shape()[0] * test_size)
    
    test_idx = permutations[:test_sample_size]
    train_idx = permutations[test_sample_size:]
    
    train_dataset = Dataset(X=dataset.X[train_idx :], y=dataset.y[train_idx] )
    test_dataset = 
    
    return train_dataset, test_dataset
=======
from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test
>>>>>>> fbc169728b98e367666356bfbcb2f3ef9365e45f
