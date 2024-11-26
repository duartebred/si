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