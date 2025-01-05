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


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    It divides the data set into training and test sets, maintaining the proportion of classes.

    Parameters
    ----------
    dataset : Dataset
        The data set to be split.
    test_size : float, optional
        Proportion of the test set to the total (default is 0.2).
    random_state : int, optional
        Seed for the random number generator (default is 42).

    Returns
    -------
    Tuple[Dataset, Dataset]
        A pair containing the training and test sets.

    Raises
    -----
    ValueError
        If `test_size` is not in the range between 0 and 1.
    """
    if not (0 < test_size < 1):
        raise ValueError("The size of the test set must be a value between 0 and 1.")

    # Configurar o gerador de números aleatórios
    np.random.seed(random_state)

    # Identificar as classes únicas e suas contagens
    labels, counts = np.unique(dataset.y, return_counts=True)

    # Inicializar listas para índices de treino e teste
    train_indices, test_indices = [], []

    # Dividir os índices para cada classe
    for label, count in zip(labels, counts):
        label_indices = np.where(dataset.y == label)[0]
        np.random.shuffle(label_indices)
        
        split_index = int(count * (1 - test_size))
        train_indices.extend(label_indices[:split_index])
        test_indices.extend(label_indices[split_index:])

    # Criar os conjuntos de treino e teste
    train_dataset = Dataset(
        X=dataset.X[train_indices],
        y=dataset.y[train_indices],
        features=dataset.features,
        label=dataset.label
    )
    test_dataset = Dataset(
        X=dataset.X[test_indices],
        y=dataset.y[test_indices],
        features=dataset.features,
        label=dataset.label
    )

    return train_dataset, test_dataset
