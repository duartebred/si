import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
import itertools
from si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(model: Model, dataset: Dataset, hyperparameter_grid: dict,
                         cv: int, n_iter: int, scoring: callable = None) -> dict:
    """
    Perform a randomized search for hyperparameter optimization using cross-validation.

    Parameters
    ----------
    model : Model
        The machine learning model to tune.
    dataset : Dataset
        The dataset to use for cross-validation.
    hyperparameter_grid : dict
        Dictionary containing hyperparameters and their possible values.
    cv : int
        Number of folds for cross-validation.
    n_iter : int
        Number of random hyperparameter combinations to evaluate.
    scoring : callable, optional
        A function to evaluate the model's performance. If None, the model's default score is used.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'scores': List of average scores for each hyperparameter combination.
        - 'hyperparameters': List of hyperparameter configurations evaluated.
        - 'best_hyperparameters': The best hyperparameter configuration.
        - 'best_score': The highest score achieved.
    """
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {type(model).__name__} does not have parameter '{parameter}'.")

    combinations = random_combinations(hyperparameter_grid, n_iter)

    results = {'scores': [], 'hyperparameters': []}

    for combination in combinations:
        parameters = {param: value for param, value in zip(hyperparameter_grid.keys(), combination)}
        for param, value in parameters.items():
            setattr(model, param, value)

        scores = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        results['scores'].append(np.mean(scores))
        results['hyperparameters'].append(parameters)

    best_idx = np.argmax(results['scores'])
    results['best_hyperparameters'] = results['hyperparameters'][best_idx]
    results['best_score'] = results['scores'][best_idx]

    return results


def random_combinations(hyperparameter_grid: dict, n_iter: int) -> list:
    """
    Randomly select a specified number of hyperparameter combinations.

    Parameters
    ----------
    hyperparameter_grid : dict
        Dictionary of hyperparameters and their possible values.
    n_iter : int
        Number of random combinations to generate.

    Returns
    -------
    list
        A list of randomly selected hyperparameter combinations.
    """
    all_combinations = list(itertools.product(*hyperparameter_grid.values()))

    random_indices = np.random.choice(len(all_combinations), size=n_iter, replace=False)
    return [all_combinations[i] for i in random_indices]
