import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset

class SelectKBest(Transformer):
    """
    """
    def __init__(self, score_func : callable, k : int, **kwargs):
        """
        """
        super().__init__(**kwargs)
        # parameters
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self, dataset : Dataset) -> 'SelectKBest':
        """
        """
        self.F, self.p = self.score_func(dataset)
        return self
    
    def _transform(self, dataset : Dataset) -> Dataset:
        """
        """
        idx = np.argsort(self.F)
        mask = idx[-self.k:]
        new_X = dataset.X[:, mask]
        new_features = dataset.features[mask]

        return Dataset(X=new_X, features=new_features, y=dataset.x, label=dataset.label)