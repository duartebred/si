import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.base.model import Model
from si.statistics.euclidean_distance import euclidean_distance 


class Kmeans (Transformer, Model):
    def __init__(self, k, max_iter : int = 300, distance : callable = euclidean_distance, **kwargs):

        self.k = k
        self.max_iter = max_iter
        self.distance = distance 

        self.centroids = None
        self.labels = None


    def _init_centroids(self, dataset: Dataset):

        seeds = np.random.permutation(dataset.shape()[0])[:self.k]
        self.centroids = dataset.X[seeds , :]


    def _get_closest_centroid(self, sample : np.ndarray):

        distance_ = self.distance(sample, self.centroids)    
        return np.argmin(distance_)
    

    def _fit(self, dataset: Dataset) -> 'Kmeans':

        self._init_centroids (dataset : Dataset)

        new_labels = np.apply_along_axis(self._get_closest_centroid, axis = 1, arr = dataset.X)

        while not convergence and j < self.max_iter:

            new_centroids = []
            for i in range(self.k):
                new_centroids = np.mean(dataset.X[new_labels == i])
                new_centroids.append(new_centroids)

            self.centroids = np.array(new_centroids)

            #convergence = not np.any(new_labels != self.labels)
            convergence = np.all(new_labels == self.labels)

            j+=1

            self.labels = new_labels

        return self
      