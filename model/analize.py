from sklearn.cluster import DBSCAN
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification


class Analize():
    def __init__(self):
        self.model = DBSCAN(eps=1, min_samples=1)

    def fit(self, X_trained):
        self.model.fit(X_trained)
        return self.model.labels_
