from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
import numpy as np

class DumbFeaturizer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [[2] for _ in X]


s = np.random.uniform(-1,0,1000)
pipe = make_pipeline(DumbFeaturizer())
output = pipe.transform(s)
print(output[0])