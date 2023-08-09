""" Torch-compatible class for scaling and inverse_scaling """
import torch
from sklearn.preprocessing import MinMaxScaler 

class MinMaxScalerTorch(MinMaxScaler):
    def fit(self, X, y=None):
        return super().fit(X, y)
    def transform(self, X):
        return torch.Tensor(super().transform(X))
    def fit_transform(self, X, y=None, **fit_params):
        return torch.Tensor(super().fit_transform(X, y, **fit_params))