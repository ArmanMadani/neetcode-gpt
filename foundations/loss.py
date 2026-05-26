import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 10 ** -7
        y_pred_eps = y_pred + epsilon
        n = -1 / len(y_true)
        l = np.sum(
            (y_true * np.log(y_pred_eps)) + 
            ((1 - y_true) * np.log(1 - y_pred_eps))
        )
        return np.round(n * l, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = -1 / len(y_true)
        loss = 0
        for i, j in zip(y_true, y_pred): 
            loss += np.sum(i * np.log(j + (10 ** -7)))
        return np.round(n * loss, 4)
