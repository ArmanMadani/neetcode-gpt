import numpy as np
from numpy.typing import NDArray
from math import e


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        func = lambda x: 1 / (1 + e ** (-x))
        vfunc = np.vectorize(func)
        return np.round(vfunc(z), 5)
        

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        return np.maximum(0, z)
