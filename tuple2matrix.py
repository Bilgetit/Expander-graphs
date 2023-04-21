"""Convert tuple to matrix. This might be a waste, as you are converting
back to a matrix later. Might want to store in a dictionary directly."""

import numpy as np


def tuple2matrix(tup: tuple, n: int = 3) -> np.ndarray:
    """Convert tuple to matrix."""
    return np.array(tup).reshape(n, n)
