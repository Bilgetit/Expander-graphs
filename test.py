from numba import jit
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
ra = np.ravel(A)
print(tuple(ra))


@jit(nopython=True)
def totuple(a):
    return tuple(map(tuple, a))


print(totuple(ra))
