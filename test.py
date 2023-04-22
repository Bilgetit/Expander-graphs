from numba import jit
import pytest
import numpy as np
from tuple_graph import get_graph
from size import degree


# A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# ra = np.ravel(A)
# print(tuple(ra))


# @jit(nopython=True)
# def totuple(a):
#     return tuple(map(tuple, a))


# print(totuple(ra))


primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
@pytest.mark.parametrize(
        "n, p",
        [
            (2, p) for p in primes
        ]
)
def test_size(n, p):
    """test that the size of the graph is correct"""
    graph = get_graph(n, p)
    assert len(graph) == degree(n, p)

if __name__ == "__main__":
    pass
