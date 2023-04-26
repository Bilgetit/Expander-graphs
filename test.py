from numba import jit
import pytest
import random
import numpy as np
from tuple_graph import get_graph
from boundary import find_boundary
from size import degree
from tuple2matrix import tuple2matrix



# A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# ra = np.ravel(A)
# print(tuple(ra))


# @jit(nopython=True)
# def totuple(a):
#     return tuple(map(tuple, a))


# print(totuple(ra))

primes = [2, 3, 5]
# primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
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

@pytest.mark.parametrize(
        "stop", 
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1_000 #, 10_000, 100_000, 1_000_000, 119
        ]
)
def test_stop(stop):
    """test that the stop parameter works"""
    graph = get_graph(2, 13, stop=stop)
    assert len(graph) == stop

@pytest.mark.parametrize(
        "stop",
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1_000 #, 10_000, 100_000, 1_000_000
        ]
)
def test_boundary_size(stop):
    """test that the boundary size is correct"""
    n=2
    p=101
    graph = get_graph(n, p, stop = 1000)
    start_matrix = random.choice(tuple(graph))
    start_matrix = tuple2matrix(start_matrix, n)
    boundary, subset, size = find_boundary(n, p, size=stop, starting_matrix=start_matrix)
    assert len(subset) == size


# @pytest.mark.parametrize(
#         "stop",
#         [
#             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1_000, 10_000, 100_000, 1_000_000
#         ]
# )
# def test_boundary_size_2(stop):
#     """test that the boundary size is correct"""
#     graph = get_graph(2, 17, stop=stop)
#     boundary, size = find_boundary(2, 17, size=stop, subset=graph)
#     assert len(boundary) == size


if __name__ == "__main__":
    pass






"""
"n, p, stop",
        [
            (2, p, stop) for p in primes for stop in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
        ]
"""