from numba import jit
import pytest
import random
import numpy as np
from tuple_graph import get_graph
from boundary import find_boundary
from random_matrix import get_matrix
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
primes = [5, 7, 11, 13, 17, 19, 23]
@pytest.mark.parametrize("n, p", [(2, p) for p in primes])
def test_size(n, p):
    """test that the size of the graph is correct"""
    graph = get_graph(n, p)
    assert len(graph) == degree(n, p)


@pytest.mark.parametrize(
    "stop",
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1_000],  # , 10_000, 100_000, 1_000_000, 119
)
def test_stop(stop):
    """test that the stop parameter works"""
    graph = get_graph(2, 13, stop=stop)
    assert len(graph) == stop


# @pytest.mark.parametrize(
#     "stop", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1_000]  # , 10_000, 100_000, 1_000_000
# )
# def test_boundary_size(stop):
#     """test that the boundary size is correct"""
#     n = 2
#     p = 101
#     graph = get_graph(n, p, stop=1000)
#     start_matrix = random.choice(tuple(graph))
#     start_matrix = tuple2matrix(start_matrix, n)
#     boundary, subset, size = find_boundary(
#         n, p, size=stop, starting_matrix=start_matrix
#     )
#     assert len(subset) == size


@pytest.mark.parametrize(
    "n, p",
    [(n, p) for n in range(2, 4) for p in primes],
)
def test_start_matrix(n, p):
    """test that the starting matrices are different. This test is prone to false negatives"""
    start_matrix1 = get_matrix(n, p, stop=100)
    start_matrix2 = get_matrix(n, p, stop=100, starting_matrix=start_matrix1)
    m1_tup = tuple(np.ravel(start_matrix1))
    m2_tup = tuple(np.ravel(start_matrix2))
    assert m1_tup != m2_tup

@pytest.mark.parametrize(
    "n, p",
    [(n, p) for n in range(2, 4) for p in primes],
)
def test_graph(n, p):
    """test that the graphs are different when the starting matrix is different"""
    start_matrix1 = get_matrix(n, p, stop=100)
    start_matrix2 = get_matrix(n, p, stop=100, starting_matrix=start_matrix1)
    graph1 = get_graph(n, p, stop=10, start_matrix = start_matrix1)
    graph2 = get_graph(n, p, stop=10, start_matrix = start_matrix2)
    assert graph1 != graph2

@pytest.mark.parametrize(
    "n, p",
    [(n, p) for n in range(2, 4) for p in primes],
)
def test_boundary(n, p):
    """test that the boundary is different when the starting matrix is different"""
    start_matrix1 = get_matrix(n, p, stop=100)
    start_matrix2 = get_matrix(n, p, stop=100, starting_matrix=start_matrix1)
    boundary1 = find_boundary(n, p, size=10, starting_matrix=start_matrix1)
    boundary2 = find_boundary(n, p, size=10, starting_matrix=start_matrix2)
    assert boundary1 != boundary2




