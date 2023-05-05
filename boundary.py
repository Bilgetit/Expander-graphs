from tuple_graph import get_graph, get_edges
from tuple2matrix import tuple2matrix
from multiprocessing import Pool
from typing import Optional
import numpy as np
import random


def get_random(subset: set[tuple], n: int, p: int, size: int) -> None:
    matrix = random.choice(list(subset))
    matrix = tuple2matrix(matrix, n)
    subset2 = get_graph(n, p, start_matrix=matrix, stop=size)

    return subset2
    for tup in subset2:
        subset.add(tup)


def find_boundary(
    n: int = 3,
    p: int = 5,
    starting_matrix: Optional[np.ndarray] = None,
    subset: Optional[set[tuple]] = None,
    size: Optional[int] = None,
) -> set[tuple]:
    """Find boundary of set. Return boundary as set of tuples, along with its size."""
    if not subset:
        subset = get_graph(n, p, start_matrix=starting_matrix, stop=size)

    # with Pool() as pool:
    #     subset2 = pool.starmap(get_random, [(subset, n, p, size)]*100 )
    # # for sub in subset2:
    # #     subset = subset.union(sub)
    # for sub in subset2:
    #     for tup in sub:
    #         subset.add(tup)

    # for _ in range(10):
    #     matrix = random.choice(list(subset))
    #     matrix = tuple2matrix(matrix, n)
    #     subset2 = get_graph(n, p, start_matrix=matrix, stop=size)

    #     for tup in subset2:
    #         subset.add(tup)
    #     subset = subset.union(subset2)
    # print(len(subset))

    edges = get_edges(n, p)
    if not size:
        size = len(subset)

    boundary: set[tuple] = set()

    for Xtup in subset:
        X = tuple2matrix(Xtup, n)
        Xes = np.matmul(X, edges)
        Xes %= p
        for Xe in Xes:
            Xe_tup = tuple(np.ravel(Xe))
            if Xe_tup not in subset:
                boundary.add(Xe_tup)
    return boundary, subset
