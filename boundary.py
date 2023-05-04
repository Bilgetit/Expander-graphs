from tuple_graph import get_graph, get_edges
from tuple2matrix import tuple2matrix
from typing import Optional
import numpy as np


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
