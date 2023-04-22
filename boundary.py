from tuple_graph import get_graph, get_edges
from tuple2matrix import tuple2matrix
from typing import Optional
import numpy as np


def find_boundary(
    subset: set[tuple] = None,
    n: int = 3,
    p: int = 5,
    starting_matrix: Optional[np.ndarray] = None,
) -> set[tuple]:
    """Find boundary of set."""
    if not subset:
        subset = get_graph(n, p, start_matrix=starting_matrix)
    edges = get_edges(n, p)
    boundary: set[tuple] = set()

    for Xtup in subset:
        X = tuple2matrix(Xtup, n)
        Xes = np.matmul(X, edges)
        Xes %= p
        for Xe in Xes:
            Xe_tup = tuple(np.ravel(Xe))
            if Xe_tup not in boundary:
                boundary.add(Xe_tup)
    return boundary
