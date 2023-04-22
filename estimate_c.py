import numpy as np
from tuple_graph import get_graph, get_edges, check_start_matrix
from typing import Optional
from tuple2matrix import tuple2matrix
from boundary import find_boundary
from size import degree
import random


def estimate_c(
    n: int,
    p: int,
    times: int = 1000,
    stop: Optional[int] = None,
    start_matrix: Optional[np.ndarray] = None,
) -> float:
    """Estimate c."""
    whole_graph = get_graph(n, p)
    max = degree(n, p)
    c = []
    for _ in range(times):
        if stop == None:
            stop = random.randint(1, max)
        start_matrix = random.choice(tuple(whole_graph))
        start_matrix = tuple2matrix(start_matrix, n)
        graph = get_graph(n, p, start_matrix=start_matrix, stop = stop)
        boundary = find_boundary(graph, n, p)
        c.append(len(boundary) / ((1 - stop / max)*stop))
    return c

if __name__ == "__main__":
    print(estimate_c(3, 5), times = 10)
