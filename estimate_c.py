import numpy as np
from tuple_graph import get_graph, get_edges, check_start_matrix
from typing import Optional
from tuple2matrix import tuple2matrix
from boundary import find_boundary
from size import degree
import random
import time


def estimate_c(
    n: int,
    p: int,
    times: int = 1000,
    end: Optional[int] = None,
    start_matrix: Optional[np.ndarray] = None,
) -> float:
    """Estimate c."""
    whole_graph = get_graph(n, p)
    max = degree(n, p)
    c = []
    for _ in range(times):
        if end:
            stop = random.randint(1, end-1)
        else:
            stop = random.randint(1, max-1)
        start_matrix = random.choice(tuple(whole_graph))
        start_matrix = tuple2matrix(start_matrix, n)
        boundary, len_boundary = find_boundary(n, p, size = stop, starting_matrix=start_matrix)
        c.append(len_boundary / ((1 - stop / max)*stop))           # c = |B| / (1 - |A|/|V|) * |A|. Find better way of len(boundary)?
    return c

def time_estimate_c(
    n: int,
    p: int,
    times: int = 1000,
    end: Optional[int] = None,
    start_matrix: Optional[np.ndarray] = None,
) -> float:
    """Timer for estimate_c."""
    t_0 = time.time()
    c = estimate_c(n, p , times = times)
    t_1 = time.time()
    print(f"{t_1-t_0=}")
    # print(c)
    print(f"min = {min(c)}")
    return c

if __name__ == "__main__":
    time_estimate_c(2, 13, times = 100)
