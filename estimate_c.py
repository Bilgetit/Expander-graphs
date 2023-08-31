import numpy as np
from multiprocessing import Pool
from typing import Optional
import time
from tuple_graph import get_graph, get_edges
from tuple2matrix import tuple2matrix
from boundary import find_boundary
from random_matrix import random_matrices
from size import degree


def get_rand_subset(
    n: int, p: int, subset_stop: int, matrix_list: Optional[np.ndarray] = []
) -> set[tuple]:
    """get random subset of graph"""
    subset = set()
    matrices = random_matrices(n=n, p=p, times=subset_stop, matrix_list=matrix_list)
    for j in range(subset_stop):
        subset.add(tuple(np.ravel(matrices[j])))
    return subset


class estimate:
    """class for estimating c, generating random subsets, and calculating the edges."""

    def __init__(
        self,
        n: int = 3,
        p: int = 5,
        times: int = 1,
        parallel: bool = False,
        subset_end: Optional[int] = None,
        matrix_end: Optional[int] = None,
        start_matrix: Optional[np.ndarray] = None,
    ):
        self.n = n
        self.p = p
        self.times = times
        self.subset_end = subset_end
        self.start_matrix = None
        self.matrix_end = matrix_end
        self.parallel = parallel
        self.degree = degree(self.n, self.p)
        self.c = []

        if start_matrix is not None:
            self.start_matrix = start_matrix

    def serial_worker(self, i: int):
        """worker without multiprocessing. Estimates single c."""
        if self.subset_end is not None and self.subset_end < self.degree:
            # self.subset_stop = np.random.randint(1, self.subset_end - 1)
            self.subset_stop = self.subset_end
        else:
            self.subset_stop = np.random.randint(1, self.degree - 1)
            """stop will be the size of the subset we are taking the boundary of."""

        # subset = get_rand_subset(self.n, self.p, self.subset_stop)
        # boundary, _ = find_boundary(self.n, self.p, subset=subset)
        # c = len(boundary) / ((1 - len(subset) / self.degree) * len(subset))
        # self.c.append(c)

        boundary, subset = find_boundary(
            self.n, self.p, size=self.subset_stop, starting_matrix=self.start_matrix
        )
        c = len(boundary) / ((1 - len(subset) / self.degree) * len(subset))
        self.c.append(c)

    def serial_do_work(self):
        """function without multiprocessing."""
        for i in range(self.times):
            self.serial_worker(i)

    def worker(self, i: int):
        """worker for multiprocessing. Estimates single c."""

        if self.subset_end is not None and self.subset_end < self.degree:
            # self.subset_stop = np.random.randint(1, self.subset_end - 1)
            self.subset_stop = self.subset_end
        else:
            self.subset_stop = np.random.randint(1, self.degree - 1)
            """stop will be the size of the subset we are taking the boundary of."""

        graph = get_graph(self.n, self.p)
        subset = np.random.choice(tuple(graph), size=self.subset_stop)

        # boundary, subset = find_boundary(
        #     self.n, self.p, size=self.subset_stop, starting_matrix=self.start_matrix[i]
        # )
        boundary, _ = find_boundary(self.n, self.p, subset=subset)

        # c = len(boundary) / ((1 - self.subset_stop / self.degree) * self.subset_stop)
        c = len(boundary) / ((1 - len(subset) / self.degree) * len(subset))
        print("hi")
        return c

    def do_work(self):
        with Pool() as pool:
            self.c = pool.map(self.worker, range(self.times))

    def main(self):
        if self.parallel:
            self.do_work()
            # c = min(self.c)
            return self.c
        else:
            self.serial_do_work()
            # min_c = min(self.c)
            # max_c = max(self.c)
            return self.c


def time_estimate_c(
    n: int,
    p: int,
    times: int = 100,
    parallel: bool = False,
    subset_end: Optional[int] = None,
    start_matrix: Optional[np.ndarray] = None,
) -> list[float]:
    """Timer for estimate_c."""
    t_0 = time.perf_counter()
    instance = estimate(
        n=n,
        p=p,
        times=times,
        parallel=parallel,
        subset_end=subset_end,
        start_matrix=start_matrix,
    )
    c = instance.main()
    t_1 = time.perf_counter()
    print(f"n = {n}, p = {p}, times = {times}, subset_end = {subset_end}")
    print(f"t_1 - t_0 = {t_1-t_0}")
    print(f"c = {c}")
    print("\n")


def get_c(
    n: int,
    p: int,
    times: int = 1,
    subset_end: Optional[int] = None,
    start_matrix: Optional[np.ndarray] = None,
    parallel: bool = False,
) -> float:
    instance = estimate(
        n=n,
        p=p,
        times=times,
        subset_end=subset_end,
        start_matrix=start_matrix,
        parallel=parallel,
    )
    c = min(instance.main())
    return c


if __name__ == "__main__":
    # time_estimate_c(3, 5, times=100, subset_end = None)
    time_estimate_c(2, 11, times=1, subset_end=1056, parallel=False)
    # time_estimate_c(3, 5, times=2, subset_end = 35)
    # time_estimate_c(2, 11, times=100, parallel=False)
