import numpy as np
from multiprocessing import Pool
from typing import Optional
import time
from tuple_graph import get_graph, get_edges
from tuple2matrix import tuple2matrix
from boundary import find_boundary
from random_matrix import get_matrix
from size import degree


class estimate:
    """class for estimating c, generating random subsets, and calculating the edges."""

    def __init__(
        self,
        n: int = 3,
        p: int = 5,
        times: int = 1000,
        parallel: bool = True,
        subset_end: Optional[int] = None,
        matrix_end: Optional[int] = None,
        start_matrix: Optional[np.ndarray] = None,
    ):
        self.n = n
        self.p = p
        self.times = times
        self.subset_end = subset_end
        self.matrix_end = matrix_end
        self.start_matrix = start_matrix
        self.parallel = parallel
        self.degree = degree(self.n, self.p)
        self.c = None

    def serial_worker(self):
        """worker for multiprocessing. Estimates single c."""
        if self.subset_end is not None:
            self.subset_stop = np.random.randint(1, self.subset_end - 1)
        else:
            self.subset_stop = np.random.randint(1, self.degree - 1)
            """stop will be the size of the subset we are taking the boundary of."""

        if self.start_matrix is not None:
            # self.stop = np.random.randint(1, 500)
            self.start_matrix = get_matrix(
                self.n, self.p, stop=self.stop, starting_matrix=self.start_matrix
            )
            """get_matrix now starts from a random matrix in the graph, given by the last self.start_matrix."""
        else:
            # self.stop = np.random.randint(1, 500)
            self.stop = 300
            self.start_matrix = get_matrix(self.n, self.p, stop=self.stop)

        boundary = find_boundary(
            self.n, self.p, size=self.subset_stop, starting_matrix=self.start_matrix
        )
        c = len(boundary) / ((1 - self.subset_stop / self.degree) * self.subset_stop)

        if c < self.c:
            self.c = c

    def serial_do_work(self):
        for _ in range(self.times):
            self.serial_worker()

    def worker(self, _):
        """worker for multiprocessing. Estimates single c."""
        if self.subset_end is not None:
            self.subset_stop = np.random.randint(1, self.subset_end - 1)
        else:
            self.subset_stop = np.random.randint(1, self.degree - 1)
            """stop will be the size of the subset we are taking the boundary of."""

        if self.start_matrix is not None:
            # self.stop = np.random.randint(1, 500)
            self.start_matrix = get_matrix(
                self.n, self.p, stop=self.stop, starting_matrix=self.start_matrix
            )
            """get_matrix now starts from a random matrix in the graph, given by the last self.start_matrix."""
        else:
            # self.stop = np.random.randint(1, 500)
            self.stop = 100
            self.start_matrix = get_matrix(self.n, self.p, stop=self.stop)

        boundary = find_boundary(
            self.n, self.p, size=self.subset_stop, starting_matrix=self.start_matrix
        )
        c = len(boundary) / ((1 - self.subset_stop / self.degree) * self.subset_stop)

        return c
        # if c < self.c:
        #     self.c = c

    # def do_work(self):
    #     for _ in range(self.times):
    #         self.worker()

    def do_work(self):
        with Pool() as pool:
            self.c = pool.map(self.worker, range(self.times))

    def main(self):
        if self.parallel:
            self.do_work()
            c = min(self.c)
            return c
        else:
            self.c = 100
            self.serial_do_work()
            return self.c


def time_estimate_c(
    n: int,
    p: int,
    times: int = 100,
    parallel: bool = True,
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
    print(f"n = {n}, p = {p}, times = {times}")
    print(f"t_1 - t_0 = {t_1-t_0}")
    print(f"c = {c}")
    print("\n")


def get_c(
    n: int,
    p: int,
    times: int = 100,
    subset_end: Optional[int] = None,
    start_matrix: Optional[np.ndarray] = None,
) -> float:
    instance = estimate(n, p, times, subset_end, start_matrix)
    c = instance.main()
    return c


if __name__ == "__main__":
    time_estimate_c(2, 11, times=100)
    time_estimate_c(2, 11, times=100, parallel=False)
