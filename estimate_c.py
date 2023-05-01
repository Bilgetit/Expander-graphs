import numpy as np
from multiprocessing import Pool
from typing import Optional
import time
from tuple_graph import get_graph, get_edges
from tuple2matrix import tuple2matrix
from boundary import find_boundary
from random_matrix import get_matrix
from size import degree


def random_matrices(
    n: int,
    p: int,
    times: int,
    start_matrix: Optional[np.ndarray] = [],
    matrix_end: Optional[int] = None,
) -> list[np.ndarray]:
    """generates random matrices."""

    if matrix_end == None:
        stop = 1
        # stop = np.random.randint(1, 500)

    else:
        stop = matrix_end

    if start_matrix == []:
        start_matrix.append(get_matrix(n, p, stop=stop))

    for i in range(1, times):
        start_matrix.append(
            get_matrix(n, p, stop=stop, starting_matrix=start_matrix[i - 1])
        )

    return start_matrix


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
        self.start_matrix = []
        self.matrix_end = matrix_end
        self.parallel = parallel
        self.degree = degree(self.n, self.p)
        self.c = []

        if start_matrix is not None:
            self.start_matrix = [start_matrix]

    def serial_worker(self, i: int):
        """worker without multiprocessing. Estimates single c."""
        if self.subset_end is not None:
            # self.subset_stop = np.random.randint(1, self.subset_end - 1)
            self.subset_stop = self.subset_end
        else:
            self.subset_stop = np.random.randint(1, self.degree - 1)
            """stop will be the size of the subset we are taking the boundary of."""

        boundary = find_boundary(
            self.n, self.p, size=self.subset_stop, starting_matrix=self.start_matrix[i]
        )
        c = len(boundary) / ((1 - self.subset_stop / self.degree) * self.subset_stop)

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

        boundary = find_boundary(
            self.n, self.p, size=self.subset_stop, starting_matrix=self.start_matrix[i]
        )
        c = len(boundary) / ((1 - self.subset_stop / self.degree) * self.subset_stop)

        return c

    def do_work(self):
        with Pool() as pool:
            self.c = pool.map(self.worker, range(self.times))

    def main(self):
        self.start_matrix = random_matrices(
            self.n, self.p, self.times, self.start_matrix, self.matrix_end
        )
        if self.parallel:
            self.do_work()
            c = min(self.c)
            return self.c
        else:
            self.serial_do_work()
            c = min(self.c)
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
    print(f"n = {n}, p = {p}, times = {times}, subset_end = {subset_end}")
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
    # time_estimate_c(3, 5, times=100, subset_end = None)
    time_estimate_c(3, 5, times=2, subset_end=100_000, parallel=False)
    # time_estimate_c(3, 5, times=2, subset_end = 35)
    # time_estimate_c(2, 11, times=100, parallel=False)
