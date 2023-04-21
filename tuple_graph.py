import numpy as np
import time
from collections import deque
from typing import Optional


def get_edges(n: int, start_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    "Starting edges of graph."
    if start_matrix is not None:
        if start_matrix.shape[0] != start_matrix.shape[1]:
            raise ValueError(f"{start_matrix=} must be square.")
        if not np.all(start_matrix >= 0):
            raise ValueError(f"{start_matrix=} must be positive.")
        return np.array([start_matrix])

    if n == 2:
        A_2 = np.array([[1, 1], [0, 1]])
        Ai_2 = np.array([[1, -1], [0, 1]])
        B_2 = np.array([[0, 1], [-1, 0]])
        Bi_2 = np.array([[0, -1], [1, 0]])

        return np.array([A_2, Ai_2, B_2, Bi_2])

    elif n == 3:
        A_3 = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        Ai_3 = np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]])
        B_3 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        Bi_3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        return np.array([A_3, Ai_3, B_3, Bi_3])

    else:
        raise ValueError(f"Not implemented for {n=}.")


class Search:
    """Class for using a Breadth first search in order to find graph."""

    def __init__(
        self,
        n: int = 3,
        p: int = 5,
        printing: bool = False,
        stop: Optional[int] = None,
        start_matrix: Optional[np.ndarray] = None,
    ):
        self.n = n
        self.p = p
        self.s = set()
        self.printing = printing
        if start_matrix is not None:
            self.edges = get_edges(n, start_matrix=start_matrix)
        else:
            self.edges = get_edges(n)
        self.count = 0
        self.quit = False
        self.stop = stop

    def do_work(self) -> None:
        x = self.queue.popleft()

        Xes = np.matmul(x, self.edges)
        Xes %= self.p

        for Xe in Xes:
            Xe_tup = tuple(np.ravel(Xe))

            if Xe_tup not in self.s:
                self.s.add(Xe_tup)
                self.queue.append(Xe)

    def worker(self) -> None:
        while self.queue:
            self.do_work()

            if self.stop is not None and self.count >= self.stop:
                self.quit = True
                break

            self.count += 1

            if self.count % 100000 == 0 and self.printing:
                print(f"on number {self.count=}")

    def main(self):
        self.queue = deque(self.edges)

        self.worker()

        return self.s


def time_set_bfs():
    t0 = time.time()
    instance = Search(n=3, p=7)
    my_set = instance.main()
    t1 = time.time()
    print(f"{t1-t0=}")
    print(f"size of set_bfs = {len(my_set)}")
    print("\n")


def get_graph(
    n: int,
    p: int,
    stop: Optional[int] = None,
    start_matrix: Optional[np.ndarray] = None,
    printing: bool = False,
):
    """Get graph of size n and prime p."""
    instance = Search(n=n, p=p, stop=stop, printing=printing, start_matrix=start_matrix)
    my_set = instance.main()
    return my_set


if __name__ == "__main__":
    time_set_bfs()
