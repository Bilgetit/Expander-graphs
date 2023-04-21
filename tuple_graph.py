import numpy as np
import time
from collections import deque


def get_edges(n: int) -> np.ndarray:
    "Starting edges of graph."
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

    def __init__(self, n: int = 3, p: int = 5, printing: bool = False):
        self.n = n
        self.p = p
        self.s = set()
        self.printing = printing
        self.edges = get_edges(n)
        self.count = 0
        self.quit = False

    def do_work(self):
        x = self.queue.popleft()

        Xes = np.matmul(x, self.edges)
        Xes %= self.p

        for Xe in Xes:
            Xe_tup = tuple(np.ravel(Xe))

            if Xe_tup not in self.s:
                self.s.add(Xe_tup)
                self.queue.append(Xe)

    def worker(self):
        while self.queue:
            self.do_work()

            self.count += 1

            if self.count % 100000 == 0:
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


if __name__ == "__main__":
    time_set_bfs()
