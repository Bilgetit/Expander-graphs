import numpy as np
import time
from collections import deque
from typing import Optional


def get_edges(n: int, p: int) -> np.ndarray:
    "Starting edges of graph."

    if n == 2:
        A_2 = np.array([[1, 1], [0, 1]])
        Ai_2 = np.array([[1, -1], [0, 1]])
        B_2 = np.array([[0, 1], [-1, 0]])
        Bi_2 = np.array([[0, -1], [1, 0]])

        Ai_2 %= p
        B_2 %= p

        return np.array([A_2, Ai_2, B_2, Bi_2])
        # return np.array([A_2, B_2])

    elif n == 3:
        A_3 = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        Ai_3 = np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]])
        B_3 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        Bi_3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        Ai_3 %= p

        return np.array([A_3, Ai_3, B_3, Bi_3])
        # return np.array([A_3, B_3])

    else:
        raise ValueError(f"Not implemented for {n=}.")
    
def get_edges_half(n: int, p: int) -> np.ndarray:
    "Starting edges of graph."

    if n == 2:
        A_2 = np.array([[1, 1], [0, 1]])
        # Ai_2 = np.array([[1, -1], [0, 1]])
        B_2 = np.array([[0, 1], [-1, 0]])
        # Bi_2 = np.array([[0, -1], [1, 0]])

        # Ai_2 %= p
        B_2 %= p

        # return np.array([A_2, Ai_2, B_2, Bi_2])
        return np.array([A_2, B_2])

    elif n == 3:
        A_3 = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        # Ai_3 = np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]])
        B_3 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        # Bi_3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        # Ai_3 %= p

        # return np.array([A_3, Ai_3, B_3, Bi_3])
        return np.array([A_3, B_3])

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
        if start_matrix is not None:
            self.s.add(tuple(np.ravel(start_matrix)))
        self.printing = printing
        self.start_matrix = start_matrix
        self.edges = get_edges(n, p)
        self.count = 0
        self.quit = False
        self.stop = stop

    def worker(self) -> None:
        x = self.queue.popleft()

        Xes = np.matmul(x, self.edges)
        Xes %= self.p

        for Xe in Xes:
            Xe_tup = tuple(np.ravel(Xe))

            if Xe_tup not in self.s:
                self.s.add(Xe_tup)
                self.queue.append(Xe)
                self.count += 1

                if self.count % 100_000 == 0 and self.printing:
                    print(f"on number {self.count=}")

                if self.stop is not None and self.count >= self.stop:
                    self.quit = True
                    break

    def do_work(self) -> None:
        while self.queue:
            self.worker()

            if self.quit:
                break

    def main(self):
        if self.start_matrix is not None:
            self.queue = deque([self.start_matrix])
            x_tup = tuple(np.ravel(self.start_matrix))
            self.s.add(x_tup)
            self.count += 1
            if self.stop is not None and self.count >= self.stop:
                self.quit = True
                return self.s
        else:
            self.queue = deque(self.edges)

        self.do_work()

        return self.s


def time_set_bfs():
    t0 = time.time()
    instance = Search(n=2, p=101, printing=True)
    my_set = instance.main()
    t1 = time.time()
    print(f" t_1 - t_0 = {t1-t0}")
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


"""
(MAT2000) Mikkels-MacBook-Pro:Expander-graphs mikkelgjestrud$ /opt/anaconda3/envs/MAT2000/bin/python /Users/mikkelgjestrud/Documents/Prosjekter/Expander-graphs/tuple_graph.py
on number self.count=1000000
on number self.count=2000000
on number self.count=3000000
on number self.count=4000000
on number self.count=5000000
on number self.count=6000000
on number self.count=7000000
on number self.count=8000000
on number self.count=9000000
on number self.count=10000000
on number self.count=11000000
on number self.count=12000000
on number self.count=13000000
on number self.count=14000000
on number self.count=15000000
on number self.count=16000000
on number self.count=17000000
on number self.count=18000000
on number self.count=19000000
on number self.count=20000000
on number self.count=21000000
on number self.count=22000000
on number self.count=23000000
on number self.count=24000000
on number self.count=25000000
on number self.count=26000000
on number self.count=27000000
on number self.count=28000000
on number self.count=29000000
on number self.count=30000000
on number self.count=31000000
on number self.count=32000000
on number self.count=33000000
on number self.count=34000000
on number self.count=35000000
on number self.count=36000000
t1-t0=7521.402703046799
size of set_bfs = 36846576
"""
