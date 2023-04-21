import numpy as np
import time
from multiprocessing import Process, Queue
import multiprocessing as mp


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
        x = self.queue.get_nowait()

        Xes = np.matmul(x, self.edges)
        Xes %= self.p

        for Xe in Xes:
            Xe_tup = tuple(np.ravel(Xe))

            if Xe_tup not in self.s:
                self.s.add(Xe_tup)
                self.queue.put(Xe)

        # self.queue.task_done()

    def worker(self, name: str):
        # if name != 'worker-0':
        #     print(f'{name}')
        #     await asyncio.sleep(0.1)
        #     print(f'{name} slept for 0.5, {self.quit}')
        while not self.quit:
            if name != "worker-0":
                print(name)
            self.do_work()
            
            if self.queue.qsize() == 0:
                print('Hola')
                self.quit = True
            self.count += 1

            # if self.count % 10001 == 0:
            #     print(f"{name} check {self.count}")
        return True

    def tuple_generate_set_bfs(
            s: set[tuple] = set(), n: int = 3, p: int = 11
            ):
        """Function for generating graph.

        Uses tuples in order to find unique matrices.

        Arguments:
            s (set): set of edges
                (stored as tuples of flattened matrices)
            n (int): dim of matrix (must be square)
            p (int): prime number, for graph

        Returns:
            set of tuples: matrices of the found graph.
        """

        edges = get_edges(n)
        queue = Queue(maxsize=0)
        for e in edges:
            queue.put(e)

        while queue:
            x = queue.popleft()
            Xes = np.matmul(x, edges)
            Xes = Xes % p
            for Xe in Xes:
                Xe_tup = tuple(np.ravel(Xe))
                if Xe_tup not in s:
                    s.add(Xe_tup)
                    queue.append(Xe)

        return s

    def main(self):
        self.queue = Queue(maxsize=0)
        for e in self.edges:
            self.queue.put(e)

        # while self.queue.qsize() <= 10:
        x = self.queue.get()

        Xes = np.matmul(x, self.edges)
        Xes %= self.p

        for Xe in Xes:
            Xe_tup = tuple(np.ravel(Xe))
            if Xe_tup not in self.s:
                self.s.add(Xe_tup)
                self.queue.put(Xe)
        # self.queue.task_done()

        self.processes = []
        number_of_processes = 8

        for w in range(number_of_processes):
            p = Process(name=f'worker-{w}', target=self.worker, args=(f'worker-{w}'))
            self.processes.append(p)
            p.start()

        for p in self.processes:
            p.join()
        

        # for task in self.tasks:
        #     task.cancel()

        return self.s


# def generate_set(n: int, p: int, printing: bool = False):
#     instance = Search(n, p, printing)
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

#     found_set = asyncio.run(instance.main())

#     return found_set


def time_set_bfs():
    ctx = mp.get_context("spawn")
    (child, pipe) = ctx.Pipe(duplex=True)
    t0 = time.time()
    # print(type(edges_3[0]))
    # my_set = generate_set_bfs(s = set(edges_3[0].view(HashableNdarray)), n=n, p=p)
    instance = Search(n=3, p=5)
    my_set = instance.main()
    t1 = time.time()
    print(f"{t1-t0=}")
    print(f"size of set_bfs = {len(my_set)}")
    print("\n")


if __name__ == "__main__":
    time_set_bfs()
