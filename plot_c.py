import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from multiprocessing import Pool
from sympy import isprime
import time
from tuple_graph import get_graph, get_edges
from tuple2matrix import tuple2matrix
from boundary import find_boundary
from random_matrix import random_matrices
from size import degree
from estimate_c import get_c


def plot_c_against_A(n: int = 3, p: int = 5):
    perc = np.linspace(0.01, 0.5, 500)
    size = perc * degree(n, p)
    c = [get_c(n=n, p=p, subset_end=int(int(i))) for i in size]
    fig = plt.figure(1, (7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(perc, c)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xlabel("|A|")
    plt.ylabel("c")
    plt.show()


def plot_c_against_p(n: int, perc: float = 0.1) -> None:
    primes = [i for i in range(11, 200) if isprime(i)]
    # with Pool() as pool:
    #     c = pool.map(calc_c_p, [(n, p, perc) for p in primes])
    for p in primes:
        size = int(perc * degree(n, p))
        c = [get_c(n=n, p=p, subset_end=size * perc) for p in primes]
    fig = plt.figure(1, (7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(primes, c)
    plt.xlabel("p")
    plt.ylabel("c")
    plt.show()


def calc_c_p(n: int, p: int, perc: float = 0.1) -> float:
    size = int(perc * degree(n, p))
    c = get_c(n=n, p=p, subset_end=size)
    return c


if __name__ == "__main__":
    t_0 = time.time()
    # plot_c_against_A(n=2, p=17)
    plot_c_against_p(n=2, perc=0.1)
    t_1 = time.time()
    print(f"Time taken: {t_1 - t_0}")
