import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import cm
from multiprocessing import Pool
from sympy import isprime
import time
from size import degree
from estimate_c import get_c


def plot_c_against_A(n: int = 3, p: int = 5, plot = False) -> None:
    perc = np.linspace(0.001, 0.2, 29)
    with Pool() as pool:
        c = pool.starmap(calc_c_p, [(n, p, i) for i in perc])
    # size = perc * degree(n, p)
    # c = [get_c(n=n, p=p, subset_end=int(int(i))) for i in size]
    if plot:
        fig = plt.figure(1, (7, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(perc, c)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xlabel("|A|")
        plt.ylabel("c")
        plt.show()
    return c


def plot_c_against_p(n: int, perc: float = 0.1, plot: bool = False) -> None:
    primes = [i for i in range(11, 110) if isprime(i)]
    with Pool() as pool:
        c = pool.starmap(calc_c_p, [(n, p, perc) for p in primes])
    if plot:
        fig = plt.figure(1, (7, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(primes, c)
        plt.xlabel("p")
        plt.ylabel("c")
        plt.show()
    return c

def plot_c_3d(n: int = 2, max_p: int = 101, min_perc: float = 0.01, max_perc: float = 0.2, plot: bool = False) -> None:
    primes = [i for i in range(11, max_p) if isprime(i)]
    X = np.array(primes)
    perc = np.linspace(min_perc, max_perc, int(len(primes)))
    Y = np.array(perc)
    X, Y = np.meshgrid(X, Y)
    c = []
    for p in primes:
        c.append(np.array(c_worker(n=n, p=p, perc=perc)))
    c = np.array(c)
    print(c)

    if plot:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, c, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xlabel("p")
        plt.ylabel("|A|")
        # plt.clabel("c")
        plt.show()

def c_worker(n: int, p: int, perc: np.array) -> float:
    with Pool() as pool:
        c = pool.starmap(calc_c_p, [(n, p, percent) for percent in perc])
    return c


def calc_c_p(n: int, p: int, perc: float = 0.1) -> float:
    size = int(perc * degree(n, p))
    c = get_c(n=n, p=p, subset_end=size)
    return c


if __name__ == "__main__":
    t_0 = time.time()
    # plot_c_against_A(n=2, p=13, plot=True)
    # plot_c_against_p(n=2, perc=0.1, plot=True)
    plot_c_3d(n=2, max_p=110, min_perc=0.05, max_perc=0.2, plot=True)
    t_1 = time.time()
    print(f"Time taken: {t_1 - t_0}")
