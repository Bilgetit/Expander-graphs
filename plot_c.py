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
import pickle


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


def plot_c_against_p(n: int, max_p:int = 50, perc: float = 0.15, plot: bool = False) -> None:
    primes = [i for i in range(11, max_p) if isprime(i)]
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

def plot_c_3d(n: int = 2, max_p: int = 101, min_perc: float = 0.01, max_perc: float = 0.2, plot: bool = True, filename: str = 'FigureObject.fig.pickle') -> None:
    primes = [i for i in range(11, max_p) if isprime(i)]
    X = np.array(primes)
    length = int(len(primes))
    perc = np.linspace(min_perc, max_perc, length)
    Y = np.array(perc)
    X, Y = np.meshgrid(X, Y)
    # c = []
    # for percent in perc:
    #     c.append(np.array(c_worker(n=n, primes=primes, perc=percent)))
    with Pool() as pool:
        c = pool.starmap(c_worker, [(n, primes, i) for i in perc])
    c = np.array(c)
    print(c)
    min_c = np.min(c)
    print(f"min: {min_c}")
    index = np.where(c == min_c)
    print(f"min i: {index}")
    p_i, perc_i = index[1][0], index[0][0]
    print(f"min p: {primes[p_i]}")
    print(f"min perc: {perc[perc_i]}")

    if plot:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, c, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        # ax.scatter(X, Y, c, cmap=cm.coolwarm)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xlabel("p")
        plt.ylabel("|A|")
        pickle.dump(fig, open(filename, 'wb'))
        # plt.clabel("c")
        # plt.show()

def c_worker(n: int, primes: np.array, perc: int) -> float:
    # with Pool() as pool:
    #     c = pool.starmap(calc_c_p, [(n, p, perc) for p in primes])
    c = []
    for p in primes:
        c.append(calc_c_p(n=n, p=p, perc=perc))
    return c


def calc_c_p(n: int, p: int, perc: float = 0.1) -> float:
    size = int(perc * degree(n, p))
    c = get_c(n=n, p=p, subset_end=size)
    return c

def show_3d_figure():
    fig = pickle.load(open('FigureObject.fig.pickle', 'rb'))
    plt.show()


if __name__ == "__main__":
    t_0 = time.time()
    # plot_c_against_A(n=2, p=13, plot=True)
    # plot_c_against_p(n=2, max_p = 150, perc=0.15, plot=True)
    plot_c_3d(n=2, max_p=30, min_perc=0.01, max_perc=0.4, plot=True)
    # show_3d_figure()
    t_1 = time.time()
    print(f"Time taken: {t_1 - t_0}")
    # plt.show()


