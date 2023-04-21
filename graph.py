import sys
import numpy as np
import numpy.typing as npt
import time as time
from collections import deque
from HashArray import HashableNdarray


# sys.settrace
# set recursion limit to 1 million, REMEMBER TO CHANGE THIS BACK TO 1000 WHEN DONE
sys.setrecursionlimit(100_000)
# sys.setrecursionlimit(1000)

# Global params
n = 3  # size of matrix
p = 7  # Z/pZ
# m = int(1E1)    # number of matrices to generate


# Set which generates SL(n, Z) (n = 3). Use these as edges in the graph
A_3 = (np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])).view(HashableNdarray)
Ai_3 = (np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]])).view(HashableNdarray)
# Ai_3 = np.array([[1,2,0],[0,1,0],[0,0,1]])
B_3 = (np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).view(HashableNdarray)
Bi_3 = (np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])).view(HashableNdarray)
edges_3 = [A_3, Ai_3, B_3, Bi_3]

A_2 = (np.array([[1, 1], [0, 1]])).view(HashableNdarray)
Ai_2 = (np.array([[1, -1], [0, 1]])).view(HashableNdarray)
B_2 = (np.array([[0, 1], [-1, 0]])).view(HashableNdarray)
Bi_2 = (np.array([[0, -1], [1, 0]])).view(HashableNdarray)
edges_2 = [A_2, Ai_2, B_2, Bi_2]
# print(A, Ai, B, Bi, sep="\n")


def add_edges(*args):
    ...


def boundary_set(my_set):
    "Find boundary of set"
    wrong = 0  # number of wrong matrices
    boundary = set()
    for X in my_set:
        wrong = add_edges(boundary, X, wrong)
    print(f"{wrong=}")
    print(f"size of boundary = {len(boundary)}")  # actual size of boundary
    return boundary


def time_boundary(my_set):
    t0 = time.time()
    boundary = boundary_set(my_set)
    t1 = time.time()
    print(f"{t1-t0=}")
    print(f"size of boundary = {len(boundary)}")


"""
use this to generate a set and calculate the boundary
"""
# my_set = generate_set(m, generate_matrix, n=n, p=p)
# boundary = boundary_set(my_set)
# boundary2 = boundary_set(boundary)
# print(f"boundary = {boundary}")

"""
time boundary_set
"""
# print("time boundary_set")
# time_boundary(my_set)
# print("\n")


def generate_set_bfs(s: set[npt.NDArray[np.int_]] = set(), n: int = 3, p: int = 11):
    if n == 2:
        edges = edges_2
    if n == 3:
        edges = edges_3

    queue = deque(edges)
    count = 0
    while queue:
        count += 1
        x = queue.popleft()
        if count % 100000 == 0:
            print(f"Er på {count=}")
        for e in edges:
            Xe = np.matmul(x, e)
            Xe = Xe % p
            if Xe not in s:
                s.add(Xe)
                queue.append(Xe)

    print(f"{count=}")
    return s


# def tuple_generate_set_bfs(s: set[tuple] = set(), n: int = 3, p: int = 11):
#     if n == 2:
#         A_2 = np.array([[1, 1], [0, 1]])
#         Ai_2 = np.array([[1, -1], [0, 1]])
#         B_2 = np.array([[0, 1], [-1, 0]])
#         Bi_2 = np.array([[0, -1], [1, 0]])
#         edges_2 = np.array([A_2, Ai_2, B_2, Bi_2])
#         edges = edges_2
#     if n == 3:
#         A_3 = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
#         Ai_3 = np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]])
#         # Ai_3 = np.array([[1,2,0],[0,1,0],[0,0,1]])
#         B_3 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
#         Bi_3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
#         edges_3 = np.array([A_3, Ai_3, B_3, Bi_3])

#         edges = edges_3

#     queue = deque(edges)
#     while queue:
#         x = queue.popleft()
#         Xes = np.matmul(x, edges)
#         Xes = Xes % p
#         for Xe in Xes:
#             Xe_tup = tuple(np.ravel(Xe))
#             if Xe_tup not in s:
#                 s.add(Xe_tup)
#                 queue.append(Xe)

#     return s

def time_set_bfs():
    t0 = time.time()
    # print(type(edges_3[0]))
    # my_set = generate_set_bfs(s = set(edges_3[0].view(HashableNdarray)), n=n, p=p)
    my_set = generate_set_bfs(n=3, p=7)
    t1 = time.time()
    print(f"{t1-t0=}")
    print(f"size of set_bfs = {len(my_set)}")
    print("\n")
    # print(my_set)                               # print the set


# import cProfile
# cProfile.run('generate_set_bfs(s = set(edges_3[0].view(HashableNdarray)), n=3, p=11)')
# my_set = generate_set_2(n=n, p=p)
# print(f"size of set_2 = {len(my_set)}")


print("time generate_set_bfs")
time_set_bfs()

# print("time generate_set_2")
# time_set_2()

# print("time generate_set_dfs")
# time_set_dfs()


# sys.setrecursionlimit(1000)

"""
time generate_set
t1-t0=85.74626302719116
size of set = 4888
m=1000000


wrong=0
size of boundary = 5400
"""


"""
time generate_set
t1-t0=5.8329620361328125
size of set = 112
m=100000


wrong=0
size of boundary = 168
"""
