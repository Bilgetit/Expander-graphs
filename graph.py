import numpy as np
import time
from hashlib import sha1
from collections import deque

"""
class that makes a numpy array hashable, so that it can be used as an element in a set (or key in a dictionary)
"""


class HashableNdarray(np.ndarray):
    @classmethod
    def create(cls, array):
        return HashableNdarray(
            shape=array.shape, dtype=array.dtype, buffer=array.copy()
        )

    def __hash__(self):
        if not hasattr(self, "_HashableNdarray__hash"):
            self.__hash = int(sha1(self.view()).hexdigest(), 16)
        return self.__hash

    def __eq__(self, other):
        if not isinstance(other, HashableNdarray):
            return super().__eq__(other)
        return super().__eq__(super(HashableNdarray, other)).all()


## Parameters
n = 3  # size of matrix
p = 3  # Z/pZ
m_max = int(1e1)  # number of matrices to generate


# Set which geerates SL(n, Z) (n = 3). Use these as edges in the graph
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


def generate_whole_set(s=set(), n=3, p=11):
    if n == 2:
        edges = edges_2
    if n == 3:
        edges = edges_3

    queue = deque(edges)
    m = 0
    for e in edges:
        s.add(e)
        m += 1
    count = 0
    while queue:
        x = queue.popleft()
        for e in edges:
            Xe = np.matmul(x, e)
            Xe = Xe % p
            if Xe not in s:
                m += 1
                s.add(Xe)
                queue.append(Xe)
    print(f"{count=}")
    return s, m


def time_set_bfs():
    t0 = time.time()
    # my_set, m = generate_whole_set(s = set(edges_3[0].view(HashableNdarray)), n=n, p=p)
    my_set, m = generate_whole_set(n=n, p=p)
    t1 = time.time()
    print(f"{t1-t0=}")
    print(f"size of set_bfs = {len(my_set)}")
    print(f"{m=}")
    print("\n")


# import cProfile
# cProfile.run('generate_whole_set(s = set(edges_3[0].view(HashableNdarray)), n=3, p=11)')
# my_set = generate_set_2(n=n, p=p)
# print(f"size of set_2 = {len(my_set)}")


print("time generate_whole_set")
time_set_bfs()


def generate_subset(X, m_max, s=set(), n=3, p=11):
    if n == 2:
        edges = edges_2
    if n == 3:
        edges = edges_3
    queue = deque(X)
    s = set(X)
    m = 1
    while m < m_max and queue:
        m += 1
        x = queue.popleft()
        for e in edges:
            Xe = np.matmul(x, e)
            Xe = Xe % p
            if Xe not in s:
                s.add(Xe)
                queue.append(Xe)
    return s, m


""" 
function that returns the boundary of a set
"""


def boundary_set(my_set):
    wrong = 0  # number of wrong matrices
    boundary = set()
    # for X in my_set:
        # wrong = add_edges(boundary, X, wrong)
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
