import sys
import numpy as np
import time as time
from hashlib import sha1
from numba import njit, jit, generated_jit, float32, prange
import numba as roc
from collections import deque
from heapq import heappush, heappop



# sys.settrace

sys.setrecursionlimit(100_000)            # set recursion limit to 1 million, REMEMBER TO CHANGE THIS BACK TO 1000 WHEN DONE
# sys.setrecursionlimit(1000)
"""
class that makes a numpy array hashable, so that it can be used as an element in a set (or key in a dictionary)
"""
class HashableNdarray(np.ndarray):
    @classmethod
    def create(cls, array):
        return HashableNdarray(shape=array.shape, dtype=array.dtype, buffer=array.copy())

    def __hash__(self):
        if not hasattr(self, '_HashableNdarray__hash'):
            self.__hash = int(sha1(self.view()).hexdigest(), 16)
        return self.__hash

    def __eq__(self, other):
        if not isinstance(other, HashableNdarray):
            return super().__eq__(other)
        return super().__eq__(super(HashableNdarray, other)).all()

## Parameters
n = 3           # size of matrix
p = 3           # Z/pZ
# m = int(1E1)    # number of matrices to generate


# Set which geerates SL(n, Z) (n = 3). Use these as edges in the graph
A_3 = (np.array([[1,1,0],[0,1,0],[0,0,1]])).view(HashableNdarray)
Ai_3 = (np.array([[1,-1,0],[0,1,0],[0,0,1]])).view(HashableNdarray)
# Ai_3 = np.array([[1,2,0],[0,1,0],[0,0,1]])
B_3 = (np.array([[0,1,0],[0,0,1],[1,0,0]])).view(HashableNdarray)
Bi_3 = (np.array([[0,0,1],[1,0,0],[0,1,0]])).view(HashableNdarray)
edges_3 = [A_3, Ai_3, B_3, Bi_3]

A_2 = (np.array([[1,1],[0,1]])).view(HashableNdarray)
Ai_2 = (np.array([[1,-1],[0,1]])).view(HashableNdarray)
B_2 = (np.array([[0,1],[-1,0]])).view(HashableNdarray)
Bi_2 = (np.array([[0,-1],[1,0]])).view(HashableNdarray)
edges_2 = [A_2, Ai_2, B_2, Bi_2]
# print(A, Ai, B, Bi, sep="\n")




""" 
function that returns the boundary of a set
"""
def boundary_set(my_set):
    wrong = 0                                       # number of wrong matrices
    boundary = set()
    for X in my_set:            
        wrong = add_edges(boundary, X, wrong) 
    print(f"{wrong=}")
    print(f"size of boundary = {len(boundary)}")    # actual size of boundary
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
helper function for generate_set_2
"""
# @njit()
def add_edges(X, e, my_set, n, p, count):
    Xe = np.matmul(X, e)
    Xe = Xe % p                         # make sure that the entries are in the field Z_p
    if Xe.view(HashableNdarray) not in my_set:
        my_set.add(Xe.view(HashableNdarray))
        if n == 2:
            edges = edges_2
        if n == 3:
            edges = edges_3
        for e in edges:
            count += 1
            count = add_edges(Xe, e, my_set, n, p, count)
    return count
    

"""
function that generates a set of random matrices of size n x n
by starting with edges, and recursively adding edges to the boundary
"""
def generate_set_2(s = set(), n=3, p=11, count=0):
    if n == 2:
        edges = edges_2
    if n == 3:
        edges = edges_3

    for e in edges:
        count += 1
        s.add(e)
        count = add_edges(e, e, s, n, p, count = count)
    return s, count


@njit(fastmath=True)
def in_field(Xe, n: int, p: int):
    for a in range(0, n):
        for b in range(0, n):
            Xe[a, b] = Xe[a, b] % p
    return Xe

def generate_set_bfs(s = set(), n=3, p=11):
    if n == 2:
        edges = edges_2
    if n == 3:
        edges = edges_3

    queue = deque(edges)
    count = 0
    while queue:
        x = queue.popleft()
        s.add(x)
        count += 1
        for e in edges:
            Xe = np.matmul(x, e)
            
            if Xe not in s:
                queue.append(Xe)

    print(f"{count=}")
    return s


def time_set_bfs():
    t0 = time.time()
    print(type(edges_3[0]))
    # my_set = generate_set_bfs(s = set(edges_3[0].view(HashableNdarray)), n=n, p=p)
    my_set = generate_set_bfs(n=n, p=p)
    t1 = time.time()
    print(f"{t1-t0=}")
    print(f"size of set_bfs = {len(my_set)}")
    print("\n")
    # print(my_set)                               # print the set



# import cProfile
# cProfile.run('generate_set_bfs(s = set(edges_3[0].view(HashableNdarray)), n=3, p=11)')
# my_set = generate_set_2(n=n, p=p)
# print(f"size of set_2 = {len(my_set)}")


def time_set_2():
    t0 = time.time()
    my_set, count = generate_set_2(n=n, p=p)
    t1 = time.time()
    print(f"{t1-t0=}")
    print(f"size of set_2 = {len(my_set)}")
    print(f"{count=}")
    print("\n")
    # print(my_set)                               # print the set





# @njit(fastmath=True)
def generate_set_dfs(s = set(), n=3, p=11):
    if n == 2:
        edges = edges_2
    if n == 3:
        edges = edges_3

    queue = []
    for e in edges:
        queue.append(e)
        
    count = 0
    while queue:
        count += 1
        x = queue.pop()
        s.add(x)
        for e in edges:
            Xe = np.matmul(x, e)
            Xe = Xe % p
            if Xe not in s:
                queue.append(Xe)
    print(f"{count=}")
    return s


def time_set_dfs():
    print("\n")
    t0 = time.time()
    my_set = generate_set_dfs(n=n, p=p)
    t1 = time.time()
    print(f"{t1-t0=}")
    print(f"size of set_dfs = {len(my_set)}")
    # print(my_set)                               # print the set


print("time generate_set_bfs")
time_set_bfs()

# print("time generate_set_2")
# time_set_2()

print("time generate_set_dfs")
time_set_dfs()



sys.setrecursionlimit(1000)

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
