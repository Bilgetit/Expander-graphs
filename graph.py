import numpy as np
import time as time
from hashlib import sha1
from numba import njit
import sys

sys.setrecursionlimit(1_000_000)            # set recursion limit to 1 million, REMEMBER TO CHANGE THIS BACK TO 1000 WHEN DONE
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
m = int(1E5)    # number of matrices to generate


# Set which geerates SL(n, Z) (n = 3). Use these as edges in the graph
A_3 = np.array([[1,1,0],[0,1,0],[0,0,1]])
Ai_3 = np.array([[1,-1,0],[0,1,0],[0,0,1]])
# Ai_3 = np.array([[1,2,0],[0,1,0],[0,0,1]])
B_3 = np.array([[0,1,0],[0,0,1],[1,0,0]])
Bi_3 = np.array([[0,0,1],[1,0,0],[0,1,0]])
edges_3 = [A_3, Ai_3, B_3, Bi_3]

A_2 = np.array([[1,1],[0,1]])
Ai_2 = np.array([[1,-1],[0,1]])
B_2 = np.array([[0,1],[-1,0]])
Bi_2 = np.array([[0,-1],[1,0]])
edges_2 = [A_2, Ai_2, B_2, Bi_2]
# print(A, Ai, B, Bi, sep="\n")





"""
function that returns the last entry needed to make the determinant of a 2x2 matrix 1 (mod p)
"""
@njit()
def get_last_2(X):
    a = X[0][0]
    b = X[0][1]
    c = X[1][0]
    d = X[1][1]

    # if b == 0:                         # make sure we don't divide by 0 
    #     return None
    
    # d = (-a*c + 1) / b
    # d = d%p
    
    # if np.linalg.det(X) % p == 0:      # make sure det(X) != 0
    #     return None
    r = np.random.randint(0, high=p)
    for n in range(p):
        i = (n + r) % p
        # X[1][1] = i
        if (a*i - b*c) % p == 1:
            return i
     
    

    # if abs(d-int(d)) < 0.0000000001:            
    #     return d                     # make sure that d is an integer
    # else:
    return None



    
"""
function that returns the last entry needed to make the determinant of a 3x3 matrix 1 (mod p)
"""
@njit()
def get_last_3(X):
    a = X[0][0]
    b = X[0][1]
    c = X[0][2]
    d = X[1][0]
    e = X[1][1]
    f = X[1][2]
    g = X[2][0]
    h = X[2][1]
    i = X[2][2]

    # if (a*e - b*d) % p == 0:            # make sure we don't divide by 0 
    #     return None
    
    # if np.linalg.det(X) % p == 0:       # make sure det(X) != 0
    #     return None
    
    # i = (-b*f*g - c*d*h + c*e*g + a*f*h + 1) / (a*e - b*d)      # calculate the last entry of the matrix so that the determinant is 1 (mod p)
    # i = i%p                                                     # make sure that i is in Z_p

    # if abs(i-int(i)) < 0.0000000001:            
    #     return i                     # make sure that i is an integer

    r = np.random.randint(0, high=p)
    for n in range(p):
        i = (n + r) % p
        X[2][2] = i
        # det = (X[0][0]*X[1][1]*X[2][2] + X[0][1]*X[1][2]*X[2][0] + X[0][2]*X[1][0]*X[2][1] - X[0][2]*X[1][1]*X[2][0] - X[0][1]*X[1][0]*X[2][2] - X[0][0]*X[1][2]*X[2][1]) % p
        det = (a*e*i - a*f*h - b*d*i + b*g*f + c*d*h - c*g*e) % p
        if det == 1:
            return i
    
    # else:
    return None




"""
function that generates a random matrix of size n x n
with entries in the field Z_p, where p is a prime number
with determinant equal to 1 (mod p)
"""
# @njit()
def generate_matrix(n, p):
    X = np.random.randint(0, high=p, size=(n,n))        # nxn-matrix with random entries between 0 and p-1

    if n == 2:
        while True:    # only let pass given that we can make the determinant 1 (mod p)
            X = np.random.randint(0, high=p, size=(n,n))  
            d = get_last_2(X)         # set the last entry of the matrix to the last entry needed to make the determinant 1 (mod p)
            if d != None:
                X[1][1] = d 
                break
        return X
    
    if n == 3:
        while True:     # only let pass given that we can make the determinant 1 (mod p)
            X = np.random.randint(0, high=p, size=(n,n))
            i = get_last_3(X)         # set the last entry of the matrix to the last entry needed to make the determinant 1 (mod p)
            if i != None:
                X[2][2] = i
                break
        return X


"""
function that generates a set of matrices of size n x n
with entries in the field Z_p, where p is a prime number
with determinant equal to 1 (mod p)
"""
def generate_set(m, func, s = None, n=3, p=11):
    s = set()
    for i in range(m):                  # m = number of matrices to generate
        X = func(n=n, p=p)               # func = generate_matrix
        s.add(X.view(HashableNdarray))  # add the matrix to the set
    return s


"""
timer for make one matrix, using the generate_matrix function
"""
def time_matrix(func):
    t0 = time.time()
    X  = func(n, p) 
    t1 = time.time()
    print(X)
    print(f"det(X) = {np.linalg.det(X)}")     
    print(f"{t1-t0=}")                      # time it takes to generate 1 matrix

"""
use this to time the generate_matrix function
"""

# print("time generate_matrix")
# time_matrix(generate_matrix)
# print("\n")

""" 
timer for the generate_set function
"""
def time_set(func1, func2):
    t0 = time.time()
    my_set = func1(m, func2, n=n, p=p)          # func1 = generate_set, func2 = generate_matrix
    t1 = time.time()
    print(f"{t1-t0=}")
    print(f"size of set = {len(my_set)}")
    # print(my_set)                               # print the set
    print(f"{m=}")

"""
use this to time the generate_set function
"""

print("time generate_set")
time_set(generate_set, generate_matrix)
print("\n")

""" 
function that adds edges to the boundary of a set
"""
# @njit()
def add_edges(boundary, X, wrong):
    boundary.add(X.view(HashableNdarray))
    if n == 2:
        edges = edges_2
    if n == 3:
        edges = edges_3
    for e in edges:                                             # edges = [A, Ai, B, Bi]
        Xe = np.matmul(X, e)

        for a in range(0, n):                                   
            for b in range(0, n):
                Xe[a][b] = Xe[a][b] % p                         # make sure that the entries are in the field Z_p

        boundary.add(Xe.view(HashableNdarray))

        if abs((np.linalg.det(Xe.view(HashableNdarray)) % p) - 1) > 0.00000001:
            wrong += 1          # if-statement to check if the determinant is 1 (and count the number of wrong matrices)

    return wrong

                

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



# def check_set(my_set):
#     for X in my_set:
#         for a in range(0, n):
#             for b in range(0, n):
#                 if X[a][b] % p != X[a][b]:
#                     print("wrong")

def time_boundary(my_set):
    t0 = time.time()
    boundary = boundary_set(my_set)
    t1 = time.time()
    print(f"{t1-t0=}")
    print(f"size of boundary = {len(boundary)}")



"""
use this to generate a set and calculate the boundary
"""
my_set = generate_set(m, generate_matrix, n=n, p=p)
boundary = boundary_set(my_set)
# boundary2 = boundary_set(boundary)
# print(f"boundary = {boundary}")

"""
time boundary_set
"""
print("time boundary_set")
time_boundary(my_set)
print("\n")




def add_edges(X, e, my_set, n, p):
    Xe = np.matmul(X, e)
    for a in range(0, n):                                   
        for b in range(0, n):
            Xe[a][b] = Xe[a][b] % p                         # make sure that the entries are in the field Z_p
    if Xe.view(HashableNdarray) not in my_set:
        my_set.add(Xe.view(HashableNdarray))
        if n == 2:
            edges = edges_2
        if n == 3:
            edges = edges_3
        for e in edges:
            add_edges(Xe, e, my_set, n, p)
    

"""
function that generates a set of random matrices of size n x n
by starting with edges, and recursively adding edges to the boundary
"""
def generate_set_2(s = set(), n=3, p=11):
    if n == 2:
        edges = edges_2
    if n == 3:
        edges = edges_3

    for e in edges:
        add_edges(e, e, s, n, p)

    return s

def time_set_2():
    print("\n")
    t0 = time.time()
    my_set = generate_set_2(n=n, p=p)
    t1 = time.time()
    print(f"{t1-t0=}")
    print(f"size of set_2 = {len(my_set)}")
    # print(my_set)                               # print the set
    print(f"{m=}")

print("time generate_set_2")
time_set_2()
# my_set = generate_set_2(n=n, p=p)
# print(f"size of set_2 = {len(my_set)}")

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