import numpy as np
import time as time
from hashlib import sha1
from numba import njit


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
m = int(1E3)    # number of matrices to generate


# Set which generates SL(n, Z) (n = 3). Use these as edges in the graph
A = np.array([[1,1,0],[0,1,0],[0,0,1]])
Ai = np.array([[1,-1,0],[0,1,0],[0,0,1]])
B = np.array([[0,1,0],[0,0,1],[1,0,0]])
Bi = np.array([[0,0,1],[1,0,0],[0,1,0]])
edges = [A, Ai, B, Bi]  

# print(A, Ai, B, Bi, sep="\n")

    

"""
function that generates a random matrix of size n x n
with entries in the field Z_p, where p is a prime number
with determinant equal to 1 (mod p)
"""
# @njit()
def generate_matrix(n, p):
    X = np.random.randint(0, high=p, size=(n,n))        # nxn-matrix with random entries between 0 and p-1

    while check_matrix(X) == False:    # only let pass given that we can make the determinant 1 (mod p)
        X = np.random.randint(0, high=p, size=(n,n))  

    a = X[0][0]
    b = X[0][1]
    c = X[0][2]
    d = X[1][0]
    e = X[1][1]
    f = X[1][2]
    g = X[2][0]
    h = X[2][1]
    i = X[2][2]

    i = (-b*f*g - c*d*h + c*e*g + a*f*h + 1) / (a*e - b*d)      # calculate the last entry of the matrix so that the determinant is 1 (mod p)
    i = i%p                                                     # make the last entry be in the field Z_p 
    X[2][2] = i                                                 # set the last entry of the matrix to i  
    return X


"""
helper function for generatematrix that checks if a matrix has determinant 1 (mod p)
"""
@njit()
def check_matrix(X):
    a = X[0][0]
    b = X[0][1]
    c = X[0][2]
    d = X[1][0]
    e = X[1][1]
    f = X[1][2]
    g = X[2][0]
    h = X[2][1]
    i = X[2][2]

    if (a*e - b*d) % p == 0:            # make sure we don't divide by 0 
        return False
    
    i = (-b*f*g - c*d*h + c*e*g + a*f*h + 1) / (a*e - b*d)

    if abs(i-int(i)) < 0.0000000001:            # make sure that i is an integer
        return True
    else: 
        return False


"""
function that generates a set of matrices of size n x n
with entries in the field Z_p, where p is a prime number
with determinant equal to 1 (mod p)
"""
def generate_set(m, func, s = None, n=3, p=11):
    s = set()
    for i in range(m):                  # m = number of matrices to generate
        X= func(n, p)               # func = generate_matrix
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
    # print(s)
    print(f"{t1-t0=}")
    print(f"size of set = {len(my_set)}")
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
def add_edges(boundary, X, wrong):
    for e in edges:                                             # edges = [A, Ai, B, Bi]
        Xe = np.matmul(X, e)                                    # X*e or e*X? 
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
        boundary.add(X.view(HashableNdarray))
        wrong = add_edges(boundary, X, wrong) 
    print(f"{wrong=}")
    print(f"size of boundary = {len(boundary)}")    # actual size of boundary
    return boundary

"""
use this to generate a set and calculate the boundary
"""

my_set = generate_set(m, generate_matrix, p=p)
boundary_set(my_set)