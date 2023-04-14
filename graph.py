import numpy as np
import time as time
from hashlib import sha1
from numba import njit

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

n = 3
p = 13
m = int(1E3)



A = np.array([[1,1,0],[0,1,0],[0,0,1]])
Ai = np.array([[1,-1,0],[0,1,0],[0,0,1]])
B = np.array([[0,1,0],[0,0,1],[1,0,0]])
Bi = np.array([[0,0,1],[1,0,0],[0,1,0]])


def random_matrix(n, p, count):
    X = np.random.randint(0, high=p, size=(n,n))
    # if np.linalg.det(X) == 0:
    #     count += 1
    #     return random_matrix(n, p, count)
    
    a = X[0][0]
    b = X[0][1]
    c = X[0][2]
    d = X[1][0]
    e = X[1][1]
    f = X[1][2]
    g = X[2][0]
    h = X[2][1]
    i = X[2][2]

    if a*e - b*d == 0:
        count += 1
        return random_matrix(n, p, count)
    
    i = (-b*f*g - c*d*h + c*e*g + a*f*h + 1) / (a*e - b*d)

    if i > 0 and i < p and i-int(i) < 0.00000001:
        X[2][2] = i
        return X, count

    else: 
        count += 1
        return random_matrix(n, p, count)

"""
function that generates a random matrix of size n x n
with entries in the field Z_p, where p is a prime number
with determinant equal to 1
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

    if a*e - b*d == 0:
        return False
    
    i = (-b*f*g - c*d*h + c*e*g + a*f*h + 1) / (a*e - b*d)

    if i > 0 and i < p and i-int(i) < 0.0000000001:
        return True

    else: 
        return False
    

@njit()
def generate_matrix(n, p, count=0):
    X = np.random.randint(0, high=p, size=(n,n))  

    while check_matrix(X) == False:
        X = np.random.randint(0, high=p, size=(n,n))  
        count += 1 

    a = X[0][0]
    b = X[0][1]
    c = X[0][2]
    d = X[1][0]
    e = X[1][1]
    f = X[1][2]
    g = X[2][0]
    h = X[2][1]
    i = X[2][2]

    i = (-b*f*g - c*d*h + c*e*g + a*f*h + 1) / (a*e - b*d)
    X[2][2] = i

    return X, count

def generate_set(m, func, s = None, n=3, p=11):
    if s is None:
        s = set()
    for i in range(m):
        X, _ = func(n, p, count=0)
        # X_hashable = map(tuple, X)
        s.add(X.view(HashableNdarray))
        # s.add(HashableNdarray.create(X))
    return s


def time_generate(func):
    count = 0
    t0 = time.time()
    X, count = func(n, p, count) 
    t1 = time.time()
    print(X)
    print(f"det(X) = {np.linalg.det(X)}")     
    print(f"{t1-t0=}")
    print(f"{count=}")

# print("random_matrix")
# time_generate(random_matrix)
# print("\n")
# print("generate_matrix")
# time_generate(generate_matrix)
# print("\n")

def time_set(func1, func2):
    t0 = time.time()
    my_set = func1(m, func2) 
    t1 = time.time()
    # print(s)
    print(f"{t1-t0=}")
    print(f"size of set = {len(my_set)}")
    print(f"{m=}")


print("generate_matrix")
time_set(generate_set, generate_matrix)

my_set = generate_set(m, generate_matrix)
copy = my_set.pop()
print(copy)

def boundary_set(my_set):
    boundary = set()
    for x in my_set:
        generate_set()
    return boundary



# def timeit(func):
#     def wrapper(*args, **kwargs):
#         t0 = time.time()
#         result = func(*args, **kwargs)
#         t1 = time.time()
#         print(f"{t1-t0=}")
#         return result
#     return wrapper