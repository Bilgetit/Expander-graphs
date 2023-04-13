import numpy as np
import time

n = 3
p = 11


A = np.array([[1,1,0],[0,1,0],[0,0,1]])
Ai = np.array([[1,-1,0],[0,1,0],[0,0,1]])
B = np.array([[0,1,0],[0,0,1],[1,0,0]])
Bi = np.array([[0,0,1],[1,0,0],[0,1,0]])

"""
function that generates a random matrix of size n x n
with entries in the field Z_p, where p is a prime number
with determinant equal to 1
"""
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
        # i = int(i)
        X[2][2] = i
        # print(f"{i=}")
        # print(X)
        # print(a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h)
        # print(np.linalg.det(X))
        return X, count

    else: 
        count += 1
        return random_matrix(n, p, count)


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

    if i > 0 and i < p and i-int(i) < 0.00000001:
        return True

    else: 
        return False

def generate_matrix(n, p, count=0):
    X = np.random.randint(0, high=p, size=(n,n))  

    while check_matrix(X) == False:
        X = np.random.randint(0, high=p, size=(n,n))  
        count += 1 
    

    # while a*e - b*d != 0 or i < 0 or i > p or i-int(i) > 0.00000001 or count == 0:
    #     a = X[0][0]
    #     b = X[0][1]
    #     c = X[0][2]
    #     d = X[1][0]
    #     e = X[1][1]
    #     f = X[1][2]
    #     g = X[2][0]
    #     h = X[2][1]
    #     i = X[2][2] 
    #     i = (-b*f*g - c*d*h + c*e*g + a*f*h + 1) / (a*e - b*d)
    #     count += 1
    #     X = np.random.randint(0, high=p, size=(n,n))   

    # if a*e - b*d == 0:
    #     count += 1
    #     return generate_matrix(n, p, count)
    
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




# def set_of_matrices(n, p, count, ):
#     set = {}
#     size = 0
#     X = np.random.randint(0, high=p, size=(n,n))

#     while np.linalg.det(X) != 1 or size < n:
#         count += 1
#         X = np.random.randint(0, high=p, size=(n,n))
#         set.append(X)
#     return set, count

# def timeit(func):
#     def wrapper(*args, **kwargs):
#         t0 = time.time()
#         result = func(*args, **kwargs)
#         t1 = time.time()
#         print(f"{t1-t0=}")
#         return result
#     return wrapper


# count = 0
# t0 = time.time()
# X, count = random_matrix(n, p, count)
# t1 = time.time()
# print(f"{t1-t0=}")
# print(X)
# print(f"det(X) = {np.linalg.det(X)}")
# print(f"{count=}")

# print("\n")

count = 0
t0 = time.time()
X, count = generate_matrix(n, p, count)
t1 = time.time()
print(f"{t1-t0=}")
print(X)
print(np.linalg.det(X))
print(count)

# count = 0
# t0 = time.time()
# X, count = generate_matrix_slow(n, p, count)
# t1 = time.time()
# print(f"{t1-t0=}")
# print(X)
# print(np.linalg.det(X))
# print(count)