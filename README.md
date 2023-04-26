# Expander-graphs
Coding specific examples of expanders, and estimating their expansion factor, for a project in my bachelor program

Relevant files:
* 'tuple_graph.py': Contains the class 'Search', which uses a breadth first search to find the graph (entire or a subset), and returns a set of tuples representing the matrices, which are vertices of the graph. The matrices are elements of the set SL(n, p), where n is the dimension and p is the prime number. This is the special linear group, which is the set of all matrices with determinant 1. The values of the matrix are in the field Z/pZ, where Z is the set of integers, and p is the prime number.

* 'tuple2matrix.py': contains the function 'tuple2matrix', which takes a tuple and returns its corresponding matrix.

* 'size.py': Contains the function degree, which returns the size of the whole graph of matrices in SL(n, p). 

* 'boundary.py': Contains the function boundary, which returns the size of the boundary of a subset of SL(n, p).

* 'estimate_c.py': Contains the class estimate, which estimates the expansion factor of SL(n, p).

* 'test.py': contains tests for the functions in the other files.

