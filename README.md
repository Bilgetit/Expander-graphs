# Expander-graphs
Coding specific examples of expanders, and estimating their expansion factor, for a project in my bachelor program

## Relevant files:
* `Thesis.pdf`: This is the final paper for which this project was made. It contains the theory behind the project, and the results gathered from the code.
* `tuple_graph.py`: Contains the class 'Search', which uses a breadth first search to find the graph (entire or a subset), and returns a set of tuples representing the matrices, which are vertices of the graph. The matrices are elements of the set $\textnormal{SL}(n, p)$, where $n$ is the dimension and $p$ is the prime number. This is the special linear group, which is the set of all matrices with determinant $1$. The values of the matrix are in the field $\mathbb{Z} / p\mathbb{Z}$, where $\mathbb{Z}$ is the set of integers, and $p$ is the prime number.

* `estimate_c.py`: Contains the class estimate, which estimates the expansion factor of $\textnormal{SL}(n, p)$, using the other files given. This is the main function of the project.

* `tuple2matrix.py`: Contains the function `tuple2matrix`, which takes a tuple and returns its corresponding matrix.

* `size.py`: Contains the function degree, which returns the size of the whole graph of matrices in $\textnormal{SL}(n, p)$. 

* `boundary.py`: Contains the function boundary, which returns the size of the boundary of a subset of $\textnormal{SL}(n, p)$.

* `random_matrix.py`: Contains the function rand_matrix, which returns a random matrix in $\textnormal{SL}(n, p)$.

* `test.py`: Contains tests for the functions in the other files.

