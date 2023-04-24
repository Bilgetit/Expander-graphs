import numpy as np
from tuple_graph import get_graph



def degree(n, p):
    if n == 2:
        return (p+1)*(p**2-p)
    elif n == 3:
        return (p**2+p+1)*(p**3-p)*(p**3-p**2)
    else:
        raise ValueError(f"Not implemented for {n=}.")
    


if __name__ == "__main__":
    print(degree(3,5))