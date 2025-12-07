import time
from typing import Callable
import numpy as np
import sympy as sp
import gauss_jordan_cpp
import sys

def test_timings(func: Callable, *args):
    _ = func(*args)
    start_time = time.time()
    _ = func(*args)
    end_time = time.time()
    return round(end_time - start_time, 5)

def test_gauss_jordan(matrix_size: int, tol: float):
    print("Test gauss_jordan pybind implementation")

    matrix_A = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size)

    list_A = matrix_A.tolist()
    list_b = matrix_b.tolist()

    my_sol = gauss_jordan_cpp.gauss_jordan.solve_linear_system(list_A, list_b)
    sp_A = sp.Matrix(list_A)
    sp_b = sp.Matrix(list_b)
    sp_sol = sp_A.gauss_jordan_solve(sp_b)
    if isinstance(sp_sol, tuple):
        sp_sol_values = sp_sol[0]
    else:
        sp_sol_values = sp_sol
    
    sp_sol_flat = [float(val) for val in sp_sol_values]
    
    solver = lambda m_A, m_b: m_A.gauss_jordan_solve(m_b)
    err = np.linalg.norm(np.array(my_sol) - np.array(sp_sol_flat))

    print("my GaussJordan: on {0}x{0} took {1} seconds".format(matrix_size, test_timings(gauss_jordan_cpp.gauss_jordan.solve_linear_system, list_A, list_b)))
    print("sympy GaussJordan: on {0}x{0} took {1} seconds".format(matrix_size, test_timings(solver, sp_A, sp_b)))
    
    if err < tol:
        print("test of size {0}x{0} succeeded!".format(matrix_size))
        return True
    else:
        print("test of size {0}x{0} failed! tol {1} < err {2}".format(matrix_size, tol, err))
        return False

if __name__ == "__main__":
    tol = 1e-5
    for size in [10, 50, 100]:
        if not test_gauss_jordan(size, tol):
            sys.exit(1)