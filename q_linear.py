import util
import numpy as np
import matplotlib.pyplot as plt
import math
from qiskit import QuantumCircuit, Aer, execute
from qiskit.extensions import Initialize
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector
from linear_solvers import NumPyLinearSolver, HHL
from scipy.linalg import svd

"""
This file accesses the Qiskit Library and runs the HHL algorithm for our Quantum Linear Model
"""

def run(train_path, lamb):
    # Amplitude Encoding
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)

    M = np.matmul(train_x.T, train_x)
    # Regularization
    M = M + lamb * np.eye(M.shape[0])
    v = np.matmul(train_x.T, train_y)

    # Analyze the Determinant and Condition Variables
    print(np.linalg.det(M))
    print(np.linalg.cond(M))

    # Using Qiskit Library to run HHL Algorithm
    naive_hhl_solution = HHL().solve(M, v)
    # Using Qiskit Library to run a Classical Linear Solver (gives different theta to our linear model, but similar)
    classical_solution = NumPyLinearSolver().solve(M, v)

    # Normalize
    solution_vec = Statevector(naive_hhl_solution.state).data[4096:4100].real
    norm = naive_hhl_solution.euclidean_norm
    norm_sol = 2 * solution_vec / np.sum(solution_vec)
    print("solution_vec", solution_vec)
    print("norm", norm)
    print("norm_sol", norm_sol)
    print(Statevector(naive_hhl_solution.state).data.shape[0])
    #print(classical_solution.euclidean_norm)
    print("Classic Sol", classical_solution.state)


def main(train_path):
    _lambda = np.linspace(0.8, 1.1, 20)
    for i in _lambda:
        run(train_path, i)


if __name__ == '__main__':
    main(train_path='train.csv')
