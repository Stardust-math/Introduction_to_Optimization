import cvxpy as cp
import numpy as np

x = cp.Variable(2)
t = cp.Variable()

A0 = np.array([
    [1, 0, 7],
    [0, 5, 3],
    [7, 3, 2]
], dtype=float)

A1 = np.array([
    [2, -2, 2],
    [-2, 1, 0],
    [2, 0, 1]
], dtype=float)

A2 = np.array([
    [9, 2, 1],
    [2, 5, 6],
    [1, 6, 4]
], dtype=float)

M = A0 + x[0]*A1 + x[1]*A2

objective = cp.Minimize(t)
constraints = [
    M << t*np.eye(3)
]

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS, eps=1e-8)

print("optimal value =", prob.value)
print("optimal x =", x.value)
print("optimal t =", t.value)