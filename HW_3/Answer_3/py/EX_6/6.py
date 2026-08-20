import numpy as np
import time
import cvxpy as cp

def proj_l1_ball(a, radius=1.0):
    a = np.asarray(a, dtype=float)
    if np.linalg.norm(a, 1) <= radius:
        return a.copy(), 0.0

    u = np.sort(np.abs(a))[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u - (cssv - radius) / np.arange(1, len(u) + 1) > 0)[0][-1] + 1
    lam = (cssv[rho - 1] - radius) / rho
    x = np.sign(a) * np.maximum(np.abs(a) - lam, 0.0)
    return x, lam

# random test
np.random.seed(0)
n = 300
a = np.random.randn(n)

# custom method
t0 = time.perf_counter()
x_custom, lam = proj_l1_ball(a)
t1 = time.perf_counter()

# CVXPY
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x - a)),
                  [cp.norm1(x) <= 1])

t2 = time.perf_counter()
val = prob.solve(solver=cp.CLARABEL, verbose=False)
t3 = time.perf_counter()

x_cvx = np.array(x.value).reshape(-1)

print("custom time:", t1 - t0)
print("cvxpy time:", t3 - t2)
print("max abs diff:", np.max(np.abs(x_custom - x_cvx)))
print("custom objective:", 0.5 * np.sum((x_custom - a) ** 2))
print("cvx objective:", 0.5 * np.sum((x_cvx - a) ** 2))
print("lambda*:", lam)