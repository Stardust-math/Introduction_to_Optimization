import cvxpy as cp
import numpy as np
from itertools import product

def solve_qp(u1, u2, solver=cp.OSQP):
    x = cp.Variable(2)

    P = np.array([
        [1.0, -0.5],
        [-0.5, 2.0]
    ])
    q = np.array([-1.0, 0.0])

    A = np.array([
        [5.0, 76.0],
        [1.0,  2.0],
        [1.0, -4.0]
    ])
    u = np.array([1.0, u1, u2])

    obj = cp.Minimize(cp.quad_form(x, P) + q @ x)

    c1 = A[0] @ x <= u[0]
    c2 = A[1] @ x <= u[1]
    c3 = A[2] @ x <= u[2]

    prob = cp.Problem(obj, [c1, c2, c3])
    prob.solve(solver=solver)

    xval = np.array(x.value, dtype=float).reshape(-1)
    lam = np.array([c1.dual_value, c2.dual_value, c3.dual_value], dtype=float)

    slacks = u - A @ xval

    # Correct dual objective
    s = q + A.T @ lam
    dual_obj = -0.25 * s @ np.linalg.solve(P, s) - lam @ u

    return {
        "status": prob.status,
        "x": xval,
        "p_star": prob.value,
        "d_star": dual_obj,
        "duals": lam,
        "slacks": slacks,
        "A": A,
        "P": P,
        "q": q,
        "u": u
    }

# base point
u1_base, u2_base = -2, -3
res0 = solve_qp(u1_base, u2_base)

x_star = res0["x"]
p_star = res0["p_star"]
d_star = res0["d_star"]
lam = res0["duals"]
slacks = res0["slacks"]
A = res0["A"]

print("Base problem")
print("status =", res0["status"])
print("x* =", x_star)
print("p* =", p_star)
print("d* =", d_star)
print("primal-dual gap =", p_star - d_star)
print("dual multipliers =", lam)
print("slacks =", slacks)

# KKT check
grad = np.array([
    2*x_star[0] - x_star[1] - 1,
    4*x_star[1] - x_star[0]
], dtype=float)

stationarity = grad + A.T @ lam
comp_slack = lam * slacks

print("\nKKT check")
print("gradient =", grad)
print("stationarity residual =", stationarity)
print("||stationarity||_inf =", np.max(np.abs(stationarity)))
print("complementary slackness residual =", comp_slack)
print("||comp_slack||_inf =", np.max(np.abs(comp_slack)))
print("dual feasibility (lam >= 0) =", np.all(lam >= -1e-8))
print("primal feasibility (slacks >= 0) =", np.all(slacks >= -1e-8))

# prediction
print("\nPredictions")
for d1, d2 in product([-0.1, 0.1], repeat=2):
    u1 = -2 + d1
    u2 = -3 + d2
    pred = p_star - lam[1]*d1 - lam[2]*d2
    exact = solve_qp(u1, u2)["p_star"]
    print(f"delta=({d1:+.1f}, {d2:+.1f}) | pred={pred:.9f} | exact={exact:.9f} | gap={exact-pred:.9f}")