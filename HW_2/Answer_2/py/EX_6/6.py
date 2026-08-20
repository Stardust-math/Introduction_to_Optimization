import time
from pathlib import Path

import cvxpy as cp
import numpy as np


def project_to_simplex(y):
    """
    Project y onto the probability simplex:
        {x | x >= 0, sum(x) = 1}
    """
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u - (cssv - 1) / (np.arange(len(u)) + 1) > 0)[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    x = np.maximum(y - theta, 0.0)
    return x, theta


def objective_value(x, y):
    return 0.5 * np.linalg.norm(x - y) ** 2


def solve_with_cvxpy(y):
    n = len(y)
    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(x - y))
    constraints = [x >= 0, cp.sum(x) == 1]
    problem = cp.Problem(objective, constraints)

    # Prefer OSQP for this QP; if unavailable, fall back to default solver.
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        problem.solve(verbose=False)

    if x.value is None:
        raise RuntimeError("CVXPY failed to return a solution.")

    return np.asarray(x.value).reshape(-1), float(problem.value)


def main():
    n = 500
    seed = 0
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)

    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Custom algorithm
    t0 = time.perf_counter()
    x_alg, theta = project_to_simplex(y)
    t1 = time.perf_counter()

    alg_time = t1 - t0
    alg_obj = objective_value(x_alg, y)

    # CVXPY
    t2 = time.perf_counter()
    x_cvx, cvx_obj = solve_with_cvxpy(y)
    t3 = time.perf_counter()

    cvx_time = t3 - t2

    # Comparison
    l2_diff = float(np.linalg.norm(x_alg - x_cvx))
    linf_diff = float(np.linalg.norm(x_alg - x_cvx, ord=np.inf))

    sum_alg = float(np.sum(x_alg))
    sum_cvx = float(np.sum(x_cvx))

    min_alg = float(np.min(x_alg))
    min_cvx = float(np.min(x_cvx))

    support_alg = int(np.sum(x_alg > 1e-10))
    support_cvx = int(np.sum(x_cvx > 1e-10))

    # Save vectors
    np.savetxt(results_dir / "y.csv", y, delimiter=",")
    np.savetxt(results_dir / "x_algorithm.csv", x_alg, delimiter=",")
    np.savetxt(results_dir / "x_cvxpy.csv", x_cvx, delimiter=",")

    # Save concise summary
    with open(results_dir / "results.txt", "w", encoding="utf-8") as f:
        f.write(f"n = {n}\n")
        f.write(f"seed = {seed}\n")
        f.write(f"theta = {theta}\n")
        f.write(f"algorithm_time = {alg_time}\n")
        f.write(f"cvxpy_time = {cvx_time}\n")
        f.write(f"algorithm_objective = {alg_obj}\n")
        f.write(f"cvxpy_objective = {cvx_obj}\n")
        f.write(f"l2_difference = {l2_diff}\n")
        f.write(f"linf_difference = {linf_diff}\n")
        f.write(f"sum_x_algorithm = {sum_alg}\n")
        f.write(f"sum_x_cvxpy = {sum_cvx}\n")
        f.write(f"min_x_algorithm = {min_alg}\n")
        f.write(f"min_x_cvxpy = {min_cvx}\n")
        f.write(f"support_size_algorithm = {support_alg}\n")
        f.write(f"support_size_cvxpy = {support_cvx}\n")

    # Console output: necessary results only
    print("theta =", theta)
    print("algorithm_time =", alg_time)
    print("cvxpy_time =", cvx_time)
    print("algorithm_objective =", alg_obj)
    print("cvxpy_objective =", cvx_obj)
    print("l2_difference =", l2_diff)
    print("linf_difference =", linf_diff)
    print("sum_x_algorithm =", sum_alg)
    print("sum_x_cvxpy =", sum_cvx)
    print("min_x_algorithm =", min_alg)
    print("min_x_cvxpy =", min_cvx)
    print("support_size_algorithm =", support_alg)
    print("support_size_cvxpy =", support_cvx)


if __name__ == "__main__":
    main()