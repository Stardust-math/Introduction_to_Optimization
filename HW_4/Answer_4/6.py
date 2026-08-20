import math
import time
import numpy as np
import cvxpy as cp

np.set_printoptions(precision=6, suppress=True)


def generate_lp_feasible(n: int, m: int, seed: int = 0):
    """
    Generate a random feasible LP:
        minimize    c^T x
        subject to  A x = b, x >= 0

    We construct:
        b = A x0, with x0 > 0
    so x0 is a strictly feasible starting point.
    """
    rng = np.random.default_rng(seed)

    while True:
        A = rng.standard_normal((n, m))
        if np.linalg.matrix_rank(A) == n:
            break

    x0 = rng.uniform(0.5, 1.5, size=m)
    b = A @ x0

    # Make the problem bounded in a convenient way
    y = rng.standard_normal(n)
    s = rng.uniform(0.5, 1.5, size=m)  # strictly positive
    c = A.T @ y + s

    return A, b, c, x0


def centering_newton(A, b, c, x0, t, alpha=0.01, beta=0.5,
                     newton_tol=1e-8, max_newton=100):
    """
    Solve the equality-constrained centering problem:
        minimize    t c^T x - sum(log x_i)
        subject to  A x = b
    by Newton's method.
    """
    x = x0.astype(float).copy()
    history = []

    for it in range(max_newton + 1):
        grad = t * c - 1.0 / x
        # Hessian = diag(1/x_i^2), so inverse Hessian = diag(x_i^2)
        Hinv_diag = x ** 2

        # Schur complement system:
        # A H^{-1} A^T w = -A H^{-1} grad
        M = A @ (Hinv_diag[:, None] * A.T)
        rhs = -A @ (Hinv_diag * grad)
        w = np.linalg.solve(M, rhs)

        dx = -Hinv_diag * (grad + A.T @ w)

        lambda_sq = max(0.0, -grad @ dx)
        phi = t * (c @ x) - np.sum(np.log(x))
        eq_resid = np.linalg.norm(A @ x - b)

        history.append({
            "iter": it,
            "phi": phi,
            "lambda_sq_over_2": lambda_sq / 2.0,
            "eq_resid": eq_resid
        })

        if lambda_sq / 2.0 <= newton_tol:
            return x, it, history

        # Keep positivity
        step = 1.0
        neg_idx = dx < 0
        if np.any(neg_idx):
            step = min(step, 0.99 * np.min(-x[neg_idx] / dx[neg_idx]))

        directional_derivative = grad @ dx

        # Backtracking line search
        while True:
            x_trial = x + step * dx
            if np.all(x_trial > 0):
                phi_trial = t * (c @ x_trial) - np.sum(np.log(x_trial))
                if phi_trial <= phi + alpha * step * directional_derivative:
                    break
            step *= beta
            if step < 1e-16:
                raise RuntimeError("Backtracking line search failed.")

        x = x_trial

    raise RuntimeError("Newton method did not converge.")


def barrier_method(A, b, c, x0, mu=10.0, eps=1e-8,
                   newton_tol=1e-8, max_outer=100):
    """
    Barrier method for:
        minimize    c^T x
        subject to  A x = b, x >= 0
    """
    _, m = A.shape
    x = x0.astype(float).copy()
    t = 1.0

    outer_history = []
    total_newton_steps = 0

    tic = time.time()

    for outer_it in range(max_outer):
        x, nsteps, inner_history = centering_newton(
            A=A, b=b, c=c, x0=x, t=t, newton_tol=newton_tol
        )
        total_newton_steps += nsteps

        outer_history.append({
            "outer_iter": outer_it,
            "t": t,
            "primal_obj": float(c @ x),
            "gap_estimate": float(m / t),
            "newton_steps": int(nsteps),
            "eq_residual": float(np.linalg.norm(A @ x - b)),
            "min_x": float(np.min(x)),
            "last_lambda_sq_over_2": float(inner_history[-1]["lambda_sq_over_2"])
        })

        if m / t < eps:
            break

        t *= mu

    toc = time.time()

    return {
        "x": x,
        "outer_history": outer_history,
        "total_newton_steps": total_newton_steps,
        "runtime_sec": toc - tic
    }


def try_cvxpy_solver(A, b, c, solver_name, x_barrier):
    x = cp.Variable(A.shape[1])
    prob = cp.Problem(cp.Minimize(c @ x), [A @ x == b, x >= 0])

    try:
        tic = time.time()
        prob.solve(solver=solver_name, verbose=False)
        toc = time.time()

        return {
            "solver": str(solver_name),
            "status": prob.status,
            "time_sec": toc - tic,
            "obj": float(prob.value),
            "eq_residual": float(np.linalg.norm(A @ x.value - b)),
            "min_x": float(np.min(x.value)),
            "rel_x_diff_vs_barrier": float(
                np.linalg.norm(x.value - x_barrier) / max(1.0, np.linalg.norm(x.value))
            ),
            "obj_diff_vs_barrier": float(abs(prob.value - (c @ x_barrier)))
        }
    except Exception as e:
        return {
            "solver": str(solver_name),
            "status": f"FAILED: {e}"
        }


def print_dict_table(rows, columns, title=None, float_fmt="{:.6e}"):
    if title is not None:
        print("\n" + title)

    # compute widths
    widths = []
    for col in columns:
        max_len = len(col)
        for row in rows:
            val = row.get(col, "")
            if isinstance(val, float):
                s = float_fmt.format(val)
            else:
                s = str(val)
            max_len = max(max_len, len(s))
        widths.append(max_len)

    # header
    header = " | ".join(col.ljust(w) for col, w in zip(columns, widths))
    print(header)
    print("-" * len(header))

    # rows
    for row in rows:
        out = []
        for col, w in zip(columns, widths):
            val = row.get(col, "")
            if isinstance(val, float):
                s = float_fmt.format(val)
            else:
                s = str(val)
            out.append(s.ljust(w))
        print(" | ".join(out))


def run_part_a():
    print("=" * 80)
    print("PART (a): random instance with n = 100, m = 500")
    print("=" * 80)

    n, m = 100, 500
    A, b, c, x0 = generate_lp_feasible(n=n, m=m, seed=1)

    barrier = barrier_method(
        A=A, b=b, c=c, x0=x0,
        mu=10.0,
        eps=1e-8,
        newton_tol=1e-8
    )

    x_bar = barrier["x"]

    summary = {
        "runtime_sec": barrier["runtime_sec"],
        "objective": float(c @ x_bar),
        "total_newton_steps": barrier["total_newton_steps"],
        "outer_iterations": len(barrier["outer_history"]),
        "eq_residual": float(np.linalg.norm(A @ x_bar - b)),
        "min_x": float(np.min(x_bar))
    }

    print("\nBarrier summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.12e}")
        else:
            print(f"{k:20s}: {v}")

    print_dict_table(
        barrier["outer_history"],
        columns=[
            "outer_iter", "t", "primal_obj", "gap_estimate",
            "newton_steps", "eq_residual", "min_x"
        ],
        title="Outer iteration history",
        float_fmt="{:.6e}"
    )

    print("\nCVXPY comparison:")
    cvx_rows = []
    for solver in [cp.CLARABEL, cp.SCS]:
        cvx_rows.append(try_cvxpy_solver(A, b, c, solver, x_bar))

    # separate successful and failed rows
    success_rows = [r for r in cvx_rows if "time_sec" in r]
    fail_rows = [r for r in cvx_rows if "time_sec" not in r]

    if success_rows:
        print_dict_table(
            success_rows,
            columns=[
                "solver", "status", "time_sec", "obj", "eq_residual",
                "min_x", "rel_x_diff_vs_barrier", "obj_diff_vs_barrier"
            ],
            float_fmt="{:.6e}"
        )

    if fail_rows:
        print("\nFailed solvers:")
        for row in fail_rows:
            print(row)


def run_part_b():
    print("\n" + "=" * 80)
    print("PART (b): vary m and report total Newton steps")
    print("=" * 80)

    # To reflect the theoretical short-step style scaling,
    # we choose mu = 1 + 1/sqrt(m).
    # Fix n = 5 so that m can range from 10 to 1000.
    n_fixed = 5
    m_values = [10, 20, 50, 100, 200, 400, 600, 800, 1000]
    num_trials = 3

    rows = []

    for m in m_values:
        mu = 1.0 + 1.0 / math.sqrt(m)
        total_steps_list = []
        outer_iters_list = []

        for trial in range(num_trials):
            A, b, c, x0 = generate_lp_feasible(
                n=n_fixed, m=m, seed=11000 + 10 * m + trial
            )

            # Slightly looser tolerances here to keep the scaling experiment stable
            result = barrier_method(
                A=A, b=b, c=c, x0=x0,
                mu=mu,
                eps=1e-1,
                newton_tol=1e-4,
                max_outer=10000
            )

            total_steps_list.append(result["total_newton_steps"])
            outer_iters_list.append(len(result["outer_history"]))

        rows.append({
            "m": m,
            "mu": mu,
            "mean_total_newton_steps": float(np.mean(total_steps_list)),
            "std_total_newton_steps": float(np.std(total_steps_list)),
            "mean_outer_iterations": float(np.mean(outer_iters_list)),
            "sqrt_m": float(math.sqrt(m)),
            "sqrt_m_log2m": float(math.sqrt(m) * math.log2(m))
        })

    print_dict_table(
        rows,
        columns=[
            "m", "mu", "mean_total_newton_steps",
            "std_total_newton_steps", "mean_outer_iterations"
        ],
        title="Scaling results",
        float_fmt="{:.6e}"
    )

    # Fit N ~ a sqrt(m) + b
    x1 = np.array([row["sqrt_m"] for row in rows])
    y = np.array([row["mean_total_newton_steps"] for row in rows])
    coef1 = np.polyfit(x1, y, 1)
    yhat1 = np.polyval(coef1, x1)
    r2_1 = 1.0 - np.sum((y - yhat1) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Fit N ~ a sqrt(m) log2(m) + b
    x2 = np.array([row["sqrt_m_log2m"] for row in rows])
    coef2 = np.polyfit(x2, y, 1)
    yhat2 = np.polyval(coef2, x2)
    r2_2 = 1.0 - np.sum((y - yhat2) ** 2) / np.sum((y - np.mean(y)) ** 2)

    print("\nFit against sqrt(m):")
    print(f"N ≈ {coef1[0]:.6f} * sqrt(m) + {coef1[1]:.6f}")
    print(f"R^2 = {r2_1:.6f}")

    print("\nFit against sqrt(m) * log2(m):")
    print(f"N ≈ {coef2[0]:.6f} * sqrt(m) * log2(m) + {coef2[1]:.6f}")
    print(f"R^2 = {r2_2:.6f}")


if __name__ == "__main__":
    run_part_a()
    run_part_b()