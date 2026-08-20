import numpy as np
import pandas as pd
import time
from pathlib import Path
from numpy.linalg import norm, solve, eigvalsh
from scipy.optimize import minimize
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

OUTDIR = Path("results_opt_hw")
OUTDIR.mkdir(exist_ok=True)

# ============================================================
# Data
# ============================================================

# Linear regression dataset: diabetes (real regression dataset in scikit-learn)
lin = load_diabetes()
X_lin_raw, b_lin = lin.data, lin.target.astype(float)

# Standardize features and append a constant 1 to model an intercept
X_lin = StandardScaler().fit_transform(X_lin_raw)
A_lin = np.hstack([X_lin, np.ones((X_lin.shape[0], 1))])

# Logistic regression dataset: Breast Cancer Wisconsin (Diagnostic)
logd = load_breast_cancer()
X_log_raw = logd.data
y_log = logd.target.astype(float)  # 0/1 labels
X_log = StandardScaler().fit_transform(X_log_raw)
A_log = np.hstack([X_log, np.ones((X_log.shape[0], 1))])

n_lin, d_lin = A_lin.shape
n_log, d_log = A_log.shape

# Regularization:
#   linear: not needed, because A has full column rank after augmentation
#   logistic: added to guarantee global strong convexity
gamma_lin = 0.0
gamma_log = 1.0

# ============================================================
# Objectives
# ============================================================

def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def f_lin(x):
    r = A_lin @ x - b_lin
    return float(r @ r + 0.5 * gamma_lin * (x @ x))

def g_lin(x):
    r = A_lin @ x - b_lin
    return 2 * A_lin.T @ r + gamma_lin * x

def H_lin(_x):
    return 2 * A_lin.T @ A_lin + gamma_lin * np.eye(d_lin)

def f_log(x):
    z = A_log @ x
    return float(np.sum(np.logaddexp(0.0, z) - (1.0 - y_log) * z) + 0.5 * gamma_log * (x @ x))

def g_log(x):
    z = A_log @ x
    s = sigmoid(z)
    return A_log.T @ (s - (1.0 - y_log)) + gamma_log * x

def H_log(x):
    z = A_log @ x
    s = sigmoid(z)
    w = s * (1.0 - s)
    return A_log.T @ (A_log * w[:, None]) + gamma_log * np.eye(d_log)

# ============================================================
# Lipschitz / strong convexity constants
# ============================================================

eig_lin = eigvalsh(2 * A_lin.T @ A_lin + gamma_lin * np.eye(d_lin))
L_lin = eig_lin[-1]
m_lin = eig_lin[0]

eig_log_AtA = eigvalsh(A_log.T @ A_log)
L_log = 0.25 * eig_log_AtA[-1] + gamma_log
m_log = gamma_log

# ============================================================
# CVXPY reference solution (if available)
# ============================================================

def solve_with_reference():
    try:
        import cvxpy as cp

        # linear
        x_lin = cp.Variable(d_lin)
        obj_lin = cp.sum_squares(A_lin @ x_lin - b_lin) + 0.5 * gamma_lin * cp.sum_squares(x_lin)
        prob_lin = cp.Problem(cp.Minimize(obj_lin))
        prob_lin.solve(solver=cp.SCS, verbose=False)
        x_star_lin = np.asarray(x_lin.value).reshape(-1)
        p_star_lin = float(prob_lin.value)

        # logistic
        x_log = cp.Variable(d_log)
        z = A_log @ x_log
        obj_log = cp.sum(cp.logistic(z) - cp.multiply(1.0 - y_log, z)) + 0.5 * gamma_log * cp.sum_squares(x_log)
        prob_log = cp.Problem(cp.Minimize(obj_log))
        prob_log.solve(solver=cp.SCS, verbose=False)
        x_star_log = np.asarray(x_log.value).reshape(-1)
        p_star_log = float(prob_log.value)

        print("Reference solver: CVXPY")
        return x_star_lin, p_star_lin, x_star_log, p_star_log

    except Exception as e:
        print("CVXPY unavailable, using analytic/high-precision fallback.")
        print("Reason:", e)

        # Linear regression has a closed-form solution
        x_star_lin = solve(H_lin(None), 2 * A_lin.T @ b_lin)
        p_star_lin = f_lin(x_star_lin)

        # High-precision trust-region solve for logistic
        res = minimize(
            f_log, np.zeros(d_log), jac=g_log, hess=H_log, method="trust-exact",
            options={"gtol": 1e-14, "maxiter": 200}
        )
        x_star_log = res.x
        p_star_log = f_log(x_star_log)
        return x_star_lin, p_star_lin, x_star_log, p_star_log

# ============================================================
# Algorithms
# ============================================================

def run_gd(f, g, x0, x_star, p_star, L, maxit=50000, tol_rmse=1e-12):
    x = x0.copy()
    xs = [x.copy()]
    fs = [f(x)]
    ts = [0.0]
    t0 = time.perf_counter()

    for _ in range(maxit):
        x = x - (1.0 / L) * g(x)
        xs.append(x.copy())
        fs.append(f(x))
        ts.append(time.perf_counter() - t0)
        rmse = norm(x - x_star) / np.sqrt(x_star.size)
        if rmse <= tol_rmse:
            break

    return {"xs": xs, "fs": np.array(fs), "ts": np.array(ts)}

def run_newton(f, g, H, x0, x_star, p_star, maxit=100, tol_rmse=1e-12, alpha=1e-4, beta=0.5):
    x = x0.copy()
    xs = [x.copy()]
    fs = [f(x)]
    ts = [0.0]
    t0 = time.perf_counter()

    for _ in range(maxit):
        grad = g(x)
        hess = H(x)
        d = solve(hess, -grad)
        fx = fs[-1]
        gd = grad @ d

        t = 1.0
        while f(x + t * d) > fx + alpha * t * gd:
            t *= beta
            if t < 1e-16:
                break

        x = x + t * d
        xs.append(x.copy())
        fs.append(f(x))
        ts.append(time.perf_counter() - t0)

        rmse = norm(x - x_star) / np.sqrt(x_star.size)
        if rmse <= tol_rmse:
            break

    return {"xs": xs, "fs": np.array(fs), "ts": np.array(ts)}

def threshold_stats(run, x_star, eps_list=(1e-2, 1e-3, 1e-4)):
    rows = []
    for eps in eps_list:
        for k, x in enumerate(run["xs"]):
            rmse = norm(x - x_star) / np.sqrt(x_star.size)
            if rmse <= eps:
                rows.append({
                    "epsilon": eps,
                    "iterations": int(k),
                    "time_sec": float(run["ts"][k]),
                    "rmse_at_hit": float(rmse),
                })
                break
    return pd.DataFrame(rows)

def metrics(run, x_star, p_star):
    dist = np.array([norm(x - x_star) for x in run["xs"]], dtype=float)
    gap = np.array(run["fs"] - p_star, dtype=float)
    return np.maximum(dist, 1e-16), np.maximum(gap, 1e-16)

def save_plots(problem_name, gd_run, nt_run, x_star, p_star):
    gd_dist, gd_gap = metrics(gd_run, x_star, p_star)
    nt_dist, nt_gap = metrics(nt_run, x_star, p_star)

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(gd_dist)), gd_dist, label="Gradient descent")
    plt.plot(range(len(nt_dist)), nt_dist, label="Newton")
    plt.yscale("log")
    plt.xlabel("Iteration k")
    plt.ylabel(r"$\|x^{(k)} - x^\star\|_2$")
    plt.title(f"{problem_name}: distance to optimum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{problem_name.lower().replace(' ', '_')}_distance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(gd_gap)), gd_gap, label="Gradient descent")
    plt.plot(range(len(nt_gap)), nt_gap, label="Newton")
    plt.yscale("log")
    plt.xlabel("Iteration k")
    plt.ylabel(r"$f(x^{(k)}) - p^\star$")
    plt.title(f"{problem_name}: objective gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{problem_name.lower().replace(' ', '_')}_gap.png", dpi=200)
    plt.close()

def main():
    x_star_lin, p_star_lin, x_star_log, p_star_log = solve_with_reference()

    gd_lin = run_gd(f_lin, g_lin, np.zeros(d_lin), x_star_lin, p_star_lin, L_lin, maxit=20000)
    nt_lin = run_newton(f_lin, g_lin, H_lin, np.zeros(d_lin), x_star_lin, p_star_lin, maxit=20)

    gd_log = run_gd(f_log, g_log, np.zeros(d_log), x_star_log, p_star_log, L_log, maxit=50000)
    nt_log = run_newton(f_log, g_log, H_log, np.zeros(d_log), x_star_log, p_star_log, maxit=100)

    save_plots("Linear regression", gd_lin, nt_lin, x_star_lin, p_star_lin)
    save_plots("Logistic regression", gd_log, nt_log, x_star_log, p_star_log)

    summary = pd.DataFrame([
        {
            "problem": "Linear regression",
            "samples": n_lin,
            "dimension": d_lin,
            "gamma": gamma_lin,
            "L": float(L_lin),
            "m": float(m_lin),
            "condition_number_L_over_m": float(L_lin / m_lin),
            "p_star": float(p_star_lin),
        },
        {
            "problem": "Logistic regression",
            "samples": n_log,
            "dimension": d_log,
            "gamma": gamma_log,
            "L": float(L_log),
            "m": float(m_log),
            "condition_number_L_over_m": float(L_log / m_log),
            "p_star": float(p_star_log),
        },
    ])
    summary.to_csv(OUTDIR / "summary_table.csv", index=False)

    runtime = pd.concat([
        threshold_stats(gd_lin, x_star_lin).assign(problem_method="Linear-GD"),
        threshold_stats(nt_lin, x_star_lin).assign(problem_method="Linear-Newton"),
        threshold_stats(gd_log, x_star_log).assign(problem_method="Logistic-GD"),
        threshold_stats(nt_log, x_star_log).assign(problem_method="Logistic-Newton"),
    ], ignore_index=True)
    runtime.to_csv(OUTDIR / "runtime_table.csv", index=False)

    print("\nSummary")
    print(summary.to_string(index=False))

    print("\nRuntime to reach RMSE thresholds")
    print(runtime[["problem_method", "epsilon", "iterations", "time_sec", "rmse_at_hit"]].to_string(index=False))

if __name__ == "__main__":
    main()
