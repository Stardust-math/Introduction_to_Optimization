import numpy as np

np.set_printoptions(precision=6, suppress=True)

def f(x):
    return np.sum(x * np.log(x))

def grad(x):
    return np.log(x) + 1.0

def generate_instance(n=100, p=30, seed=20260415):
    rng = np.random.default_rng(seed)
    while True:
        A = rng.standard_normal((p, n))
        if np.linalg.matrix_rank(A) == p:
            break
    # strictly positive initial feasible point
    xhat = rng.uniform(0.1, 1.0, size=n)
    b = A @ xhat
    return A, b, xhat

def newton_eq_entropy(A, b, x0, alpha=0.01, beta=0.5, tol=1e-12, max_iter=100):
    x = x0.copy().astype(float)
    hist = []

    for k in range(max_iter):
        g = grad(x)
        Hinv = x  # since Hessian = diag(1/x), its inverse is diag(x)

        # Solve reduced KKT system:
        # (A diag(x) A^T) w = -A diag(x) g
        M = A @ (Hinv[:, None] * A.T)
        rhs = -A @ (Hinv * g)
        w = np.linalg.solve(M, rhs)

        # Newton step
        dx = -Hinv * (g + A.T @ w)

        # Newton decrement squared
        lam_sq = -g @ dx

        fval = f(x)
        feas = np.linalg.norm(A @ x - b)
        hist.append((k, fval, lam_sq / 2.0, feas))

        if lam_sq / 2.0 <= tol:
            return x, w, hist

        # backtracking line search
        t = 1.0

        # maintain positivity
        neg = dx < 0
        if np.any(neg):
            t = min(t, 0.99 * np.min(-x[neg] / dx[neg]))

        gd = g @ dx
        while True:
            xn = x + t * dx
            if np.all(xn > 0) and f(xn) <= fval + alpha * t * gd:
                break
            t *= beta

        x = xn

    return x, w, hist

if __name__ == "__main__":
    n, p = 100, 30
    A, b, x0 = generate_instance(n=n, p=p, seed=20260415)

    x_star, nu_star, hist = newton_eq_entropy(A, b, x0)

    print("rank(A) =", np.linalg.matrix_rank(A))
    print("iterations =", len(hist))
    print("initial objective =", f(x0))
    print("final objective   =", f(x_star))
    print("feasibility residual =", np.linalg.norm(A @ x_star - b))
    print("stationarity residual (inf-norm) =",
          np.linalg.norm(grad(x_star) + A.T @ nu_star, ord=np.inf))

    print("\nIteration history:")
    print("k    f(x)                lambda^2/2          ||Ax-b||_2")
    for k, fv, dec, feas in hist:
        print(f"{k:<2d}   {fv: .10f}   {dec: .10e}   {feas: .3e}")