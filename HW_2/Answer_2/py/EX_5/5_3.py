from pathlib import Path
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


SEED = 0
VAL_RATIO = 0.2
LAMBDAS = np.logspace(-4, 6, 60)


def load_csv_matrix(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",")
    except ValueError:
        return np.loadtxt(path, delimiter=",", skiprows=1)


def load_csv_vector(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=",")
    except ValueError:
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return np.asarray(arr).reshape(-1)


def train_val_split(A: np.ndarray, b: np.ndarray, val_ratio: float, seed: int):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_ratio))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return A[tr_idx], b[tr_idx], A[val_idx], b[val_idx]


def compute_scale(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    scale = A.std(axis=0, ddof=0)
    scale[scale < eps] = 1.0
    return scale


def fit_ridge(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    x = cp.Variable(A.shape[1])
    objective = cp.Minimize(cp.sum_squares(A @ x - b) + lam * cp.sum_squares(x))
    prob = cp.Problem(objective)
    prob.solve(solver=cp.SCS, eps=1e-7, verbose=False)
    return np.asarray(x.value).reshape(-1)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "cal_housing"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    A = load_csv_matrix(data_dir / "train_data.csv")
    b = load_csv_vector(data_dir / "train_target.csv")
    A_test = load_csv_matrix(data_dir / "test_data.csv")
    b_test = load_csv_vector(data_dir / "test_target.csv")

    A_tr, b_tr, A_val, b_val = train_val_split(A, b, VAL_RATIO, SEED)
    scale_tr = compute_scale(A_tr)
    A_tr_std = A_tr / scale_tr

    val_errors = []
    for lam in LAMBDAS:
        z = fit_ridge(A_tr_std, b_tr, float(lam))
        x_val = z / scale_tr
        val_error = float(np.linalg.norm(A_val @ x_val - b_val) ** 2)
        val_errors.append(val_error)

    best_idx = int(np.argmin(val_errors))
    best_lambda = float(LAMBDAS[best_idx])

    scale_full = compute_scale(A)
    A_std = A / scale_full
    z_star = fit_ridge(A_std, b, best_lambda)
    x_star = z_star / scale_full

    train_error = float(np.linalg.norm(A @ x_star - b) ** 2)
    test_error = float(np.linalg.norm(A_test @ x_star - b_test) ** 2)
    l2_norm_sq = float(np.linalg.norm(x_star, 2) ** 2)

    curve_train_errors = []
    curve_l2_norm_sq = []
    curve_val_errors = []
    for lam in LAMBDAS:
        z = fit_ridge(A_std, b, float(lam))
        x_model = z / scale_full
        curve_train_errors.append(float(np.linalg.norm(A @ x_model - b) ** 2))
        curve_l2_norm_sq.append(float(np.linalg.norm(x_model, 2) ** 2))
        curve_val_errors.append(float(np.linalg.norm(A_val @ (fit_ridge(A_tr_std, b_tr, float(lam)) / scale_tr) - b_val) ** 2))

    curve = np.column_stack([LAMBDAS, curve_train_errors, curve_l2_norm_sq, curve_val_errors])
    np.savetxt(
        results_dir / "5_3_curve.csv",
        curve,
        delimiter=",",
        header="lambda,train_error,l2_norm_sq,val_error",
        comments=""
    )
    np.savetxt(results_dir / "5_3_x_ridge.csv", x_star, delimiter=",")

    plt.figure(figsize=(6, 4))
    plt.plot(curve_train_errors, curve_l2_norm_sq, marker="o", linewidth=1)
    plt.scatter([train_error], [l2_norm_sq], marker="x", s=80)
    plt.xlabel(r"$\|Ax-b\|_2^2$")
    plt.ylabel(r"$\|x\|_2^2$")
    plt.title("Ridge trade-off curve")
    plt.tight_layout()
    plt.savefig(results_dir / "5_3_ridge_tradeoff.png", dpi=200)
    plt.close()

    with open(results_dir / "5_3_results.txt", "w", encoding="utf-8") as f:
        f.write(f"best_lambda = {best_lambda}\n")
        f.write(f"train_error = {train_error}\n")
        f.write(f"test_error = {test_error}\n")
        f.write(f"l2_norm_sq = {l2_norm_sq}\n")
        f.write("x_Ridge =\n")
        for val in x_star:
            f.write(f"{val}\n")

    print("best_lambda =", best_lambda)
    print("train_error =", train_error)
    print("test_error =", test_error)
    print("l2_norm_sq =", l2_norm_sq)
    print("x_Ridge =", x_star)


if __name__ == "__main__":
    main()