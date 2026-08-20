from pathlib import Path
import numpy as np
import cvxpy as cp


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


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "cal_housing"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    A = load_csv_matrix(data_dir / "train_data.csv")
    b = load_csv_vector(data_dir / "train_target.csv")
    A_test = load_csv_matrix(data_dir / "test_data.csv")
    b_test = load_csv_vector(data_dir / "test_target.csv")

    x = cp.Variable(A.shape[1])

    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    prob = cp.Problem(objective)
    prob.solve()

    x_star = np.asarray(x.value).reshape(-1)
    train_error = float(np.linalg.norm(A @ x_star - b) ** 2)
    test_error = float(np.linalg.norm(A_test @ x_star - b_test) ** 2)

    np.savetxt(results_dir / "5_1_x_ls.csv", x_star, delimiter=",")

    with open(results_dir / "5_1_results.txt", "w", encoding="utf-8") as f:
        f.write(f"train_error = {train_error}\n")
        f.write(f"test_error = {test_error}\n")
        f.write("x_LS =\n")
        for val in x_star:
            f.write(f"{val}\n")

    print("train_error =", train_error)
    print("test_error =", test_error)
    print("x_LS =", x_star)


if __name__ == "__main__":
    main()