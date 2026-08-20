import cvxpy as cp

x = cp.Variable(4)

objective = cp.Minimize(x[0] + 2*x[1] + 3*x[2] - x[3])
constraints = [
    x[0] - x[1] + x[2] - 2*x[3] <= 6,
    2*x[0] + x[1] - x[2] >= 2,
    -x[0] + x[1] - x[2] + x[3] >= 8,
    0 <= x[0], x[0] <= 3,
    1 <= x[1], x[1] <= 4,
    0 <= x[2], x[2] <= 10,
    2 <= x[3], x[3] <= 5
]

prob = cp.Problem(objective, constraints)
prob.solve()

print("optimal value =", prob.value)
print("optimal x =", x.value)