import cvxpy as cp

x = cp.Variable(2)

objective = cp.Minimize(9*cp.square(x[0]) + 9*cp.square(x[1]) - 30*x[0] - 72*x[1])
constraints = [
    -2*x[0] - x[1] >= -4,
    x[0] >= 0,
    x[1] >= 0
]

prob = cp.Problem(objective, constraints)
prob.solve()

print("optimal value =", prob.value)
print("optimal x =", x.value)