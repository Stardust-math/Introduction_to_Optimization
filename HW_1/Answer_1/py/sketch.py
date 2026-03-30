import numpy as np
import matplotlib.pyplot as plt

# f(x) = x^2 + 0.2x^4
def f(x):
    return x**2 + 0.2*x**4

# x0 = 0.5
x0 = 0.5

# f(x0) = x0^2 + 0.2x0^4
f0 = f(x0)

# f'(x) = 2x + 0.8x^3, so f'(x0) = 2x0 + 0.8x0^3
grad0 = 2*x0 + 0.8*x0**3

# l(x) = f(x0) + f'(x0)(x - x0)
def linear(x):
    return f0 + grad0 * (x - x0)

# m = 1, L = 4
m = 1.0
L = 4.0

x = np.linspace(-1.5, 1.5, 400)

# l(x) + (L/2)(x - x0)^2
upper = linear(x) + 0.5 * L * (x - x0)**2

# l(x) + (m/2)(x - x0)^2
lower = linear(x) + 0.5 * m * (x - x0)**2

plt.figure(figsize=(8, 5))
plt.plot(x, f(x), label=r'$f(x)=x^2+0.2x^4$')
plt.plot(x, linear(x), label=r'$f(x_0)+f^\prime(x_0)(x-x_0)$')
plt.plot(x, upper, label=r'$f(x_0)+f^\prime(x_0)(x-x_0)+\frac{L}{2}(x-x_0)^2$')
plt.plot(x, lower, label=r'$f(x_0)+f^\prime(x_0)(x-x_0)+\frac{m}{2}(x-x_0)^2$')
plt.scatter([x0], [f0], label=r'$x_0$')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sketch for parts (a) and (b)')
plt.grid(True)
plt.show()