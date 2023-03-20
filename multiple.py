import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

plt.rcParams['toolbar'] = 'toolmanager'

def system1(x):
    return [np.sin(x[0]) - x[1]**3,
            x[0]**2 + x[1]**2-12]

def system2(x):
    return [(x[0]**2+x[1]**2-7)**3-x[0]**2 * x[1]**3,
            x[1]-x[0]]

def plot_system(system, label, a=-10, b=10):
    x = np.linspace(a, b, 1000)
    y = np.linspace(a, b, 1000)
    X, Y = np.meshgrid(x, y)
    Z1 = system([X, Y])[0]
    Z2 = system([X, Y])[1]

    plt.contour(X, Y, Z1, levels=[0], colors='r')
    plt.contour(X, Y, Z2, levels=[0], colors='b')
    plt.title(label)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axhline(0, color="black", alpha=0.7)
    plt.axvline(0, color="black", alpha=0.7)
    plt.grid(True)

def simple_iteration(system, x0, tol=1e-5, max_iter=10000):
    def jacobian(f, x, eps = 1e-4):
        jac = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            x_plus = np.array(x)
            x_minus = np.array(x)
            x_plus[i] += eps
            x_minus[i] -= eps
            jac[:, i] = (np.array(f(x_plus)) - np.array(f(x_minus))) / (2 * eps)
        return jac

    x = np.array(x0)
    err = x
    for i in range(max_iter):
        dx = np.linalg.solve(jacobian(system, x), -np.array(system(x)))
        x_new = x + dx
        err = np.linalg.norm(x_new - x)
        x = x_new
        if err < tol:
            return x, i + 1, err
    return x, max_iter, err

systems = [system1, system2]
labels = ['System 1', 'System 2']

print("Choose a system of nonlinear equations:")
for i, label in enumerate(labels):
    print(f"{i + 1}: {label}")
choice = int(input("Enter the number of your choice: ")) - 1

plot_system(systems[choice], labels[choice])
plt.show()

tol = float(input("Enter the tolerance: "))

x0 = [float(x) for x in input("Enter the initial approximations x1 and x2 separated by space: ").split()]

x, iterations, err = simple_iteration(systems[choice], x0, tol)
solution = fsolve(systems[choice], x0)

print(f"Solution: x1 = {x[0]}, x2 = {x[1]}")
print(f"Iterations: {iterations}")
print(f"Error vector: {err}")

is_solution_correct = np.allclose(solution, x)
print(f"Is the solution correct? {'Yes' if is_solution_correct else 'No'}")
