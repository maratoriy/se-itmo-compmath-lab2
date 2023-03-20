from enum import Enum

import matplotlib.pyplot as plt
import numpy as np


def f1(x):
    return x ** 3 + 2.28 * x ** 2 - 1.934 * x - 3.907


def f2(x):
    return np.sin(x) - x / 2


def f3(x):
    return np.exp(-x) - x


def df1(x):
    return 3 * x ** 2 + 4.56 * x - 1.934


def df2(x):
    return np.cos(x) - 0.5


def df3(x):
    return -np.exp(-x) - 1


def ddf1(x):
    return 6 * x + 4.56


def ddf2(x):
    return -np.sin(x)


def ddf3(x):
    return np.exp(-x)


def number_of_roots(f, a, b, step=0.01):
    i = a
    roots = []
    while (i < b):
        if f(i - step) * f(i) < 0:
            roots.append((2 * i - step) / 2)
        i += step
    return roots


def plot_function(f, a, b):
    a = (1.1 * a if a < 0 else a / 1.1)
    b = (1.1 * b if b > 0 else b / 1.1)

    x = np.linspace(a, b, 1000)
    y = f(x)
    plt.axhline(0, color="black", alpha=0.7)
    plt.axvline(0, color="black", alpha=0.7)
    plt.xlim(a, b)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Function plot")
    plt.grid(True)

    plt.show()


def secant(f, x0, x1, tol, max_iter):
    for i in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x2 - x1) < tol:
            return x2, i + 1
        x0, x1 = x1, x2
    return x1, max_iter


def bisection(f, a, b, tol, max_iter):
    for i in range(max_iter):
        x = (a + b) / 2
        if abs(a - b) < tol:
            return x, i + 1
        if f(a) * f(x) < 0:
            b = x
        else:
            a = x

    return (a + b) / 2, max_iter


def simple_iteration(f, x0, l, tol, max_iter):
    for i in range(max_iter):
        x = x0 + l * f(x0)
        if abs(x - x0) < tol:
            return x, i + 1
        x0 = x
    return x0, max_iter


functions = [f1, f2, f3]
derivatives = [df1, df2, df3]
second_derivatives = [ddf1, ddf2, ddf3]
function_names = ["x^3+2.28x^2-1.934x-3.907", "sin(x) - x/2", "e^(-x) - x"]

print("Select the equation to solve:")
for i, name in enumerate(function_names):
    print(f"{i + 1}. {name}")

choice = int(input("Enter the number of the equation: ")) - 1
f = functions[choice]
df = derivatives[choice]
ddf = second_derivatives[choice]

a, b = map(float, input("Enter the interval bounds (a, b): ").split())

if a >= b:
    print("a should be greater than b")
    exit(4)

first_roots = number_of_roots(f, a, b)
roots_num = len(first_roots)

plot_function(f, a, b)

if roots_num == 0:
    print("Number of roots is zero")
    exit(1)

if roots_num > 1:
    print("Number of roots more than 1")
    exit(1)

tol = float(input("Enter the tolerance: "))

Method = Enum('Method', ['Bisection', 'Secant', 'Simple_iteration'])
methods = [Method.Bisection, Method.Secant, Method.Simple_iteration]

print("Select the method to use:")
for i, method in enumerate(methods):
    print(f"{i + 1}. {method.name}")

choice = int(input("Enter the number of the method: ")) - 1

match methods[choice]:
    case Method.Bisection:
        result, iters = bisection(f, a, b, tol, 10000)
        print(f"Result: {result}, done in {iters} iterations")

    case Method.Secant:
        if f(a) * f(b) > 0:
            print("f(a)*f(b) > 0, then no convergence")

        x0 = (a if f(a) * ddf(a) > 0 else b)

        x1 = float(input("Enter the second initial value (x_1) for method: "))
        if x1 < a or x1 > b:
            print("Entered value is outer of [a,b]")
            exit(2)

        result, iters = secant(f, x0, x1, tol, 10000)
        print(f"Result: {result}, done in {iters} iterations")

    case Method.Simple_iteration:
        l = -1 / max(df(a), df(b))
        result, iters = simple_iteration(f, a, l, tol, 10000)
        print(f"Result: {result}, done in {iters} iterations")

plot_function(f, a, b)
