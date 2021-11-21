import math

import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt


def test_function1(x):
    return 3.0 ** x


def test_function2(x):
    return math.sin(x)


def get_chebyshev_nulls_points(n, a, b):
    return np.array([(a + b) / 2 + (b - a) * math.cos((2 * k + 1) * math.pi / (2 * n)) / 2 for k in range(n)])


def get_equally_distant_points(n, step):
    return np.array(np.arange(-step * (n // 2), step * (n // 2) + 1, step))


def w(i, x_points):
    res = 1
    x_k = x_points[i]
    for j in range(len(x_points)):
        if j != i:
            res *= x_k - x_points[j]
    print("Denominator: ", res)
    return res


def interpolate_i_point(i, x_points):
    coefficients = np.zeros(x_points.shape[0])
    coefficients[0] = 1.0 / w(i, x_points)

    for k in range(x_points.shape[0]):
        if k != i:
            print(k)
            new_coefficients = np.zeros(x_points.shape[0])
            for j in range(x_points.shape[0] - 1):
                new_coefficients[j + 1] += coefficients[j]
                new_coefficients[j] -= x_points[k] * coefficients[j]

            coefficients = new_coefficients

            print(coefficients)
    return coefficients


def lagrange(x_points, y_points):
    polynomial = np.zeros(x_points.shape[0])
    for i in range(x_points.shape[0]):
        coefficients = interpolate_i_point(i, x_points)
        for k in range(x_points.shape[0]):
            polynomial[k] += y_points[i] * coefficients[k]

    return polynomial


if __name__ == "__main__":
    N = 11
    function = test_function2
    ch_points = get_chebyshev_nulls_points(N, -5, 5)
    eq_d_points = get_equally_distant_points(N, 1)
    ch_values = np.array([function(x) for x in ch_points])
    eq_d_values = np.array([function(x) for x in eq_d_points])
    print(ch_points, eq_d_points)
    print(ch_values, eq_d_values)
    ch_coefficients = lagrange(ch_points, ch_values)
    eq_d_coefficients = lagrange(eq_d_points, eq_d_values)

    L_eq = poly.Polynomial(ch_coefficients)
    L_ch = poly.Polynomial(eq_d_coefficients)
    x_new = np.arange(-5, 5, 0.1)

    fig, axs = plt.subplots(3, figsize=(10, 8))
    axs[0].plot(x_new, L_eq(x_new), 'b', label='Equally distant')
    axs[1].plot(x_new, L_ch(x_new), 'g', label='Chebyshev')
    axs[2].plot(x_new, np.array([function(x) for x in x_new]), 'r', label='Function')
    fig.suptitle('Lagrange Polynomials')
    plt.show()
