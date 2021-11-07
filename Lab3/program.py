import math

import numpy as np
from scipy import optimize
"""
Modified Newton method was impossible to apply for system in first version of problem, so equation set was modified:
    2x - 1.5y^2 + z^2 - 5 = 0
    6xyz - x + 5y + 3z + 1 = 0
    5xz - z - 1 = 0
"""


def lecture_example_a(x):
    return np.array([[1 - 0.25 * math.cos(x[0] - x[1]) / 2, 0.25 * math.cos(x[0] + x[1]) / 2],
                  [0.25 * math.sin(x[0] + x[1]) / 2, 1 + 0.25 * math.sin(x[0] + x[1]) / 2]])


def a_derivatives(x):
    a = np.array([[2, 3 * x[1], 2 * x[2]],
                 [6 * x[1] * x[2] - 1, 6 * x[0] * x[2] + 5, 6 * x[0] * x[1] + 3],
                 [5 * x[2], 0, 5 * x[0] - 1]])
    return np.linalg.inv(a)


def lecture_example(x):
    res = np.array([x[0] - 0.5 * math.sin(x[0] - x[1]) / 2,
                    x[1] - 0.5 * math.cos(x[0] + x[1]) / 2])
    return res


def lab_terms_function(x):
    res = np.array([2.0 * x[0] - 1.5 * x[1] ** 2 + x[2] ** 2 - 5.0,
                    6.0 * x[0] * x[1] * x[2] - x[0] + 5.0 * x[1] + 3.0 * x[2] + 1,
                    5.0 * x[0] * x[2] - x[2] - 1.0])

    # print("values", res)
    return res


def modified_newton_method(system_function, derivative_function, e):
    a = derivative_function([0.0, 0.0])
    print(a)
    fault = e + 1
    x_cur = np.array([0, 0])
    while fault >= e:
        x_next = np.subtract(x_cur, a @ system_function(x_cur))
        fault = np.amax(np.absolute(np.subtract(np.absolute(x_next), np.absolute(x_cur))))
        x_cur = x_next
        print("x_cur", x_cur)

    return x_cur


def eigenvalues_scalar(a, epsilon):
    fault = epsilon + 1
    x_cur = np.ones(a.shape[0])
    x_next = a @ x_cur
    l_cur = np.dot(x_next, x_cur) / np.dot(x_cur, x_cur)
    x_cur = x_next
    # print("value", l_cur)

    while fault >= epsilon:
        e = x_cur / np.linalg.norm(x_cur)
        x_cur = a @ e
        l_next = np.dot(x_cur, e) / np.dot(e, e)
        fault = np.abs(l_next - l_cur)
        l_cur = l_next
        # print("value", l_cur)

    return l_cur


if __name__ == '__main__':
    print(modified_newton_method(lab_terms_function, a_derivatives, 0.0001))
    print(optimize.newton_krylov(lab_terms_function, [0, 0]))
    # a_matrix = np.array([[5, 1, 2], [1, 4, 1], [2, 1, 3]])
    # a_matrix = np.array([[5, 2, 2, 1], [2, 5, -1, -1], [1, -1, 6, -1], [1, -1, -1, 4]])
    # print(eigenvalues_scalar(a_matrix, 0.000001))
    # w, v = np.linalg.eig(a_matrix)
    # print(w)



