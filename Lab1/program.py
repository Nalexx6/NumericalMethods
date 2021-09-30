import numpy as np
import matplotlib.pyplot as plt
import math

# Input equation: x^3 - 3x^2 - 14x - 8 = 0


def input_function(x):
    return x**3 - 3*x**2 - 14*x - 8


def first_der(x):
    return 3*x**2 - 6*x - 14


def second_der(x):
    return 6*x - 6


def estimate_q(left, right):
    m1 = min(first_der(left), first_der(right))
    m2 = max(second_der(left), second_der(right))
    return m2 * (right - left) / (2*m1)


def prior_estimate(left, right, fault, q):
    return math.floor(math.log2(((right - left) / fault)) / math.log(1/q)) + 1


def newton_method(x, fault, counter):
    if abs(input_function(x)) < fault:
        return x, counter

    next_x = x - input_function(x) / first_der(x)
    return newton_method(next_x, fault, counter + 1)


def modified_newton_method(x, fault, x0_der, counter):
    if abs(input_function(x)) < fault:
        return x, counter

    next_x = x - input_function(x) / x0_der
    return modified_newton_method(next_x, fault, x0_der, counter + 1)


if __name__ == "__main__":
    left_ = 5.4
    right_ = 6
    fault_ = 10 ** -5
    q_ = estimate_q(left_, right_)

    print("q =", q_)
    print("prior estimate =", prior_estimate(left_, right_, fault_, q_))

    root, post_estimate = newton_method(right_ - (right_ - left_) / 2, fault_, 0)
    print("----Newton method----")
    print("x* =", root, " | f(x*) =", input_function(root), " | post-estimate =", post_estimate)

    print("----modified Newton method----")
    root_mod, post_estimate_mod = modified_newton_method(right_ - (right_ - left_) / 2, fault_, first_der(right_), 0)
    print("x* =", root_mod, " | f(x*) =", input_function(root_mod), " | post-estimate =", post_estimate_mod)

    #
    # func_data = np.arange(-3, 6, 0.01)
    # plt.plot(func_data, input_function(func_data))
    # plt.show()

    # From graph we can see, that largest root of function belongs to [5, 6] interval
