import numpy as np
import matplotlib.pyplot as plt
import math

# Input equation: x^3 - 3x^2 - 14x - 8 = 0


def input_function(x):
    return x**3 - 3*x**2 - 14*x - 8


#
# func_data = np.arange(-3, 6, 0.01)
# plt.plot(func_data, input_function(func_data))
# plt.show()

# From graph we can see, that largest root of function belongs to [5, 6] interval

def first_der(x):
    return 3*x**2 - 6*x - 14


def second_der(x):
    return 6*x - 6


def estimate_q(left, right):
    m1 = min(first_der(left), first_der(right))
    m2 = max(second_der(left), second_der(right))
    return m2 * (right - left) / (2*m1)


def prior_estimate(left, right, fault):
    return math.floor(math.log2(((right - left) / fault)) / math.log(1/estimate_q(left, right))) + 1


def modified_newton_method(x, fault, x0_der, counter):
    next_x = x - input_function(x) / x0_der

    if input_function(next_x) < fault:
        return next_x, counter
    else:
        return modified_newton_method(next_x, fault, x0_der, counter + 1)


if __name__ == "__main__":
    left_ = 5
    right_ = 6
    fault_ = 10 ** -4

    print(prior_estimate(left_, right_, fault_))

    root, post_estimate = modified_newton_method(right_, fault_, first_der(right_), 1)
    print(root, input_function(root), post_estimate)