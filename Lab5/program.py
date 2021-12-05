import numpy as np
import random
import matplotlib.pyplot as plt

n = 11
step = 1.0


def get_equally_distant_points():
    return np.array(np.arange(-step * (n // 2), step * (n // 2) + 1, step))


def get_points_values():
    return np.array([random.uniform(0, n) for _ in range(n)])


def get_A():
    res = np.array([[0.0 for _ in range(n - 2)] for _ in range(n - 2)])
    res[0][0] = 2 * step / 3
    res[0][1] = step / 6
    res[n - 3][n - 4] = step / 6
    res[n - 3][n - 3] = 2 * step / 3

    for i in range(1, n - 3):
        res[i][i - 1] = step / 6
        res[i][i] = 2 * step / 3
        res[i][i + 1] = step / 6

    return res


def get_H():
    res = np.array([[0.0 for _ in range(n)] for _ in range(n - 2)])

    for i in range(n - 2):
        res[i][i] = 1 / step
        res[i][i + 1] = -2 / step
        res[i][i + 2] = 1 / step

    return res


points = get_equally_distant_points()
values = get_points_values()


def spline():
    m = np.linalg.inv(get_A()) @ get_H() @ values

    def s(x):
        for i in range(1, n):
            if x <= points[i]:
                if i == 1:
                    mI_1 = 0.0
                    mI = m[0]

                elif i == n - 1:
                    mI_1 = m[n - 3]
                    mI = 0.0
                else:
                    mI_1 = m[i - 2]
                    mI = m[i - 1]

                return (mI_1 * (points[i] - x) ** 3) / (6 * step) \
                       + mI * (x - points[i - 1]) ** 3 / (6 * step) \
                       + (values[i - 1] - (mI_1 * step ** 2) / 6) * (points[i] - x) / step + \
                       (values[i] - (mI * step ** 2) / 6) * (x - points[i - 1]) / step

    return s


def get_L(x, f):
    def L(p, i):
        return (f[i - 1] * (p - x[i]) * (p - x[i + 1])) / 2 - \
               f[i] * (p - x[i - 1]) * (p - x[i + 1]) + \
               (f[i + 1] * (p - x[i - 1]) * (p - x[i])) / 2

    return L


if __name__ == "__main__":
    print(points)
    print(values)
    # data = np.round(np.arange(points[0], points[len(points) - 1] + 0.1, 0.1), 1)
    # print(data)
    s = spline()

    for i, (start, stop) in enumerate(zip(points, points[1:])):
        x_hat = np.linspace(start, stop, num=50)
        y_hat = [s(x) for x in x_hat]
        plt.plot(x_hat, y_hat)
    plt.show()

    L = get_L(points, values)
    for i, (start, stop) in enumerate(zip(points[::2], points[2::2])):
        x_hat = np.linspace(start, stop, num=50)
        y_hat = [L(x, 2 * i + 1) for x in x_hat]
        plt.plot(x_hat, y_hat)

    plt.show()
