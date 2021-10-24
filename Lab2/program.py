import numpy as np


def get_p_permutation(k, l, dim):
    p = np.eye(dim)
    p[[k, l]] = p[[l, k]]

    return p


def get_m(matrix, k, dim):
    m = np.eye(dim)
    for i in range(k, matrix.shape[0]):
        if k == i:
            m[i][k] = 1 / matrix[i][k]
        else:
            m[i][k] = -matrix[i][k] / matrix[k][k]

    return m


def gauss(matrix):
    res = matrix
    for i in range(matrix.shape[0]):
        p = get_p_permutation(i, np.argmax(matrix, 0)[i], matrix.shape[0])
        res = p @ res
        res = get_m(res, i, res.shape[0]) @ res

    coefficients = res[:, 0:4]
    values = res[:, 4]
    return np.linalg.solve(coefficients, values)


def zeidel(coefficients, values, e):
    fault = e + 1
    x_cur = np.zeros(coefficients.shape[0])
    x_next = np.zeros(coefficients.shape[0])
    while fault >= e:
        for i in range(x_next.shape[0]):
            for j in range(x_next.shape[0]):
                if i > j:
                    x_next[i] -= x_next[j] * coefficients[i][j]
                elif i < j:
                    x_next[i] -= x_cur[j] * coefficients[i][j]
                else:
                    x_next[i] += values[i]

                # print(x_next[i])
            x_next[i] /= coefficients[i][i]

        fault = np.amax(np.subtract(np.absolute(x_next), np.absolute(x_cur)))
        # print(fault)
        x_cur = x_next
        x_next = np.zeros(coefficients.shape[0])

    return x_cur


if __name__ == "__main__":
    matrix = np.array([[5, 2, 1, 1, 4], [2, 5, -1, -1, 3], [1, -1, 6, -1, 2], [1, -1, -1, 4, 1]])
    a_matrix = np.array([[5, 2, 1, 1], [2, 5, -1, -1], [1, -1, 6, -1], [1, -1, -1, 4]])
    b_array = np.array([4, 3, 2, 1])

    expected_values = np.linalg.solve(a_matrix, b_array)
    received_values_gauss = gauss(matrix)
    received_values_zeidel = zeidel(a_matrix, b_array, 0.000001)

    print(expected_values)
    print(received_values_gauss)
    print(received_values_zeidel)
