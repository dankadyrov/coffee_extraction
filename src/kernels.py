import numpy as np

h = 0.25


def kernel(distance) -> float:
    q = distance / h
    sigma = 7 / (478 * np.pi * h ** 2)
    if q <= 1:
        return sigma * ((3 - q) ** 5 - 6 * (2 - q) ** 5 + 15 * (1 - q) ** 5)
    elif q <= 2:
        return sigma * ((3 - q) ** 5 - 6 * (2 - q) ** 5)
    elif q <= 3:
        return sigma * (3 - q) ** 5
    else:
        return 0


def kernel_derivative(x, y) -> np.ndarray[
    float, float]:
    q = np.sqrt(x ** 2 + y ** 2) / h
    dq_dx = x / (h * np.sqrt(x ** 2 + y ** 2))
    dq_dy = y / (h * np.sqrt(x ** 2 + y ** 2))
    sigma = 7 / (478 * np.pi * h ** 2)
    if q <= 1:
        kernel_der_x = sigma * (- 5 * (3 - q) ** 4 + 30 * (2 - q) ** 4 - 75 * (1 - q) ** 4) * dq_dx
        kernel_der_y = sigma * (- 5 * (3 - q) ** 4 + 30 * (2 - q) ** 4 - 75 * (1 - q) ** 4) * dq_dy
    elif q <= 2:
        kernel_der_x = sigma * (- 5 * (3 - q) ** 4 + 30 * (2 - q) ** 4) * dq_dx
        kernel_der_y = sigma * (- 5 * (3 - q) ** 4 + 30 * (2 - q) ** 4) * dq_dy
    elif q <= 3:
        kernel_der_x = sigma * (- 5 * (3 - q) ** 4) * dq_dx
        kernel_der_y = sigma * (- 5 * (3 - q) ** 4) * dq_dy
    else:
        return np.array([0, 0])
    return np.array([kernel_der_x, kernel_der_y])
