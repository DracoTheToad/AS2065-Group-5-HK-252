import numpy as np


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Giải bằng normal equation
def solve_normal_equation(X, y):
    """
    Giải hệ phương trình Gradient L(w) = 0 
    Công thức: w = (X^T * X)^-1 * X^T * y
    """
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w


# Giải bằng gradient descent
def loss_gradient(
    y: np.ndarray,
    y_hat: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:

    n = len(y)

    if len(y_hat) != n or x.shape[0] != n:
        raise ValueError("Shape mismatch.")

    error = y - y_hat
    dw = -(2 / n) * (x.T @ error)

    return dw


def grad_descent(
    X: np.ndarray, y: np.ndarray,
    max_iter: int = 1000,
    alpha: float = 0.5,
    m_1: float = 0.1,
    tol: float = 1e-7,
    random_state = 42,
) -> np.ndarray:
    np.random.seed(random_state)
    w = np.random.rand(X.shape[1])
    for _ in range(max_iter):
        # Tính loss để so với loss mới sau khi đã cập nhật các hệ số
        y_hat = X @ w
        old_loss = mean_squared_error(y, y_hat)

        # Tính gradient
        gradient = loss_gradient(y, y_hat, X)
        grad_mag_squared = gradient.dot(gradient)

        # Kiểm tra điều kiện dừng
        grad_mag = np.sqrt(grad_mag_squared)
        if grad_mag < tol:
            break

        # Backtrackintg
        t_k = 1.0
        while True: 
            # Tính loss mới
            trial_w = w - t_k * gradient
            trial_y_hat = X @ trial_w
            trial_loss = mean_squared_error(y, trial_y_hat)

            # Xét điều kiện Armijo
            loss_diff = trial_loss - old_loss
            armijo_line = - m_1 * t_k * grad_mag_squared
            if loss_diff <= armijo_line:
                w = trial_w
                break
            t_k = t_k * alpha

    return w
