import numpy as np


# Giải bằng normal equation
def solve_normal_equation(X, y):
    """
    Giải hệ phương trình Gradient L(w) = 0 
    Công thức: w = (X^T * X)^-1 * X^T * y
    """
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w

