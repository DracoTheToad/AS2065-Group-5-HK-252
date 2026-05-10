import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler

from utils import *
from solver import *

# Chuẩn bị dữ liệu
data = fetch_california_housing()
X_raw = data.data
y = data.target

# Chuẩn hóa dữ liệu (Gradient Descent bắt buộc cần chuẩn hóa để hội tụ)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_with_bias = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

# Đo thời gian Phương trình Chuẩn tắc (Giải trực tiếp ma trận)
start_time_normal = time.time()
# w = (X^T X)^-1 X^T y
w_normal = solve_normal_equation(X_with_bias, y)
time_normal = time.time() - start_time_normal

# Đo thời gian giải trực tiếp bằng sklearn
sk_lin_model = LinearRegression()
start_time_sk_lin = time.time()
sk_lin_model.fit(X_scaled, y)
time_sk_lin = time.time() - start_time_sk_lin

# Đo thời gian Gradient Descent (Phương pháp lặp)
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
start_time_sgd_reg = time.time()
sgd_reg.fit(X_scaled, y)
time_sgd = time.time() - start_time_sgd_reg

# Trực quan hóa hiệu suất
methods = ['Phương trình Chuẩn tắc', 'Gradient Descent', 'Giải trực tiếp (sklearn)', 'SGD (sklearn)']
times = [time_normal, time_grad_backtrack, time_sk_lin, time_sgd]

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, times, color=['#2ecc71', '#e74c3c'], width=0.5)

plt.ylabel('Thời gian thực thi (giây)')
plt.title('So sánh hiệu suất')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Thêm chú thích về kết quả
print(f"Thời gian Normal Equation: {time_normal:.6f}s")
print(f"Thời gian Normal Equation: {time_sk_lin:.6f}s")
print(f"Thời gian Gradient Descent: {time_sgd:.6f}s")

plt.show()
