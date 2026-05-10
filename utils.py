import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression


# Load dữ liệu 1 biến
def prepare_data_1_var():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target # Giá nhà trung bình (đơn vị 100,000$)
    
    # Để minh họa đồ thị 2D/3D dễ hiểu, ta chọn đặc trưng quan trọng nhất: 'MedInc' (Thu nhập trung bình)
    # Tuy nhiên, hàm tính toán vẫn sẽ hỗ trợ đa biến.
    X_raw = df[['MedInc']].values 
    
    # Thêm cột 1 vào ma trận X để tính toán Intercept (w0) 
    X = np.c_[np.ones(X_raw.shape[0]), X_raw]
    return X, y, X_raw

# Thêm dữ liệu
def prepare_data_multivar():
    """Tải dữ liệu và thêm cột bias (cột 1) vào ma trận X."""
    data = fetch_california_housing()
    X_raw = data.data
    y = data.target
    feature_names = data.feature_names

    # Thêm cột 1 để tính toán Intercept w0
    X = np.c_[np.ones(X_raw.shape[0]), X_raw]
    return X, y, X_raw, feature_names

# So sánh phương pháp
def compare_methods(X, y, w_normal, feature_names, method_name):
    """So sánh kết quả với Scikit-learn."""
    # 1. Kết quả từ Scikit-learn
    # Lưu ý: X ở đây không cần cột 1 vì sklearn tự thêm intercept
    model = LinearRegression()
    model.fit(X[:, 1:], y) 
    w_sklearn = np.insert(model.coef_, 0, model.intercept_)

    # 2. Tạo DataFrame so sánh trọng số
    comparison_df = pd.DataFrame({
        'Feature': ['Intercept'] + list(feature_names),
        method_name: w_normal,
        'Scikit-learn': w_sklearn,
        'Difference': np.abs(w_normal - w_sklearn)
    })
    
    return comparison_df, w_sklearn


# Trực quan hóa
# Trực quan hóa sự khác biệt các hệ số
def visualize_comparison(comparison_df, method_name):
    """Trực quan hóa sự tương đồng giữa các hệ số."""
    plt.figure(figsize=(12, 6))
    
    # Vẽ biểu đồ cột so sánh trọng số (trừ Intercept vì giá trị thường quá lớn)
    df_plot = comparison_df.iloc[1:] 
    
    x = np.arange(len(df_plot))
    width = 0.35

    plt.bar(x - width/2, df_plot[method_name], width, label=method_name, color='skyblue')
    plt.bar(x + width/2, df_plot['Scikit-learn'], width, label='Scikit-learn', color='salmon', alpha=0.7)

    plt.xlabel('Các đặc trưng (Features)')
    plt.ylabel('Giá trị trọng số (Weights)')
    plt.title('So sánh hệ số hồi quy giữa 2 phương pháp')
    plt.xticks(x, df_plot['Feature'], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Trực quan hóa sự lồi của hàm loss MSE
def plot_convexity(X, y, w_opt):
    # Tạo lưới cho w0 (intercept) và w1 (slope của MedInc)
    w0_vals = np.linspace(w_opt[0] - 2, w_opt[0] + 2, 50)
    w1_vals = np.linspace(w_opt[1] - 1, w_opt[1] + 1, 50)
    W0, W1 = np.meshgrid(w0_vals, w1_vals)
    
    # Tính Loss (MSE) tại mỗi điểm trên lưới
    # L(w) = (1/2n) * ||Xw - y||^2 
    Z = np.array([
        (1 / (2 * len(y))) * np.sum((X.dot(np.array([w0, w1])) - y)**2)
        for w0, w1 in zip(np.ravel(W0), np.ravel(W1))
    ]).reshape(W0.shape)

    fig = plt.figure(figsize=(12, 6))
    
    # Biểu đồ bề mặt 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(W0, W1, Z, cmap='terrain', alpha=0.8)
    ax1.scatter(w_opt[0], w_opt[1], (1/(2*len(y)))*np.sum((X.dot(w_opt)-y)**2), color='red', s=50)
    ax1.set_title("Bề mặt hàm mất mát (Lồi)")
    ax1.set_xlabel("w0 (Intercept)")
    ax1.set_ylabel("w1 (MedInc Weight)")

    # Biểu đồ đường đồng mức (Contour)
    ax2 = fig.add_subplot(122)
    cp = ax2.contour(W0, W1, Z, levels=30)
    ax2.clabel(cp, inline=1, fontsize=10)
    ax2.plot(w_opt[0], w_opt[1], 'ro') # Điểm tối ưu toàn cục
    ax2.set_title("Đường đồng mức và Nghiệm tối ưu")
    
    plt.tight_layout()
    plt.show()
