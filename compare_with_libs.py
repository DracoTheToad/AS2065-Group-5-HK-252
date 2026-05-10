from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from utils import *
from solvers import *


if __name__ == "__main__":
    # Chuẩn bị dữ liệu
    X, y, X_raw, feature_names = prepare_data_multivar()

    # So sánh normal equation với sklearn
    w_normal = solve_normal_equation(X, y)
    normal_comp_df, w_sklearn = compare_methods(
        X, y, w_normal,
        feature_names, 'Normal Equation'
    )

    print("--- BẢNG SO SÁNH HỆ SỐ (Normal equation vs Scikit-learn) ---")
    print(normal_comp_df.to_string(index=False))

    y_pred_grad = X.dot(w_normal)
    mse_grad = mean_squared_error(y, y_pred_grad)
    print(f"\nMSE (Normal Equation): {mse_grad:.10f}")

    visualize_comparison(normal_comp_df, 'Normal Equation')

    # So sánh gradient descent với sklearn
    scaler = StandardScaler()
    X_raw = scaler.fit_transform(X_raw, y)
    X_scaled = np.c_[np.ones(X_raw.shape[0]), X_raw]

    w_grad = grad_descent(X_scaled, y)
    grad_comp_df, w_sklearn = compare_methods(
        X_scaled, y, w_grad,
        feature_names, 'Gradient Descent'
    )

    print("--- BẢNG SO SÁNH HỆ SỐ (Gradient descent backtracking vs Scikit-learn) ---")
    print(grad_comp_df.to_string(index=False))

    y_pred_grad = X_scaled.dot(w_grad)
    mse_grad = mean_squared_error(y, y_pred_grad)
    print(f"\nMSE (Gradient descent): {mse_grad:.10f}")

    visualize_comparison(grad_comp_df, 'Gradient Descent')
