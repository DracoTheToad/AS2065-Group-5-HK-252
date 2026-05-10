from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from utils import *
from solver import *


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

