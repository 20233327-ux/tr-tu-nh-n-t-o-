"""
=============================================================================
Phase 2: Tiền xử lý dữ liệu - Hệ thống phân loại bệnh tim
=============================================================================
Mục đích:
- Xử lý missing values
- Mã hóa biến phân loại
- Chuẩn hóa dữ liệu số
- Chia tập train/test (stratified)
- Kiểm tra cân bằng lớp
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Fix console encoding on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ============================================================================
# CẤU HÌNH
# ============================================================================
DATA_DIR = "data"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================================================================
# TẢI DỮ LIỆU
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 2: TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 70)

    data_path = os.path.join(DATA_DIR, "heart.csv")
    if not os.path.exists(data_path):
        print("❌ Chưa có dữ liệu! Vui lòng chạy 01_eda.py trước.")
        exit(1)

    df = pd.read_csv(data_path)
    print(f"\n📊 Dữ liệu gốc: {df.shape[0]} mẫu × {df.shape[1]} cột")

    # ============================================================================
    # 1. XỬ LÝ MISSING VALUES
    # ============================================================================
    print("\n" + "-" * 50)
    print("1. XỬ LÝ MISSING VALUES")
    print("-" * 50)

    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing > 0:
        print(f"\n⚠️  Tổng missing values: {total_missing}")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                n_missing = df[col].isnull().sum()
                pct = n_missing / len(df) * 100
                print(f"   {col}: {n_missing} ({pct:.1f}%)")

        # Xử lý missing values
        # Biến số: thay bằng median
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Biến phân loại: thay bằng mode
        cat_cols = [c for c in df.columns if c not in numeric_cols]

        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"   → {col}: thay bằng median = {median_val:.2f}")

        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"   → {col}: thay bằng mode = {mode_val}")

        print(f"\n✅ Sau xử lý: {df.isnull().sum().sum()} missing values")
    else:
        print("\n✅ Không có missing values!")

    # ============================================================================
    # 2. CHUYỂN ĐỔI KIỂU DỮ LIỆU
    # ============================================================================
    print("\n" + "-" * 50)
    print("2. CHUYỂN ĐỔI KIỂU DỮ LIỆU")
    print("-" * 50)

    # Đảm bảo tất cả cột đều numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Xử lý lại missing values sau chuyển đổi
    if df.isnull().sum().sum() > 0:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

    # Xác định loại biến
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

    # Chuyển biến phân loại sang int
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(int)

    print(f"\n   Biến số ({len(numeric_features)}): {numeric_features}")
    print(f"   Biến phân loại ({len(categorical_features)}): {categorical_features}")

    # ============================================================================
    # 3. TÁCH FEATURES VÀ TARGET
    # ============================================================================
    print("\n" + "-" * 50)
    print("3. TÁCH FEATURES VÀ TARGET")
    print("-" * 50)

    X = df.drop('target', axis=1)
    y = df['target']

    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Tỷ lệ nhãn: 0={y.value_counts().get(0, 0)}, 1={y.value_counts().get(1, 0)}")

    # ============================================================================
    # 4. CHIA TẬP TRAIN/TEST (Stratified)
    # ============================================================================
    print("\n" + "-" * 50)
    print("4. CHIA TẬP TRAIN/TEST")
    print("-" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"   Train: {X_train.shape[0]} mẫu ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"   Test:  {X_test.shape[0]} mẫu ({TEST_SIZE*100:.0f}%)")
    print(f"\n   Train - Tỷ lệ nhãn:")
    print(f"     0 (Không bệnh): {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    print(f"     1 (Có bệnh):    {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
    print(f"\n   Test - Tỷ lệ nhãn:")
    print(f"     0 (Không bệnh): {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
    print(f"     1 (Có bệnh):    {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")

    # ============================================================================
    # 5. CHUẨN HÓA DỮ LIỆU SỐ
    # ============================================================================
    print("\n" + "-" * 50)
    print("5. CHUẨN HÓA DỮ LIỆU SỐ (StandardScaler)")
    print("-" * 50)

    scaler = StandardScaler()

    # Chỉ chuẩn hóa các biến số
    numeric_in_X = [col for col in numeric_features if col in X_train.columns]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_in_X] = scaler.fit_transform(X_train[numeric_in_X])
    X_test_scaled[numeric_in_X] = scaler.transform(X_test[numeric_in_X])

    print(f"   Đã chuẩn hóa {len(numeric_in_X)} đặc trưng số: {numeric_in_X}")
    print(f"\n   Thống kê sau chuẩn hóa (train):")
    print(f"   {'Đặc trưng':<12s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"   {'-'*44}")
    for col in numeric_in_X:
        vals = X_train_scaled[col]
        print(f"   {col:<12s} {vals.mean():>8.3f} {vals.std():>8.3f} {vals.min():>8.3f} {vals.max():>8.3f}")

    # ============================================================================
    # 6. KIỂM TRA CÂN BẰNG LỚP
    # ============================================================================
    print("\n" + "-" * 50)
    print("6. KIỂM TRA CÂN BẰNG LỚP")
    print("-" * 50)

    ratio = y_train.value_counts().min() / y_train.value_counts().max()
    print(f"   Tỷ lệ cân bằng: {ratio:.3f}")

    if ratio < 0.5:
        print("   ⚠️  Dữ liệu không cân bằng! Cần xem xét sử dụng class_weight hoặc SMOTE")
    else:
        print("   ✅ Dữ liệu tương đối cân bằng, không cần xử lý thêm")

    # ============================================================================
    # 7. LƯU DỮ LIỆU ĐÃ XỬ LÝ
    # ============================================================================
    print("\n" + "-" * 50)
    print("7. LƯU DỮ LIỆU ĐÃ XỬ LÝ")
    print("-" * 50)

    X_train_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    # Lưu scaler params để tái sử dụng
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))

    print(f"   ✅ X_train: {os.path.join(PROCESSED_DIR, 'X_train.csv')}")
    print(f"   ✅ X_test:  {os.path.join(PROCESSED_DIR, 'X_test.csv')}")
    print(f"   ✅ y_train: {os.path.join(PROCESSED_DIR, 'y_train.csv')}")
    print(f"   ✅ y_test:  {os.path.join(PROCESSED_DIR, 'y_test.csv')}")
    print(f"   ✅ Scaler:  {os.path.join('models', 'scaler.pkl')}")

    # ============================================================================
    # TỔNG KẾT
    # ============================================================================
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TIỀN XỬ LÝ DỮ LIỆU!")
    print("=" * 70)
    print(f"\n📊 Tóm tắt:")
    print(f"   - Tổng mẫu: {len(df)}")
    print(f"   - Số đặc trưng: {X.shape[1]}")
    print(f"   - Train/Test: {X_train.shape[0]}/{X_test.shape[0]}")
    print(f"   - Missing values đã xử lý: {total_missing}")
    print(f"   - Chuẩn hóa: StandardScaler trên {len(numeric_in_X)} đặc trưng số")
    print(f"   - Cân bằng lớp: {'OK' if ratio >= 0.5 else 'Cần xử lý'}")
