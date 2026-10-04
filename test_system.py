"""
=============================================================================
HỆ THỐNG KIỂM THỬ TỰ ĐỘNG - CARDIOVASCULAR AI DIAGNOSIS SYSTEM
=============================================================================
Kiểm thử toàn diện tính toàn vẹn của:
1. Pipeline dữ liệu (Raw data & Processed data)
2. 6 Mô hình Machine Learning & Scaler
3. Khả năng dự đoán và ngưỡng xác suất
4. Thư viện biểu đồ & Tài nguyên báo cáo
5. Khả năng tương thích nền tảng (Windows / UTF-8)
=============================================================================
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import joblib

# Fix encoding on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


class TestHeartDiseaseAISystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = "data"
        cls.processed_dir = os.path.join("data", "processed")
        cls.models_dir = "models"
        cls.outputs_dir = "outputs"
        cls.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        cls.numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        cls.model_names = [
            'random_forest',
            'xgboost',
            'logistic_regression',
            'svm',
            'knn',
            'neural_network_mlp'
        ]

    def test_01_raw_data_integrity(self):
        """Kiểm tra sự tồn tại và tính hợp lệ của dữ liệu thô heart.csv"""
        raw_path = os.path.join(self.data_dir, "heart.csv")
        self.assertTrue(os.path.exists(raw_path), f"Không tìm thấy file: {raw_path}")
        df = pd.read_csv(raw_path)
        self.assertGreaterEqual(len(df), 250, "Số lượng mẫu dữ liệu quá ít")
        self.assertIn('target', df.columns, "Thiếu cột target")
        for f in self.feature_names:
            self.assertIn(f, df.columns, f"Thiếu đặc trưng: {f}")
        # Kiểm tra nhãn nhị phân
        unique_targets = set(df['target'].dropna().unique())
        self.assertTrue(unique_targets.issubset({0, 1}), f"Target chứa giá trị bất thường: {unique_targets}")

    def test_02_processed_data_integrity(self):
        """Kiểm tra dữ liệu sau tiền xử lý train/test"""
        x_train_path = os.path.join(self.processed_dir, "X_train.csv")
        x_test_path = os.path.join(self.processed_dir, "X_test.csv")
        y_train_path = os.path.join(self.processed_dir, "y_train.csv")
        y_test_path = os.path.join(self.processed_dir, "y_test.csv")

        for path in [x_train_path, x_test_path, y_train_path, y_test_path]:
            self.assertTrue(os.path.exists(path), f"Thiếu file tiền xử lý: {path}")

        X_train = pd.read_csv(x_train_path)
        X_test = pd.read_csv(x_test_path)
        y_train = pd.read_csv(y_train_path)
        y_test = pd.read_csv(y_test_path)

        self.assertEqual(len(X_train), len(y_train), "X_train và y_train không khớp số mẫu")
        self.assertEqual(len(X_test), len(y_test), "X_test và y_test không khớp số mẫu")
        self.assertEqual(X_train.shape[1], len(self.feature_names), "Số lượng cột không khớp")
        self.assertEqual(X_test.shape[1], len(self.feature_names), "Số lượng cột không khớp")
        self.assertFalse(X_train.isnull().any().any(), "X_train vẫn còn missing values")
        self.assertFalse(X_test.isnull().any().any(), "X_test vẫn còn missing values")

    def test_03_scaler_loading_and_transform(self):
        """Kiểm tra mô hình chuẩn hóa Scaler"""
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        self.assertTrue(os.path.exists(scaler_path), f"Thiếu scaler: {scaler_path}")
        scaler = joblib.load(scaler_path)
        self.assertIsNotNone(scaler)
        
        # Test transform một mẫu giả định
        dummy_df = pd.DataFrame([[55, 130, 250, 150, 1.0, 0]], columns=self.numeric_features)
        transformed = scaler.transform(dummy_df)
        self.assertEqual(transformed.shape, (1, 6), "Kích thước transform không đúng")

    def test_04_all_models_loading_and_prediction(self):
        """Kiểm tra cả 6 mô hình ML đã lưu có load được và dự đoán chính xác"""
        x_test_path = os.path.join(self.processed_dir, "X_test.csv")
        X_test = pd.read_csv(x_test_path)

        for m_name in self.model_names:
            model_path = os.path.join(self.models_dir, f"{m_name}.pkl")
            self.assertTrue(os.path.exists(model_path), f"Thiếu mô hình: {model_path}")
            model = joblib.load(model_path)
            self.assertIsNotNone(model, f"Không thể load {m_name}")

            # Predict labels
            preds = model.predict(X_test)
            self.assertEqual(len(preds), len(X_test), f"Model {m_name} dự đoán sai số lượng mẫu")
            self.assertTrue(set(np.unique(preds)).issubset({0, 1}), f"Model {m_name} sinh nhãn không hợp lệ")

            # Predict probabilities
            self.assertTrue(hasattr(model, "predict_proba"), f"Model {m_name} không có predict_proba")
            probas = model.predict_proba(X_test)
            self.assertEqual(probas.shape, (len(X_test), 2), f"Kích thước xác suất sai cho {m_name}")
            self.assertTrue(np.all(probas >= 0.0) and np.all(probas <= 1.0), f"Xác suất vượt ngưỡng [0, 1] ở {m_name}")

    def test_05_sample_batch_file_integrity(self):
        """Kiểm tra file dữ liệu mẫu cho tính năng Batch Prediction"""
        sample_path = os.path.join(self.data_dir, "sample_patients.csv")
        self.assertTrue(os.path.exists(sample_path), f"Thiếu file dữ liệu mẫu: {sample_path}")
        df = pd.read_csv(sample_path)
        self.assertGreaterEqual(len(df), 5, "Dữ liệu mẫu cần ít nhất 5 dòng")
        for col in self.feature_names:
            self.assertIn(col, df.columns, f"Thiếu cột đặc trưng {col} trong sample_patients.csv")

    def test_06_reports_and_charts_existence(self):
        """Kiểm tra các biểu đồ và báo cáo cốt lõi đã được kết xuất đầy đủ"""
        required_outputs = [
            os.path.join(self.outputs_dir, "eda", "01_target_distribution.png"),
            os.path.join(self.outputs_dir, "eda", "04_correlation_matrix.png"),
            os.path.join(self.outputs_dir, "model_comparison", "model_comparison.csv"),
            os.path.join(self.outputs_dir, "model_comparison", "01_model_comparison.png"),
            os.path.join(self.outputs_dir, "model_comparison", "02_roc_curves.png"),
            os.path.join(self.outputs_dir, "model_comparison", "03_confusion_matrices.png"),
            os.path.join(self.outputs_dir, "feature_importance", "10_overall_feature_ranking.png"),
            os.path.join(self.outputs_dir, "report", "final_report.png"),
            os.path.join(self.outputs_dir, "report", "report.txt"),
        ]
        for path in required_outputs:
            self.assertTrue(os.path.exists(path), f"Thiếu tài nguyên báo cáo: {path}")

    def test_07_clinical_presets_and_ensemble(self):
        """Kiểm tra suy luận lâm sàng các hồ sơ mẫu và thuật toán Ensemble Voting"""
        scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
        models = [joblib.load(os.path.join(self.models_dir, f"{m}.pkl")) for m in self.model_names]

        # Case 1: Người khỏe mạnh (Low risk)
        case_low = pd.DataFrame([[32, 0, 2, 112, 175, 0, 0, 178, 0, 0.0, 1, 0, 3]], columns=self.feature_names)
        case_low_scaled = case_low.copy()
        case_low_scaled[self.numeric_features] = scaler.transform(case_low[self.numeric_features])
        probas_low = [m.predict_proba(case_low_scaled)[0, 1] for m in models]
        ensemble_low = float(np.mean(probas_low))
        self.assertLess(ensemble_low, 0.35, f"Hồ sơ người khỏe mạnh có xác suất quá cao: {ensemble_low:.2f}")

        # Case 2: Ca nguy cơ cao (High risk)
        case_high = pd.DataFrame([[64, 1, 4, 168, 298, 1, 1, 105, 1, 2.8, 2, 2, 7]], columns=self.feature_names)
        case_high_scaled = case_high.copy()
        case_high_scaled[self.numeric_features] = scaler.transform(case_high[self.numeric_features])
        probas_high = [m.predict_proba(case_high_scaled)[0, 1] for m in models]
        ensemble_high = float(np.mean(probas_high))
        self.assertGreater(ensemble_high, 0.70, f"Hồ sơ nguy cơ cao có xác suất quá thấp: {ensemble_high:.2f}")

        # Batch Inference simulation
        sample_path = os.path.join(self.data_dir, "sample_patients.csv")
        sample_df = pd.read_csv(sample_path)
        batch_scaled = sample_df[self.feature_names].copy()
        batch_scaled[self.numeric_features] = scaler.transform(batch_scaled[self.numeric_features])
        for m in models:
            batch_probas = m.predict_proba(batch_scaled)[:, 1]
            self.assertEqual(len(batch_probas), len(sample_df))
            self.assertFalse(np.isnan(batch_probas).any(), "Batch probas chứa giá trị NaN")


if __name__ == '__main__':
    print("=" * 70)
    print("CHẠY KIỂM THỬ TỰ ĐỘNG TOÀN BỘ HỆ THỐNG AI (UNITTEST)")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHeartDiseaseAISystem)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print("🎉 TẤT CẢ CÁC BÀI KIỂM THỬ ĐÃ VƯỢT QUA 100% (PASS)!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print(f"❌ CÓ {len(result.failures) + len(result.errors)} LỖI KIỂM THỬ CẦN KHẮC PHỤC!")
        print("=" * 70)
        sys.exit(1)
