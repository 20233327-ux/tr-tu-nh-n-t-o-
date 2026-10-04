"""
=============================================================================
Phase 4: Phân tích yếu tố ảnh hưởng - Hệ thống phân loại bệnh tim
=============================================================================
Mục đích:
- Feature Importance từ mỗi mô hình (built-in)
- SHAP Values - giải thích mức đóng góp từng đặc trưng
- Permutation Importance
- Tổng hợp xếp hạng yếu tố ảnh hưởng lớn nhất
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap
import joblib
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
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs/feature_importance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# Tên đặc trưng tiếng Việt
FEATURE_NAMES_VI = {
    'age': 'Tuổi',
    'sex': 'Giới tính',
    'cp': 'Loại đau ngực',
    'trestbps': 'Huyết áp nghỉ',
    'chol': 'Cholesterol',
    'fbs': 'Đường huyết đói',
    'restecg': 'Điện tâm đồ nghỉ',
    'thalach': 'Nhịp tim max',
    'exang': 'Đau ngực gắng sức',
    'oldpeak': 'ST depression',
    'slope': 'Slope ST',
    'ca': 'Số mạch máu',
    'thal': 'Thalassemia'
}

# ============================================================================
# TẢI DỮ LIỆU & MÔ HÌNH
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4: PHÂN TÍCH YẾU TỐ ẢNH HƯỞNG")
    print("=" * 70)

    print("\n📥 Đang tải dữ liệu và mô hình...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

    feature_names = X_train.columns.tolist()
    feature_names_vi = [FEATURE_NAMES_VI.get(f, f) for f in feature_names]

    # Tải các mô hình đã lưu
    loaded_models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
        'SVM': 'svm.pkl',
        'KNN': 'knn.pkl',
        'Neural Network': 'neural_network_mlp.pkl'
    }

    for name, filename in model_files.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            loaded_models[name] = joblib.load(filepath)
            print(f"   ✅ {name}: {filename}")
        else:
            print(f"   ⚠️  {name}: không tìm thấy {filename}")

    # ============================================================================
    # 1. FEATURE IMPORTANCE - LOGISTIC REGRESSION (Coefficients)
    # ============================================================================
    print("\n" + "=" * 70)
    print("1. FEATURE IMPORTANCE - LOGISTIC REGRESSION (Coefficients)")
    print("=" * 70)

    if 'Logistic Regression' in loaded_models:
        lr_model = loaded_models['Logistic Regression']
        lr_importance = np.abs(lr_model.coef_[0])
        lr_df = pd.DataFrame({
            'Feature': feature_names,
            'Feature_VI': feature_names_vi,
            'Importance': lr_importance,
            'Coefficient': lr_model.coef_[0]
        }).sort_values('Importance', ascending=False)

        print("\n   Xếp hạng theo |coefficient|:")
        for i, (_, row) in enumerate(lr_df.iterrows()):
            direction = "⬆️ Tăng nguy cơ" if row['Coefficient'] > 0 else "⬇️ Giảm nguy cơ"
            print(f"   {i+1:2d}. {row['Feature_VI']:<20s} ({row['Feature']:<10s}): "
                  f"{row['Importance']:.4f}  {direction}")

        # Biểu đồ
        fig, ax = plt.subplots(figsize=(12, 7))
        colors_lr = ['#e74c3c' if c > 0 else '#3498db' for c in lr_df['Coefficient'].values]
        bars = ax.barh(lr_df['Feature_VI'].values[::-1], lr_df['Coefficient'].values[::-1],
                       color=colors_lr[::-1], edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Coefficient Value', fontsize=13)
        ax.set_title('Logistic Regression - Hệ số hồi quy\n(Đỏ = Tăng nguy cơ bệnh | Xanh = Giảm nguy cơ)',
                     fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "01_lr_coefficients.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("\n  ✅ Lưu: 01_lr_coefficients.png")

    # ============================================================================
    # 2. FEATURE IMPORTANCE - RANDOM FOREST (Gini Importance)
    # ============================================================================
    print("\n" + "=" * 70)
    print("2. FEATURE IMPORTANCE - RANDOM FOREST (Gini Importance)")
    print("=" * 70)

    if 'Random Forest' in loaded_models:
        rf_model = loaded_models['Random Forest']
        rf_importance = rf_model.feature_importances_
        rf_df = pd.DataFrame({
            'Feature': feature_names,
            'Feature_VI': feature_names_vi,
            'Importance': rf_importance
        }).sort_values('Importance', ascending=False)

        print("\n   Xếp hạng Gini Importance:")
        for i, (_, row) in enumerate(rf_df.iterrows()):
            bar = "█" * int(row['Importance'] * 50)
            print(f"   {i+1:2d}. {row['Feature_VI']:<20s} ({row['Feature']:<10s}): "
                  f"{row['Importance']:.4f} {bar}")

        # Biểu đồ
        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = plt.cm.RdYlGn_r
        norm_vals = rf_df['Importance'].values / rf_df['Importance'].values.max()
        colors_rf = [cmap(v) for v in norm_vals[::-1]]
        ax.barh(rf_df['Feature_VI'].values[::-1], rf_df['Importance'].values[::-1],
                color=colors_rf, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Gini Importance', fontsize=13)
        ax.set_title('Random Forest - Feature Importance (Gini)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "02_rf_feature_importance.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("\n  ✅ Lưu: 02_rf_feature_importance.png")

    # ============================================================================
    # 3. FEATURE IMPORTANCE - XGBOOST (Gain)
    # ============================================================================
    print("\n" + "=" * 70)
    print("3. FEATURE IMPORTANCE - XGBOOST (Gain)")
    print("=" * 70)

    if 'XGBoost' in loaded_models:
        xgb_model = loaded_models['XGBoost']
        xgb_importance = xgb_model.feature_importances_
        xgb_df = pd.DataFrame({
            'Feature': feature_names,
            'Feature_VI': feature_names_vi,
            'Importance': xgb_importance
        }).sort_values('Importance', ascending=False)

        print("\n   Xếp hạng XGBoost Importance:")
        for i, (_, row) in enumerate(xgb_df.iterrows()):
            bar = "█" * int(row['Importance'] * 50)
            print(f"   {i+1:2d}. {row['Feature_VI']:<20s} ({row['Feature']:<10s}): "
                  f"{row['Importance']:.4f} {bar}")

        # Biểu đồ
        fig, ax = plt.subplots(figsize=(12, 7))
        norm_vals = xgb_df['Importance'].values / xgb_df['Importance'].values.max()
        colors_xgb = [plt.cm.YlOrRd(v) for v in norm_vals[::-1]]
        ax.barh(xgb_df['Feature_VI'].values[::-1], xgb_df['Importance'].values[::-1],
                color=colors_xgb, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Feature Importance (Gain)', fontsize=13)
        ax.set_title('XGBoost - Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "03_xgb_feature_importance.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("\n  ✅ Lưu: 03_xgb_feature_importance.png")

    # ============================================================================
    # 4. PERMUTATION IMPORTANCE (trên tất cả mô hình)
    # ============================================================================
    print("\n" + "=" * 70)
    print("4. PERMUTATION IMPORTANCE")
    print("=" * 70)

    perm_results = {}

    for name, model in loaded_models.items():
        print(f"\n   🔄 Tính Permutation Importance cho {name}...")
        try:
            perm = permutation_importance(model, X_test, y_test,
                                           n_repeats=10, random_state=RANDOM_STATE,
                                           n_jobs=-1, scoring='accuracy')
            perm_results[name] = perm.importances_mean
            top3 = np.argsort(perm.importances_mean)[::-1][:3]
            top3_names = [feature_names_vi[j] for j in top3]
            top3_vals = [perm.importances_mean[j] for j in top3]
            print(f"      Top 3: {top3_names[0]} ({top3_vals[0]:.4f}), "
                  f"{top3_names[1]} ({top3_vals[1]:.4f}), "
                  f"{top3_names[2]} ({top3_vals[2]:.4f})")
        except Exception as e:
            print(f"      ⚠️  Lỗi: {e}")

    # Biểu đồ Permutation Importance tổng hợp
    if perm_results:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for i, (name, importances) in enumerate(perm_results.items()):
            if i >= 6:
                break
            ax = axes[i]
            sorted_idx = np.argsort(importances)
            sorted_names = [feature_names_vi[j] for j in sorted_idx]
            sorted_vals = importances[sorted_idx]

            colors_perm = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(sorted_vals)))
            ax.barh(sorted_names, sorted_vals, color=colors_perm, edgecolor='black', linewidth=0.5)
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Permutation Importance')

        # Ẩn axes thừa
        for j in range(len(perm_results), 6):
            axes[j].set_visible(False)

        plt.suptitle('Permutation Importance - Tất cả mô hình', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "04_permutation_importance.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("\n  ✅ Lưu: 04_permutation_importance.png")

    # ============================================================================
    # 5. SHAP VALUES (Random Forest & XGBoost)
    # ============================================================================
    print("\n" + "=" * 70)
    print("5. SHAP VALUES - PHÂN TÍCH SÂU")
    print("=" * 70)

    # SHAP cho Random Forest
    if 'Random Forest' in loaded_models:
        print("\n   🔄 Tính SHAP values cho Random Forest...")
        try:
            rf_model = loaded_models['Random Forest']
            explainer_rf = shap.TreeExplainer(rf_model)
            shap_values_rf = explainer_rf.shap_values(X_test)

            # Lấy SHAP values cho class 1 (có bệnh)
            if isinstance(shap_values_rf, list):
                shap_vals = shap_values_rf[1]
            else:
                shap_vals = shap_values_rf

            # SHAP Summary Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_vals, X_test,
                              feature_names=feature_names_vi,
                              show=False, max_display=13)
            plt.title('SHAP Summary Plot - Random Forest\n(Ảnh hưởng của từng đặc trưng đến dự đoán bệnh tim)',
                      fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "05_shap_summary_rf.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print("   ✅ Lưu: 05_shap_summary_rf.png")

            # SHAP Bar Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_vals, X_test,
                              feature_names=feature_names_vi,
                              plot_type="bar", show=False, max_display=13)
            plt.title('SHAP Feature Importance (Mean |SHAP|) - Random Forest',
                      fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "06_shap_bar_rf.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print("   ✅ Lưu: 06_shap_bar_rf.png")

        except Exception as e:
            print(f"   ⚠️  Lỗi SHAP Random Forest: {e}")

    # SHAP cho XGBoost
    if 'XGBoost' in loaded_models:
        print("\n   🔄 Tính SHAP values cho XGBoost...")
        try:
            xgb_model = loaded_models['XGBoost']
            explainer_xgb = shap.TreeExplainer(xgb_model)
            shap_values_xgb = explainer_xgb.shap_values(X_test)

            # SHAP Summary Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values_xgb, X_test,
                              feature_names=feature_names_vi,
                              show=False, max_display=13)
            plt.title('SHAP Summary Plot - XGBoost\n(Ảnh hưởng của từng đặc trưng đến dự đoán bệnh tim)',
                      fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "07_shap_summary_xgb.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print("   ✅ Lưu: 07_shap_summary_xgb.png")

            # SHAP Bar Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values_xgb, X_test,
                              feature_names=feature_names_vi,
                              plot_type="bar", show=False, max_display=13)
            plt.title('SHAP Feature Importance (Mean |SHAP|) - XGBoost',
                      fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "08_shap_bar_xgb.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print("   ✅ Lưu: 08_shap_bar_xgb.png")

            # SHAP Dependence Plot cho top 4 features
            print("\n   📊 Tạo SHAP Dependence Plots cho top features...")
            top4_idx = np.argsort(np.abs(shap_values_xgb).mean(axis=0))[::-1][:4]

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            for i, feat_idx in enumerate(top4_idx):
                plt.sca(axes[i])
                shap.dependence_plot(feat_idx, shap_values_xgb, X_test,
                                     feature_names=feature_names_vi,
                                     show=False, ax=axes[i])
                axes[i].set_title(f'SHAP Dependence: {feature_names_vi[feat_idx]}',
                                  fontsize=12, fontweight='bold')

            plt.suptitle('SHAP Dependence Plots - Top 4 Features (XGBoost)',
                         fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "09_shap_dependence_xgb.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print("   ✅ Lưu: 09_shap_dependence_xgb.png")

        except Exception as e:
            print(f"   ⚠️  Lỗi SHAP XGBoost: {e}")

    # ============================================================================
    # 6. TỔNG HỢP XẾP HẠNG YẾU TỐ ẢNH HƯỞNG
    # ============================================================================
    print("\n" + "=" * 70)
    print("6. TỔNG HỢP XẾP HẠNG YẾU TỐ ẢNH HƯỞNG")
    print("=" * 70)

    # Thu thập rankings từ các phương pháp
    rankings = {}

    # Logistic Regression coefficients
    if 'Logistic Regression' in loaded_models:
        lr_model = loaded_models['Logistic Regression']
        lr_imp = np.abs(lr_model.coef_[0])
        lr_rank = np.argsort(lr_imp)[::-1]
        rankings['LR Coef'] = {feature_names[i]: rank+1 for rank, i in enumerate(lr_rank)}

    # Random Forest
    if 'Random Forest' in loaded_models:
        rf_imp = loaded_models['Random Forest'].feature_importances_
        rf_rank = np.argsort(rf_imp)[::-1]
        rankings['RF Gini'] = {feature_names[i]: rank+1 for rank, i in enumerate(rf_rank)}

    # XGBoost
    if 'XGBoost' in loaded_models:
        xgb_imp = loaded_models['XGBoost'].feature_importances_
        xgb_rank = np.argsort(xgb_imp)[::-1]
        rankings['XGB Gain'] = {feature_names[i]: rank+1 for rank, i in enumerate(xgb_rank)}

    # Permutation Importance (average across models)
    if perm_results:
        avg_perm = np.zeros(len(feature_names))
        for imp in perm_results.values():
            avg_perm += imp
        avg_perm /= len(perm_results)
        perm_rank = np.argsort(avg_perm)[::-1]
        rankings['Perm Avg'] = {feature_names[i]: rank+1 for rank, i in enumerate(perm_rank)}

    # Tạo bảng tổng hợp
    if rankings:
        ranking_df = pd.DataFrame(rankings)
        ranking_df['Trung bình'] = ranking_df.mean(axis=1)
        ranking_df = ranking_df.sort_values('Trung bình')
        ranking_df.index = [FEATURE_NAMES_VI.get(f, f) + f' ({f})' for f in ranking_df.index]

        print("\n📊 Bảng xếp hạng tổng hợp (1 = quan trọng nhất):\n")
        print(ranking_df.to_string())

        # Biểu đồ tổng hợp
        fig, ax = plt.subplots(figsize=(14, 8))

        # Lấy tên features theo thứ tự
        sorted_features = ranking_df.index.tolist()[::-1]
        avg_ranks = ranking_df['Trung bình'].values[::-1]

        # Color code
        n = len(sorted_features)
        colors_final = plt.cm.RdYlGn(np.linspace(0.8, 0.2, n))

        bars = ax.barh(sorted_features, avg_ranks, color=colors_final,
                       edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Xếp hạng trung bình (thấp = quan trọng hơn)', fontsize=13)
        ax.set_title('TỔNG HỢP XẾP HẠNG YẾU TỐ ẢNH HƯỞNG ĐẾN BỆNH TIM\n'
                     '(Trung bình từ LR, RF, XGBoost, Permutation Importance)',
                     fontsize=14, fontweight='bold')

        # Thêm giá trị
        for bar, val in zip(bars, avg_ranks):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}', ha='left', va='center', fontsize=11, fontweight='bold')

        ax.invert_xaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "10_overall_feature_ranking.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("\n  ✅ Lưu: 10_overall_feature_ranking.png")

        # In top 5 yếu tố ảnh hưởng lớn nhất
        print("\n\n" + "=" * 70)
        print("🏆 TOP 5 YẾU TỐ ẢNH HƯỞNG LỚN NHẤT ĐẾN NGUY CƠ MẮC BỆNH TIM:")
        print("=" * 70)
        top5 = ranking_df.head(5)
        for i, (feat, row) in enumerate(top5.iterrows()):
            print(f"\n   {i+1}. {feat}")
            print(f"      Xếp hạng trung bình: {row['Trung bình']:.1f}")
            for method in rankings.keys():
                print(f"      - {method}: #{int(row[method])}")

    # ============================================================================
    # TỔNG KẾT
    # ============================================================================
    print("\n\n" + "=" * 70)
    print("✅ HOÀN THÀNH PHÂN TÍCH YẾU TỐ ẢNH HƯỞNG!")
    print("=" * 70)
    print(f"\n📁 Tất cả biểu đồ: {OUTPUT_DIR}/")
    print(f"   - 01_lr_coefficients.png")
    print(f"   - 02_rf_feature_importance.png")
    print(f"   - 03_xgb_feature_importance.png")
    print(f"   - 04_permutation_importance.png")
    print(f"   - 05_shap_summary_rf.png")
    print(f"   - 06_shap_bar_rf.png")
    print(f"   - 07_shap_summary_xgb.png")
    print(f"   - 08_shap_bar_xgb.png")
    print(f"   - 09_shap_dependence_xgb.png")
    print(f"   - 10_overall_feature_ranking.png")
