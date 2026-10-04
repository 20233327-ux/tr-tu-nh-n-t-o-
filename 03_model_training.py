"""
=============================================================================
Phase 3: Huấn luyện & Đánh giá 6 Mô hình ML - Phân loại bệnh tim
=============================================================================
Mô hình:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Neural Network (MLP)

Đánh giá: Accuracy, Precision, Recall, F1-Score, AUC-ROC
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)
import joblib
import os
import sys
import time
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
OUTPUT_DIR = "outputs/model_comparison"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5

# ============================================================================
# TẢI DỮ LIỆU ĐÃ XỬ LÝ
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 70)

    print("\n📥 Đang tải dữ liệu đã xử lý...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

    print(f"   Train: {X_train.shape[0]} mẫu, Test: {X_test.shape[0]} mẫu")
    print(f"   Số đặc trưng: {X_train.shape[1]}")

    # ============================================================================
    # ĐỊNH NGHĨA CÁC MÔ HÌNH & THAM SỐ TÌM KIẾM
    # ============================================================================

    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss',
                                    use_label_encoder=False),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        },
        'SVM': {
            'model': SVC(random_state=RANDOM_STATE, probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Neural Network (MLP)': {
            'model': MLPClassifier(random_state=RANDOM_STATE, max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50), (100, 50, 30)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
    }

    # ============================================================================
    # HUẤN LUYỆN & TỐI ƯU HÓA
    # ============================================================================
    results = {}
    best_models = {}

    print(f"\n🔧 Bắt đầu huấn luyện {len(models)} mô hình...")
    print(f"   Cross-validation: {CV_FOLDS}-fold (Stratified)")
    print(f"   Tối ưu hóa: RandomizedSearchCV")
    print()

    for name, config in models.items():
        print(f"{'─' * 60}")
        print(f"🔄 [{name}]")
        print(f"{'─' * 60}")
        start_time = time.time()

        # Cross-validation ban đầu
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(config['model'], X_train, y_train, cv=cv, scoring='accuracy')
        print(f"   CV Accuracy (trước tuning): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Hyperparameter tuning
        # Giới hạn số tổ hợp để chạy nhanh hơn
        n_combinations = 1
        for v in config['params'].values():
            n_combinations *= len(v)

        if n_combinations > 50:
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                config['model'], config['params'],
                n_iter=min(30, n_combinations),
                cv=cv, scoring='accuracy',
                n_jobs=-1, random_state=RANDOM_STATE, verbose=0
            )
        else:
            search = GridSearchCV(
                config['model'], config['params'],
                cv=cv, scoring='accuracy',
                n_jobs=-1, verbose=0
            )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_models[name] = best_model

        # Đánh giá trên tập test
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba)

        elapsed = time.time() - start_time

        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'AUC-ROC': auc_score,
            'CV Mean': search.best_score_,
            'CV Std': cv_scores.std(),
            'Best Params': search.best_params_,
            'Time (s)': elapsed
        }

        print(f"   Tham số tốt nhất: {search.best_params_}")
        print(f"   CV Accuracy (sau tuning): {search.best_score_:.4f}")
        print(f"   Test Accuracy: {acc:.4f}")
        print(f"   Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print(f"   AUC-ROC: {auc_score:.4f}")
        print(f"   ⏱️  Thời gian: {elapsed:.1f}s")

        # Lưu mô hình
        model_filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        joblib.dump(best_model, os.path.join(MODEL_DIR, f"{model_filename}.pkl"))
        print(f"   💾 Lưu: {model_filename}.pkl")
        print()

    # ============================================================================
    # BẢNG SO SÁNH TỔNG HỢP
    # ============================================================================
    print("\n" + "=" * 70)
    print("📊 BẢNG SO SÁNH TỔNG HỢP CÁC MÔ HÌNH")
    print("=" * 70)

    results_df = pd.DataFrame(results).T
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'CV Mean', 'Time (s)']
    print("\n" + results_df[metrics_cols].to_string())

    # Highlight mô hình tốt nhất cho mỗi metric
    print("\n\n🏆 MÔ HÌNH TỐT NHẤT CHO TỪNG CHỈ SỐ:")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
        best_name = results_df[metric].idxmax()
        best_val = results_df[metric].max()
        print(f"   {metric:12s}: {best_name} ({best_val:.4f})")

    # Lưu bảng kết quả
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
    print(f"\n💾 Bảng so sánh: {os.path.join(OUTPUT_DIR, 'model_comparison.csv')}")

    # ============================================================================
    # BIỂU ĐỒ SO SÁNH CÁC METRICS
    # ============================================================================
    print("\n📊 Tạo biểu đồ so sánh...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Màu sắc cho các mô hình
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    model_names = list(results.keys())

    # 1. Bar chart - Accuracy & F1-Score
    ax = axes[0, 0]
    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, [results[m]['Accuracy'] for m in model_names],
                   width, label='Accuracy', color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, [results[m]['F1-Score'] for m in model_names],
                   width, label='F1-Score', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Mô hình')
    ax.set_ylabel('Giá trị')
    ax.set_title('So sánh Accuracy & F1-Score', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    # Thêm giá trị lên bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    # 2. Bar chart - Precision & Recall
    ax = axes[0, 1]
    bars1 = ax.bar(x - width/2, [results[m]['Precision'] for m in model_names],
                   width, label='Precision', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, [results[m]['Recall'] for m in model_names],
                   width, label='Recall', color='#f39c12', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Mô hình')
    ax.set_ylabel('Giá trị')
    ax.set_title('So sánh Precision & Recall', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    # 3. AUC-ROC comparison
    ax = axes[1, 0]
    auc_values = [results[m]['AUC-ROC'] for m in model_names]
    bars = ax.barh(model_names, auc_values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('AUC-ROC Score')
    ax.set_title('So sánh AUC-ROC', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, auc_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

    # 4. Radar chart - Tổng hợp
    axes[1, 1].remove()
    metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
    angles += angles[:1]

    ax = fig.add_subplot(2, 2, 4, polar=True)
    for i, name in enumerate(model_names):
        values = [results[name][m] for m in metrics_for_radar]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_for_radar, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Radar Chart - Tổng hợp', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_model_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Lưu: 01_model_comparison.png")

    # ============================================================================
    # ROC CURVES
    # ============================================================================
    print("\n📊 Tạo ROC Curves...")

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (name, model) in enumerate(best_models.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2.5,
                label=f'{name} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    ax.set_title('ROC Curves - So sánh 6 mô hình', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_roc_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Lưu: 02_roc_curves.png")

    # ============================================================================
    # CONFUSION MATRICES
    # ============================================================================
    print("\n📊 Tạo Confusion Matrices...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (name, model) in enumerate(best_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Không bệnh', 'Có bệnh'],
                    yticklabels=['Không bệnh', 'Có bệnh'],
                    annot_kws={'size': 16})
        axes[i].set_title(f'{name}', fontsize=13, fontweight='bold')
        axes[i].set_xlabel('Dự đoán')
        axes[i].set_ylabel('Thực tế')

    plt.suptitle('Confusion Matrix - 6 Mô hình', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_confusion_matrices.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Lưu: 03_confusion_matrices.png")

    # ============================================================================
    # CLASSIFICATION REPORTS
    # ============================================================================
    print("\n" + "=" * 70)
    print("📊 BÁO CÁO PHÂN LOẠI CHI TIẾT")
    print("=" * 70)

    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        print(f"\n{'─' * 50}")
        print(f"📋 {name}")
        print(f"{'─' * 50}")
        print(classification_report(y_test, y_pred,
                                    target_names=['Không bệnh', 'Có bệnh']))

    # ============================================================================
    # TỔNG KẾT
    # ============================================================================
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH HUẤN LUYỆN & ĐÁNH GIÁ!")
    print("=" * 70)

    # Tìm mô hình tốt nhất theo F1-Score (cân bằng precision & recall)
    best_by_f1 = max(results.items(), key=lambda x: x[1]['F1-Score'])
    best_by_auc = max(results.items(), key=lambda x: x[1]['AUC-ROC'])
    best_by_recall = max(results.items(), key=lambda x: x[1]['Recall'])

    print(f"\n🏆 ĐỀ XUẤT:")
    print(f"   Tốt nhất theo F1-Score:  {best_by_f1[0]} ({best_by_f1[1]['F1-Score']:.4f})")
    print(f"   Tốt nhất theo AUC-ROC:  {best_by_auc[0]} ({best_by_auc[1]['AUC-ROC']:.4f})")
    print(f"   Tốt nhất theo Recall:   {best_by_recall[0]} ({best_by_recall[1]['Recall']:.4f})")
    print(f"   → Trong y tế, Recall (độ nhạy) rất quan trọng để không bỏ sót bệnh nhân!")

    print(f"\n📁 Biểu đồ: {OUTPUT_DIR}/")
    print(f"📁 Mô hình: {MODEL_DIR}/")
