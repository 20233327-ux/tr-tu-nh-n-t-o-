"""
=============================================================================
Phase 5: Tổng hợp báo cáo - Hệ thống phân loại bệnh tim
=============================================================================
Mục đích:
- Tổng hợp kết quả từ tất cả các phase
- Tạo báo cáo tổng kết so sánh mô hình
- Đề xuất mô hình tốt nhất
- Tạo biểu đồ tổng kết
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
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
COMPARISON_DIR = "outputs/model_comparison"
OUTPUT_DIR = "outputs/report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# TẢI KẾT QUẢ
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 5: TỔNG HỢP BÁO CÁO")
    print("=" * 70)

    # Tải bảng so sánh mô hình
    comparison_path = os.path.join(COMPARISON_DIR, "model_comparison.csv")
    if not os.path.exists(comparison_path):
        print("❌ Chưa có kết quả so sánh! Vui lòng chạy 03_model_training.py trước.")
        exit(1)

    results_df = pd.read_csv(comparison_path, index_col=0)

    # Tải dữ liệu
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()
    df_raw = pd.read_csv("data/heart.csv")

    print(f"\n📊 Đã tải kết quả so sánh {len(results_df)} mô hình")

    # ============================================================================
    # 1. BÁO CÁO TỔNG HỢP
    # ============================================================================
    print("\n" + "=" * 70)
    print("📊 BÁO CÁO TỔNG HỢP - HỆ THỐNG PHÂN LOẠI BỆNH TIM")
    print("=" * 70)

    print(f"\n{'─' * 60}")
    print("1. THÔNG TIN DỮ LIỆU")
    print(f"{'─' * 60}")
    print(f"   Dataset:        UCI Heart Disease (Cleveland)")
    print(f"   Tổng mẫu:       {len(df_raw)}")
    print(f"   Số đặc trưng:   {df_raw.shape[1] - 1}")
    print(f"   Tỷ lệ bệnh:    {df_raw['target'].mean()*100:.1f}%")
    print(f"   Train/Test:     80%/20%")

    print(f"\n{'─' * 60}")
    print("2. KẾT QUẢ SO SÁNH MÔ HÌNH")
    print(f"{'─' * 60}")

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    print(f"\n{'Mô hình':<25s}", end="")
    for m in metrics:
        print(f"{m:>12s}", end="")
    print()
    print("─" * 85)

    for name, row in results_df.iterrows():
        print(f"   {name:<22s}", end="")
        for m in metrics:
            if m in row.index:
                val = row[m]
                print(f"{val:>12.4f}", end="")
            else:
                print(f"{'N/A':>12s}", end="")
        print()

    # ============================================================================
    # 2. BIỂU ĐỒ TỔNG KẾT CUỐI CÙNG
    # ============================================================================
    print(f"\n📊 Tạo biểu đồ tổng kết...")

    fig = plt.figure(figsize=(24, 18))
    fig.suptitle('BÁO CÁO TỔNG KẾT - HỆ THỐNG PHÂN LOẠI BỆNH TIM',
                 fontsize=20, fontweight='bold', y=0.98)

    # Color scheme
    model_colors = {
        'Logistic Regression': '#3498db',
        'Random Forest': '#2ecc71',
        'XGBoost': '#e74c3c',
        'SVM': '#f39c12',
        'KNN': '#9b59b6',
        'Neural Network (MLP)': '#1abc9c'
    }

    model_names = results_df.index.tolist()
    colors = [model_colors.get(name, '#95a5a6') for name in model_names]

    # --- Panel 1: Tổng hợp metrics ---
    ax1 = fig.add_subplot(2, 3, 1)
    available_metrics = [m for m in metrics if m in results_df.columns]
    x = np.arange(len(model_names))
    width = 0.15
    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics)/2) * width
        vals = results_df[metric].values
        ax1.bar(x + offset, vals, width, label=metric, alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace(' ', '\n').replace('(', '\n(') for n in model_names], fontsize=8)
    ax1.set_ylabel('Score')
    ax1.set_title('Tổng hợp Metrics', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.set_ylim(0, 1.1)

    # --- Panel 2: Xếp hạng tổng hợp ---
    ax2 = fig.add_subplot(2, 3, 2)
    # Tính điểm tổng hợp (trung bình các metrics)
    composite_scores = results_df[available_metrics].mean(axis=1).sort_values(ascending=True)
    sorted_names = composite_scores.index.tolist()
    sorted_colors = [model_colors.get(n, '#95a5a6') for n in sorted_names]

    bars = ax2.barh(sorted_names, composite_scores.values, color=sorted_colors,
                    edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Điểm tổng hợp (Trung bình)')
    ax2.set_title('Xếp hạng tổng hợp', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, composite_scores.values):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2.,
                 f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

    # --- Panel 3: Heatmap metrics ---
    ax3 = fig.add_subplot(2, 3, 3)
    heatmap_data = results_df[available_metrics].copy()
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, ax=ax3, linewidths=1,
                cbar_kws={'shrink': 0.8})
    ax3.set_title('Heatmap - So sánh chi tiết', fontsize=14, fontweight='bold')
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=9)

    # --- Panel 4: Accuracy vs Training Time ---
    ax4 = fig.add_subplot(2, 3, 4)
    if 'Time (s)' in results_df.columns:
        for i, name in enumerate(model_names):
            ax4.scatter(results_df.loc[name, 'Time (s)'],
                        results_df.loc[name, 'Accuracy'],
                        s=200, color=colors[i], edgecolor='black', linewidth=1, zorder=5)
            ax4.annotate(name, (results_df.loc[name, 'Time (s)'],
                         results_df.loc[name, 'Accuracy']),
                         textcoords="offset points", xytext=(10, 5), fontsize=8)
        ax4.set_xlabel('Training Time (seconds)', fontsize=12)
        ax4.set_ylabel('Accuracy', fontsize=12)
        ax4.set_title('Accuracy vs Thời gian huấn luyện', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

    # --- Panel 5: Precision vs Recall trade-off ---
    ax5 = fig.add_subplot(2, 3, 5)
    for i, name in enumerate(model_names):
        if 'Precision' in results_df.columns and 'Recall' in results_df.columns:
            ax5.scatter(results_df.loc[name, 'Recall'],
                        results_df.loc[name, 'Precision'],
                        s=250, color=colors[i], edgecolor='black', linewidth=1,
                        zorder=5, marker='o')
            ax5.annotate(name, (results_df.loc[name, 'Recall'],
                         results_df.loc[name, 'Precision']),
                         textcoords="offset points", xytext=(10, 5), fontsize=8)

    ax5.set_xlabel('Recall (Độ nhạy)', fontsize=12)
    ax5.set_ylabel('Precision (Độ chính xác)', fontsize=12)
    ax5.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
    ax5.set_xlim(0.5, 1.05)
    ax5.set_ylim(0.5, 1.05)
    ax5.grid(True, alpha=0.3)

    # Đường F1-Score iso-lines
    for f1_val in [0.6, 0.7, 0.8, 0.9]:
        recall_range = np.linspace(0.5, 1.0, 100)
        precision_range = f1_val * recall_range / (2 * recall_range - f1_val)
        valid = (precision_range > 0.5) & (precision_range <= 1.0)
        ax5.plot(recall_range[valid], precision_range[valid], '--', alpha=0.3, color='gray')
        if valid.any():
            idx = np.argmin(np.abs(recall_range[valid] - 0.9))
            ax5.annotate(f'F1={f1_val}', (recall_range[valid][idx], precision_range[valid][idx]),
                         fontsize=7, alpha=0.5)

    # --- Panel 6: Đề xuất cuối cùng ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Tìm mô hình tốt nhất
    best_overall = composite_scores.idxmax()
    best_accuracy = results_df['Accuracy'].idxmax() if 'Accuracy' in results_df.columns else 'N/A'
    best_recall = results_df['Recall'].idxmax() if 'Recall' in results_df.columns else 'N/A'
    best_f1 = results_df['F1-Score'].idxmax() if 'F1-Score' in results_df.columns else 'N/A'
    best_auc = results_df['AUC-ROC'].idxmax() if 'AUC-ROC' in results_df.columns else 'N/A'

    recommendation_text = f"""
    🏆 ĐỀ XUẤT MÔ HÌNH

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    🥇 Tốt nhất tổng hợp:
       {best_overall}
       (Score: {composite_scores.max():.4f})

    🎯 Tốt nhất Accuracy:
       {best_accuracy} ({results_df.loc[best_accuracy, 'Accuracy']:.4f})

    🔬 Tốt nhất Recall (Y tế):
       {best_recall} ({results_df.loc[best_recall, 'Recall']:.4f})

    ⚖️  Tốt nhất F1-Score:
       {best_f1} ({results_df.loc[best_f1, 'F1-Score']:.4f})

    📈 Tốt nhất AUC-ROC:
       {best_auc} ({results_df.loc[best_auc, 'AUC-ROC']:.4f})

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    💡 KHUYẾN NGHỊ:
    Trong lĩnh vực y tế, Recall (độ nhạy)
    là chỉ số quan trọng nhất để không
    bỏ sót bệnh nhân mắc bệnh tim.
    """

    ax6.text(0.05, 0.95, recommendation_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='orange', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, "final_report.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Lưu: final_report.png")

    # ============================================================================
    # 3. LƯU BÁO CÁO TEXT
    # ============================================================================
    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BÁO CÁO TỔNG KẾT - HỆ THỐNG PHÂN LOẠI BỆNH TIM\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. THÔNG TIN DỮ LIỆU\n")
        f.write(f"   Dataset: UCI Heart Disease (Cleveland)\n")
        f.write(f"   Tổng mẫu: {len(df_raw)}\n")
        f.write(f"   Số đặc trưng: {df_raw.shape[1] - 1}\n")
        f.write(f"   Tỷ lệ bệnh: {df_raw['target'].mean()*100:.1f}%\n\n")

        f.write("2. KẾT QUẢ SO SÁNH MÔ HÌNH\n\n")
        f.write(results_df[available_metrics].to_string())
        f.write("\n\n")

        f.write("3. XẾP HẠNG TỔNG HỢP\n\n")
        for i, (name, score) in enumerate(composite_scores.sort_values(ascending=False).items()):
            f.write(f"   #{i+1}: {name} (Score: {score:.4f})\n")
        f.write("\n")

        f.write("4. ĐỀ XUẤT\n\n")
        f.write(f"   Mô hình tốt nhất tổng hợp: {best_overall}\n")
        f.write(f"   Mô hình tốt nhất Accuracy: {best_accuracy}\n")
        f.write(f"   Mô hình tốt nhất Recall: {best_recall}\n")
        f.write(f"   Mô hình tốt nhất F1-Score: {best_f1}\n")
        f.write(f"   Mô hình tốt nhất AUC-ROC: {best_auc}\n\n")

        f.write("   💡 Trong y tế, Recall quan trọng nhất để không bỏ sót bệnh nhân.\n")

    print(f"  ✅ Lưu: {report_path}")

    # ============================================================================
    # TỔNG KẾT CUỐI CÙNG
    # ============================================================================
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TOÀN BỘ HỆ THỐNG!")
    print("=" * 70)

    print(f"\n📁 Cấu trúc kết quả:")
    print(f"   outputs/")
    print(f"   ├── eda/                    # Biểu đồ phân tích thăm dò")
    print(f"   ├── model_comparison/       # So sánh mô hình + ROC curves")
    print(f"   ├── feature_importance/     # SHAP + Feature importance")
    print(f"   └── report/                 # Báo cáo tổng kết")
    print(f"   models/                     # Mô hình đã lưu (.pkl)")

    print(f"\n🏆 KẾT QUẢ CUỐI CÙNG:")
    print(f"   Mô hình tốt nhất: {best_overall}")
    print(f"   Điểm tổng hợp:    {composite_scores.max():.4f}")
    print(f"\n   Trong y tế, ưu tiên mô hình có Recall cao nhất: {best_recall}")
