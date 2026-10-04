"""
=============================================================================
Phase 1: Phân tích dữ liệu thăm dò (EDA) - Hệ thống phân loại bệnh tim
=============================================================================
Mục đích:
- Thống kê mô tả dữ liệu
- Trực quan hóa phân phối từng đặc trưng
- Ma trận tương quan
- Phân tích tỷ lệ mắc bệnh theo nhóm
- Phát hiện outliers & missing values
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
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
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_style("whitegrid")
sns.set_palette("husl")

OUTPUT_DIR = "outputs/eda"
DATA_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# TẢI DỮ LIỆU
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 1: PHÂN TÍCH DỮ LIỆU THĂM DÒ (EDA)")
    print("=" * 70)

    # Tải dataset UCI Heart Disease
    local_csv = os.path.join(DATA_DIR, "heart.csv")
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df = None

    if os.path.exists(local_csv):
        print(f"\n📂 Đang đọc dữ liệu cục bộ từ: {local_csv}")
        try:
            df = pd.read_csv(local_csv)
            print("✅ Đã tải dữ liệu cục bộ thành công!")
        except Exception as e:
            print(f"⚠️ Không đọc được file cục bộ: {e}")

    if df is None:
        print("\n📥 Đang tải dữ liệu UCI Heart Disease từ internet...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        try:
            df = pd.read_csv(url, names=column_names, na_values='?')
            print("✅ Tải dữ liệu thành công từ UCI Repository!")
        except Exception:
            print("⚠️  Không thể tải từ UCI, thử tải từ nguồn thay thế...")
            try:
                # Thử từ sklearn OpenML
                heart = fetch_openml(name='heart-disease', version=1, as_frame=True)
                df = heart.frame
                if 'num' in df.columns:
                    df = df.rename(columns={'num': 'target'})
                print("✅ Tải dữ liệu thành công từ OpenML!")
            except Exception:
                print("❌ Không thể tải dữ liệu. Vui lòng kiểm tra kết nối mạng.")
                exit(1)

    # Chuyển target thành nhị phân (0: không bệnh, 1: có bệnh) nếu chưa chuyển
    if df['target'].max() > 1:
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Lưu dữ liệu gốc
    df.to_csv(os.path.join(DATA_DIR, "heart.csv"), index=False)
    print(f"💾 Đã lưu dữ liệu tại: {os.path.join(DATA_DIR, 'heart.csv')}")
    print(f"📊 Kích thước dữ liệu: {df.shape[0]} mẫu × {df.shape[1]} đặc trưng")

    # ============================================================================
    # 1. THỐNG KÊ MÔ TẢ
    # ============================================================================
    print("\n" + "=" * 70)
    print("1. THỐNG KÊ MÔ TẢ")
    print("=" * 70)

    print("\n📋 Thông tin dữ liệu:")
    print(df.info())

    print("\n📊 Thống kê mô tả:")
    desc = df.describe()
    print(desc.to_string())

    print(f"\n📊 Tỷ lệ mắc bệnh:")
    target_counts = df['target'].value_counts()
    print(f"   Không bệnh (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"   Có bệnh    (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")

    # ============================================================================
    # 2. KIỂM TRA MISSING VALUES
    # ============================================================================
    print("\n" + "=" * 70)
    print("2. KIỂM TRA MISSING VALUES")
    print("=" * 70)

    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'Số lượng': missing, 'Tỷ lệ (%)': missing_pct})
    missing_df = missing_df[missing_df['Số lượng'] > 0]

    if len(missing_df) > 0:
        print("\n⚠️  Có missing values:")
        print(missing_df.to_string())
    else:
        print("\n✅ Không có missing values!")

    # ============================================================================
    # 3. TRỰC QUAN HÓA PHÂN PHỐI BIẾN MỤC TIÊU
    # ============================================================================
    print("\n📊 Tạo biểu đồ phân phối biến mục tiêu...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Không bệnh (0)', 'Có bệnh (1)']
    sizes = [target_counts.get(0, 0), target_counts.get(1, 0)]
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 13}, explode=(0.05, 0.05))
    axes[0].set_title('Tỷ lệ mắc bệnh tim', fontsize=15, fontweight='bold')

    # Bar chart
    bars = axes[1].bar(labels, sizes, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title('Số lượng theo nhóm', fontsize=15, fontweight='bold')
    axes[1].set_ylabel('Số lượng')
    for bar, count in zip(bars, sizes):
        axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                     str(count), ha='center', va='bottom', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_target_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Lưu: 01_target_distribution.png")

    # ============================================================================
    # 4. HISTOGRAM CÁC ĐẶC TRƯNG SỐ
    # ============================================================================
    print("\n📊 Tạo histogram các đặc trưng số...")

    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, col in enumerate(numeric_features):
        ax = axes[i]
        # Histogram theo nhóm target
        df[df['target'] == 0][col].hist(ax=ax, bins=25, alpha=0.6, color='#2ecc71',
                                          label='Không bệnh', edgecolor='black', linewidth=0.5)
        df[df['target'] == 1][col].hist(ax=ax, bins=25, alpha=0.6, color='#e74c3c',
                                          label='Có bệnh', edgecolor='black', linewidth=0.5)
        ax.set_title(f'Phân phối: {col}', fontsize=13, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Số lượng')
        ax.legend()

    # Ẩn subplot thừa
    axes[-1].set_visible(False)

    plt.suptitle('Phân phối các đặc trưng số theo nhóm bệnh', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_numeric_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Lưu: 02_numeric_distributions.png")

    # ============================================================================
    # 5. PHÂN PHỐI CÁC BIẾN PHÂN LOẠI
    # ============================================================================
    print("\n📊 Tạo biểu đồ phân phối biến phân loại...")

    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
    cat_labels = {
        'sex': {0: 'Nữ', 1: 'Nam'},
        'cp': {1: 'Typical\nangina', 2: 'Atypical\nangina', 3: 'Non-anginal\npain', 4: 'Asymptomatic'},
        'fbs': {0: '≤120 mg/dl', 1: '>120 mg/dl'},
        'restecg': {0: 'Bình thường', 1: 'ST-T bất\nthường', 2: 'LV\nhypertrophy'},
        'exang': {0: 'Không', 1: 'Có'},
        'slope': {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'}
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for i, col in enumerate(categorical_features):
        ax = axes[i]
        ct = pd.crosstab(df[col], df['target'])
        ct.columns = ['Không bệnh', 'Có bệnh']
        if col in cat_labels:
            ct.index = [cat_labels[col].get(idx, str(idx)) for idx in ct.index]
        ct.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=0.5)
        ax.set_title(f'{col}', fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Số lượng')
        ax.legend(title='')
        ax.tick_params(axis='x', rotation=0)

    plt.suptitle('Phân phối biến phân loại theo nhóm bệnh', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_categorical_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Lưu: 03_categorical_distributions.png")

    # ============================================================================
    # 6. MA TRẬN TƯƠNG QUAN
    # ============================================================================
    print("\n📊 Tạo ma trận tương quan...")

    # Chuyển đổi tất cả cột sang numeric trước khi tính correlation
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    fig, ax = plt.subplots(figsize=(14, 12))
    corr_matrix = df_numeric.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Ma trận tương quan giữa các đặc trưng', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_correlation_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Lưu: 04_correlation_matrix.png")

    # In top tương quan với target
    print("\n📊 Tương quan với biến mục tiêu (target):")
    target_corr = corr_matrix['target'].drop('target').abs().sort_values(ascending=False)
    for feat, corr_val in target_corr.items():
        direction = "+" if corr_matrix['target'][feat] > 0 else "-"
        print(f"   {feat:12s}: {direction}{corr_val:.3f}")

    # ============================================================================
    # 7. BOXPLOT PHÁT HIỆN OUTLIERS
    # ============================================================================
    print("\n📊 Tạo boxplot phát hiện outliers...")

    fig, axes = plt.subplots(1, len(numeric_features), figsize=(20, 6))

    for i, col in enumerate(numeric_features):
        bp = axes[i].boxplot([df[df['target'] == 0][col].dropna(),
                              df[df['target'] == 1][col].dropna()],
                             tick_labels=['Không\nbệnh', 'Có\nbệnh'],
                             patch_artist=True,
                             boxprops=dict(linewidth=1.5),
                             medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        axes[i].set_title(col, fontsize=13, fontweight='bold')
        axes[i].set_ylabel('Giá trị')

    plt.suptitle('Boxplot phát hiện Outliers theo nhóm bệnh', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_boxplots_outliers.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Lưu: 05_boxplots_outliers.png")

    # ============================================================================
    # 8. PAIRPLOT CÁC ĐẶC TRƯNG QUAN TRỌNG
    # ============================================================================
    print("\n📊 Tạo pairplot các đặc trưng quan trọng nhất...")

    top_features = target_corr.head(4).index.tolist()
    plot_cols = top_features + ['target']

    # Đảm bảo tất cả cột đều numeric
    df_plot = df[plot_cols].apply(pd.to_numeric, errors='coerce').dropna()

    g = sns.pairplot(df_plot, hue='target',
                     palette={0: '#2ecc71', 1: '#e74c3c'},
                     diag_kind='kde',
                     plot_kws={'alpha': 0.6, 's': 30},
                     height=2.5)
    g.fig.suptitle(f'Pairplot: Top 4 đặc trưng tương quan với bệnh tim', y=1.02,
                   fontsize=14, fontweight='bold')
    # Cập nhật legend labels
    handles = g._legend_data.values() if hasattr(g, '_legend_data') else []
    try:
        new_labels = ['Không bệnh', 'Có bệnh']
        g._legend.set_title('')
        for t, l in zip(g._legend.texts, new_labels):
            t.set_text(l)
    except Exception:
        pass

    plt.savefig(os.path.join(OUTPUT_DIR, "06_pairplot_top_features.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Lưu: 06_pairplot_top_features.png")

    # ============================================================================
    # 9. PHÂN TÍCH THEO NHÓM TUỔI
    # ============================================================================
    print("\n📊 Tạo biểu đồ phân tích theo nhóm tuổi...")

    df['age_group'] = pd.cut(df['age'], bins=[20, 35, 45, 55, 65, 80],
                              labels=['20-35', '36-45', '46-55', '56-65', '66-80'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Tỷ lệ bệnh theo nhóm tuổi
    age_disease = df.groupby('age_group')['target'].mean() * 100
    bars = axes[0].bar(age_disease.index.astype(str), age_disease.values,
                       color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(age_disease))),
                       edgecolor='black', linewidth=0.5)
    axes[0].set_title('Tỷ lệ mắc bệnh tim theo nhóm tuổi', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Nhóm tuổi')
    axes[0].set_ylabel('Tỷ lệ mắc bệnh (%)')
    for bar, pct in zip(bars, age_disease.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                     f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Tỷ lệ bệnh theo giới tính
    sex_disease = df.groupby('sex')['target'].mean() * 100
    bars = axes[1].bar(['Nữ', 'Nam'], sex_disease.values,
                       color=['#FF69B4', '#4169E1'], edgecolor='black', linewidth=0.5)
    axes[1].set_title('Tỷ lệ mắc bệnh tim theo giới tính', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Giới tính')
    axes[1].set_ylabel('Tỷ lệ mắc bệnh (%)')
    for bar, pct in zip(bars, sex_disease.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                     f'{pct:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_age_sex_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Lưu: 07_age_sex_analysis.png")

    # Xóa cột tạm
    df.drop('age_group', axis=1, inplace=True)

    # ============================================================================
    # TỔNG KẾT EDA
    # ============================================================================
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH EDA!")
    print("=" * 70)
    print(f"\n📁 Tất cả biểu đồ đã lưu tại: {OUTPUT_DIR}/")
    print(f"   - 01_target_distribution.png")
    print(f"   - 02_numeric_distributions.png")
    print(f"   - 03_categorical_distributions.png")
    print(f"   - 04_correlation_matrix.png")
    print(f"   - 05_boxplots_outliers.png")
    print(f"   - 06_pairplot_top_features.png")
    print(f"   - 07_age_sex_analysis.png")
    print(f"\n📊 Dữ liệu gốc: {os.path.join(DATA_DIR, 'heart.csv')}")
    print(f"📊 Kích thước: {df.shape[0]} mẫu × {df.shape[1]} đặc trưng")
