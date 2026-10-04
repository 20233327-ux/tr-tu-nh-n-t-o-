"""
=============================================================================
HỆ THỐNG TRÍ TUỆ NHÂN TẠO DỰ BÁO & HỖ TRỢ CHẨN ĐOÁN BỆNH TIM MẠCH (CardioAI)
=============================================================================
Nền tảng Hỗ trợ Ra Quyết định Y khoa (Clinical Decision Support System - CDSS)
Tích hợp 6 mô hình Machine Learning:
- Random Forest Classifier (Accuracy: 91.8% - Best Overall)
- XGBoost Classifier (Accuracy: 88.5%)
- Logistic Regression (Accuracy: 88.5% - Recall: 92.9%)
- Support Vector Machine - SVM (Accuracy: 83.6%)
- K-Nearest Neighbors - KNN (Accuracy: 82.0%)
- Multi-Layer Perceptron - Neural Network (Accuracy: 77.0%)
Giải thích quyết định với Explainable AI (SHAP & Permutation Importance)
=============================================================================
"""

import os
import sys
import io
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Fix console encoding on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ============================================================================
# CẤU HÌNH TRANG WEB STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="CardioAI - Hệ Thống Dự Báo & Chẩn Đoán Bệnh Tim Mạch",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# TUỲ BIẾN GIAO DIỆN (CUSTOM CSS)
# ============================================================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #881337 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        max-width: 900px;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    .badge-container {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    
    .badge-pill {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    
    .card-metric {
        background: white;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .prediction-card-high {
        background: linear-gradient(180deg, #fff1f2 0%, #ffe4e6 100%);
        border: 2px solid #f43f5e;
        border-radius: 14px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: 0 10px 20px -5px rgba(244, 63, 94, 0.2);
    }
    
    .prediction-card-moderate {
        background: linear-gradient(180deg, #fffbeb 0%, #fef3c7 100%);
        border: 2px solid #f59e0b;
        border-radius: 14px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: 0 10px 20px -5px rgba(245, 158, 11, 0.2);
    }
    
    .prediction-card-low {
        background: linear-gradient(180deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.2);
    }
    
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .reference-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #334155;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TÊN VÀ ĐỊNH NGHĨA BIẾN LÂM SÀNG
# ============================================================================
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

NUMERIC_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

FEATURE_LABELS_VI = {
    'age': 'Tuổi (Năm)',
    'sex': 'Giới tính',
    'cp': 'Loại đau ngực lâm sàng',
    'trestbps': 'Huyết áp lúc nghỉ (mmHg)',
    'chol': 'Cholesterol huyết thanh (mg/dl)',
    'fbs': 'Đường huyết đói > 120mg/dl',
    'restecg': 'Kết quả điện tâm đồ (ECG)',
    'thalach': 'Nhịp tim tối đa đạt được (bpm)',
    'exang': 'Đau thắt ngực khi gắng sức',
    'oldpeak': 'ST depression (Gắng sức)',
    'slope': 'Độ dốc đoạn ST',
    'ca': 'Số mạch máu chính bị nhuộm (0-3)',
    'thal': 'Thalassemia (Tưới máu cơ tim)'
}

# ============================================================================
# TẢI TẤT CẢ MÔ HÌNH VÀ SCALER (CACHE RESOURCE)
# ============================================================================
@st.cache_resource(show_spinner="Đang tải các mô hình trí tuệ nhân tạo...")
def load_all_models_and_scaler():
    models = {}
    model_configs = {
        'Random Forest': ('random_forest.pkl', 'Mô hình Rừng ngẫu nhiên (Acc: 91.8% - Best Overall)'),
        'XGBoost': ('xgboost.pkl', 'Thuật toán Gradient Boosting tiên tiến (Acc: 88.5%)'),
        'Logistic Regression': ('logistic_regression.pkl', 'Hồi quy Logistic tuyến tính (Recall: 92.9% - Tốt nhất cho Y tế)'),
        'Support Vector Machine (SVM)': ('svm.pkl', 'Mô hình SVM ranh giới phi tuyến (Acc: 83.6%)'),
        'K-Nearest Neighbors (KNN)': ('knn.pkl', 'Mô hình K láng giềng gần nhất (Acc: 82.0%)'),
        'Neural Network (MLP)': ('neural_network_mlp.pkl', 'Mạng nơ-ron nhân tạo đa tầng MLP (Acc: 77.0%)')
    }
    
    scaler = None
    try:
        scaler_path = os.path.join("models", "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Lỗi khi tải Scaler: {e}")

    for display_name, (filename, desc) in model_configs.items():
        filepath = os.path.join("models", filename)
        if os.path.exists(filepath):
            try:
                models[display_name] = {
                    'model': joblib.load(filepath),
                    'desc': desc
                }
            except Exception as e:
                st.warning(f"Không thể đọc mô hình {display_name}: {e}")

    return models, scaler

ALL_MODELS, SCALER = load_all_models_and_scaler()

# Tải bảng số liệu so sánh nếu có
@st.cache_data
def load_comparison_data():
    csv_path = os.path.join("outputs", "model_comparison", "model_comparison.csv")
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path, index_col=0)
        except Exception:
            return None
    return None

COMPARISON_DF = load_comparison_data()

# Cache SHAP Explainer để tăng tốc độ phân tích giải thích
@st.cache_resource(show_spinner=False)
def get_tree_explainer(_model):
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return None

# ============================================================================
# CÁC BỆNH NHÂN MẪU (PRESET PROFILES)
# ============================================================================
PRESET_PATIENTS = {
    "👤 Tự nhập dữ liệu mới": {
        'age': 55, 'sex': 1, 'cp': 1, 'trestbps': 130, 'chol': 240, 'fbs': 0,
        'restecg': 0, 'thalach': 150, 'exang': 0, 'oldpeak': 1.0, 'slope': 2, 'ca': 0, 'thal': 3
    },
    "🟢 Case 1: Khỏe mạnh - Thanh niên (Nguy cơ rất thấp)": {
        'age': 32, 'sex': 0, 'cp': 2, 'trestbps': 112, 'chol': 175, 'fbs': 0,
        'restecg': 0, 'thalach': 178, 'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3
    },
    "🔴 Case 2: Nguy cơ cao - Tắc mạch & Đau thắt ngực (Cần can thiệp gấp)": {
        'age': 64, 'sex': 1, 'cp': 4, 'trestbps': 168, 'chol': 298, 'fbs': 1,
        'restecg': 1, 'thalach': 105, 'exang': 1, 'oldpeak': 2.8, 'slope': 2, 'ca': 2, 'thal': 7
    },
    "🟡 Case 3: Nguy cơ trung bình - Người cao tuổi có tiền sử huyết áp": {
        'age': 58, 'sex': 1, 'cp': 3, 'trestbps': 142, 'chol': 255, 'fbs': 0,
        'restecg': 0, 'thalach': 138, 'exang': 0, 'oldpeak': 1.2, 'slope': 2, 'ca': 1, 'thal': 3
    },
    "🔵 Case 4: Nghi ngờ bệnh tim tiềm ẩn ở phụ nữ trung niên": {
        'age': 53, 'sex': 0, 'cp': 3, 'trestbps': 136, 'chol': 265, 'fbs': 0,
        'restecg': 2, 'thalach': 152, 'exang': 1, 'oldpeak': 1.4, 'slope': 2, 'ca': 0, 'thal': 6
    }
}

# ============================================================================
# HEADER VÀ BANNER HỆ THỐNG
# ============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>❤️</span> CardioAI - Hệ Thống Trí Tuệ Nhân Tạo Dự Báo & Chẩn Đoán Bệnh Tim
    </div>
    <div class="hero-subtitle">
        Hệ thống hỗ trợ ra quyết định lâm sàng (Clinical Decision Support) ứng dụng học máy đa mô hình, 
        giải thích minh bạch thuật toán với Explainable AI (SHAP) và sàng lọc nguy cơ bệnh mạch vành tự động.
    </div>
    <div class="badge-container">
        <span class="badge-pill">🧠 6 Mô hình AI Machine Learning</span>
        <span class="badge-pill">🎯 Độ chính xác: 91.8% (Random Forest)</span>
        <span class="badge-pill">🔬 Độ nhạy y tế (Recall): 92.9%</span>
        <span class="badge-pill">📊 Dữ liệu: UCI Cleveland Heart Disease</span>
        <span class="badge-pill">🔍 Giải thích quyết định với SHAP</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - CẤU HÌNH MÔ HÌNH VÀ CÔNG CỤ
# ============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822143.png", width=70)
    st.markdown("### ⚙️ Cấu Hình Mô Hình AI")
    
    model_choices = list(ALL_MODELS.keys()) + ["🌟 Tất cả mô hình (Ensemble Voting)"]
    selected_model_name = st.selectbox(
        "Lựa chọn mô hình dự đoán:",
        options=model_choices,
        index=0,
        help="Chọn mô hình Machine Learning thực thi suy luận chẩn đoán."
    )
    
    if selected_model_name in ALL_MODELS:
        st.info(f"ℹ️ **Chi tiết:** {ALL_MODELS[selected_model_name]['desc']}")
    else:
        st.info("ℹ️ **Ensemble Voting:** Kết hợp dự báo và tính trung bình xác suất từ tất cả 6 mô hình AI để đạt độ ổn định cao nhất.")
        
    st.markdown("---")
    st.markdown("### 💡 Thông Số Lâm Sàng Bình Thường")
    st.markdown("""
    - **Huyết áp nghỉ:** < 120/80 mmHg
    - **Cholesterol toàn phần:** < 200 mg/dL
    - **Đường huyết lúc đói:** < 100 mg/dL
    - **Nhịp tim tối đa:** ~ (220 - Tuổi) bpm
    - **ST Depression:** = 0.0 mm
    - **Số mạch tắc (ca):** 0 mạch
    """)
    
    st.markdown("---")
    st.markdown("<p style='font-size: 0.85rem; color: #64748b;'>Hệ thống CardioAI CDSS v2.0<br>© 2026 AI Healthcare Lab</p>", unsafe_allow_html=True)

# ============================================================================
# CÁC TAB CHỨC NĂNG CHÍNH CỦA ỨNG DỤNG
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🩺 Chẩn Đoán Cá Nhân", 
    "📁 Dự Đoán Hàng Loạt (Batch)", 
    "📊 So Sánh 6 Mô Hình AI", 
    "🔍 Giải Thích Quyết Định (SHAP)", 
    "📈 Khám Phá Dữ Liệu (EDA)", 
    "ℹ️ Kiến Trúc & Hướng Dẫn"
])

# ============================================================================
# HÀM BỔ TRỢ SUY LUẬN (PREDICTION HELPER)
# ============================================================================
def predict_patient(data_df, model_name):
    """
    Tiền xử lý và dự đoán xác suất nguy cơ mắc bệnh tim mạch.
    Trả về: risk_score (0-100), prediction_label (0 hoặc 1), và dict xác suất từng model.
    """
    if SCALER is None or not ALL_MODELS:
        return 0.0, 0, {}

    # Chuẩn hóa các cột số
    scaled_df = data_df.copy()
    scaled_df[NUMERIC_FEATURES] = SCALER.transform(data_df[NUMERIC_FEATURES])
    
    model_probas = {}
    
    for m_name, m_info in ALL_MODELS.items():
        try:
            m = m_info['model']
            p = m.predict_proba(scaled_df)[0][1]
            model_probas[m_name] = p
        except Exception as e:
            model_probas[m_name] = 0.5

    if model_name in ALL_MODELS:
        chosen_proba = model_probas[model_name]
    else:
        # Ensemble Average
        chosen_proba = float(np.mean(list(model_probas.values())))

    risk_score = chosen_proba * 100
    prediction = 1 if chosen_proba >= 0.5 else 0
    return risk_score, prediction, model_probas, scaled_df

# ============================================================================
# TAB 1: CHẨN ĐOÁN CÁ NHÂN
# ============================================================================
with tab1:
    st.markdown('<p class="section-title">🩺 Phân Tích Nguy Cơ Bệnh Tim Theo Bệnh Nhân</p>', unsafe_allow_html=True)
    
    # 1. Chọn hồ sơ mẫu (Quick presets)
    preset_choice = st.selectbox(
        "⚡ Chọn hồ sơ bệnh nhân mẫu để kiểm thử nhanh:",
        options=list(PRESET_PATIENTS.keys()),
        index=0
    )
    p_data = PRESET_PATIENTS[preset_choice]

    # 2. Form nhập liệu chỉ số lâm sàng
    with st.form("individual_patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📌 Thông số sinh hiệu & Khám cơ bản")
            age = st.number_input("Tuổi bệnh nhân", min_value=18, max_value=100, value=int(p_data['age']))
            
            sex_idx = 0 if p_data['sex'] == 1 else 1
            sex = st.selectbox("Giới tính", options=[(1, "Nam"), (0, "Nữ")], index=sex_idx, format_func=lambda x: x[1])
            
            cp_options = [
                (1, "1: Typical Angina - Cơn đau thắt ngực điển hình"),
                (2, "2: Atypical Angina - Đau thắt ngực không điển hình"),
                (3, "3: Non-anginal Pain - Đau không do tim"),
                (4, "4: Asymptomatic - Không có triệu chứng đau ngực")
            ]
            cp_map = {1: 0, 2: 1, 3: 2, 4: 3}
            cp_idx = cp_map.get(int(p_data['cp']), 0)
            cp = st.selectbox(
                "Loại đau thắt ngực (Chest Pain Type)", 
                options=cp_options,
                index=cp_idx,
                format_func=lambda x: x[1],
                help="Phân loại cơn đau theo chuẩn lâm sàng (1: Điển hình, 2: Không điển hình, 3: Không do tim, 4: Không triệu chứng)."
            )
            
            trestbps = st.number_input("Huyết áp tâm thu lúc nghỉ (mmHg)", min_value=70, max_value=240, value=int(p_data['trestbps']))
            chol = st.number_input("Cholesterol toàn phần trong huyết thanh (mg/dl)", min_value=100, max_value=600, value=int(p_data['chol']))
            
            fbs_idx = int(p_data['fbs'])
            fbs = st.selectbox(
                "Đường huyết lúc đói > 120 mg/dl (Fasting Blood Sugar)",
                options=[(0, "0: Không (<= 120 mg/dl - Bình thường)"), (1, "1: Có (> 120 mg/dl - Nguy cơ tiểu đường)")],
                index=fbs_idx,
                format_func=lambda x: x[1]
            )
            
            restecg_idx = int(p_data['restecg'])
            restecg = st.selectbox(
                "Kết quả điện tâm đồ lúc nghỉ (Resting ECG)",
                options=[
                    (0, "0: Bình thường (Normal)"),
                    (1, "1: Bất thường sóng ST-T (ST-T wave abnormality)"),
                    (2, "2: Phì đại thất trái theo tiêu chuẩn Estes (Left ventricular hypertrophy)")
                ],
                index=restecg_idx,
                format_func=lambda x: x[1]
            )

        with col2:
            st.markdown("##### 📌 Thử nghiệm gắng sức & Chẩn đoán hình ảnh")
            thalach = st.number_input("Nhịp tim tối đa đạt được (bpm)", min_value=60, max_value=220, value=int(p_data['thalach']))
            
            exang_idx = int(p_data['exang'])
            exang = st.selectbox(
                "Xuất hiện đau ngực khi gắng sức (Exercise Induced Angina)",
                options=[(0, "0: Không"), (1, "1: Có đau ngực khi gắng sức")],
                index=exang_idx,
                format_func=lambda x: x[1]
            )
            
            oldpeak = st.number_input(
                "ST Depression (Đoạn ST chênh xuống so với lúc nghỉ - mm)",
                min_value=0.0, max_value=7.0, value=float(p_data['oldpeak']), step=0.1
            )
            
            slope_options = [
                (1, "1: Upsloping - Dốc lên (Tiên lượng tốt)"),
                (2, "2: Flat - Đi ngang (Nghi ngờ thiếu máu cơ tim)"),
                (3, "3: Downsloping - Dốc xuống (Dấu hiệu thiếu máu cục bộ nặng)")
            ]
            slope_map = {1: 0, 2: 1, 3: 2}
            slope_idx = slope_map.get(int(p_data['slope']), 0)
            slope = st.selectbox(
                "Độ dốc đoạn ST ở đỉnh gắng sức (Peak exercise ST segment slope)",
                options=slope_options,
                index=slope_idx,
                format_func=lambda x: x[1]
            )
            
            ca_idx = int(p_data['ca'])
            ca = st.selectbox(
                "Số mạch máu chính bị nhuộm qua soi huỳnh quang (0-3)",
                options=[0, 1, 2, 3],
                index=ca_idx,
                help="Số lượng động mạch vành chính bị tắc nghẽn hoặc hẹp dòng chảy."
            )
            
            thal_map = {3: 0, 6: 1, 7: 2}
            thal_val = int(p_data['thal'])
            thal_idx = thal_map.get(thal_val, 0)
            thal = st.selectbox(
                "Tưới máu cơ tim Thalassemia (Nuclear Stress Test)",
                options=[
                    (3, "3: Bình thường (Normal flow)"),
                    (6, "6: Khuyết tật cố định (Fixed defect - Mô sẹo nhồi máu cũ)"),
                    (7, "7: Khuyết tật có thể phục hồi (Reversible defect - Đang thiếu máu cơ tim)")
                ],
                index=thal_idx,
                format_func=lambda x: x[1]
            )
            
        submit_btn = st.form_submit_button(
            label="🔬 BẮT ĐẦU PHÂN TÍCH NGUY CƠ LÂM SÀNG",
            use_container_width=True
        )

    # 3. Xử lý suy luận khi nhấn nút Submit hoặc chọn preset
    should_evaluate = submit_btn or (preset_choice != "👤 Tự nhập dữ liệu mới")
    if should_evaluate:
        # Tạo DataFrame dữ liệu
        patient_row = pd.DataFrame([[
            age, sex[0], cp[0], trestbps, chol, fbs[0], restecg[0],
            thalach, exang[0], oldpeak, slope[0], ca, thal[0]
        ]], columns=FEATURE_NAMES)
        
        for col in patient_row.columns:
            if col not in NUMERIC_FEATURES:
                patient_row[col] = patient_row[col].astype(int)

        risk_score, prediction, model_probas, scaled_df = predict_patient(patient_row, selected_model_name)
        
        st.markdown("---")
        st.markdown('<p class="section-title">📊 Kết Quả Đánh Giá Lâm Sàng & Khuyến Nghị</p>', unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1.1, 1.2])
        
        with res_col1:
            # Hộp hiển thị rủi ro
            if risk_score >= 65.0:
                st.markdown(f"""
                <div class="prediction-card-high">
                    <h3 style="color: #e11d48; margin-top: 0;">🚨 NGUY CƠ CAO (Xác suất: {risk_score:.1f}%)</h3>
                    <p style="font-size: 1.05rem; color: #1e293b; margin-bottom: 8px;">
                        Mô hình AI chẩn đoán bệnh nhân có <strong>khả năng cao mắc bệnh lý động mạch vành / bệnh tim mạch</strong>.
                    </p>
                    <p style="font-size: 0.95rem; color: #475569;">
                        Mô hình đang sử dụng: <strong>{selected_model_name}</strong>. Cần liên hệ bác sĩ chuyên khoa tim mạch để thực hiện cận lâm sàng chuyên sâu.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif risk_score >= 35.0:
                st.markdown(f"""
                <div class="prediction-card-moderate">
                    <h3 style="color: #d97706; margin-top: 0;">⚠️ NGUY CƠ TRUNG BÌNH (Xác suất: {risk_score:.1f}%)</h3>
                    <p style="font-size: 1.05rem; color: #1e293b; margin-bottom: 8px;">
                        Mô hình AI nhận định bệnh nhân có <strong>nguy cơ tim mạch ở mức cảnh báo</strong>.
                    </p>
                    <p style="font-size: 0.95rem; color: #475569;">
                        Các chỉ số huyết áp, cholesterol hoặc nghiệm pháp gắng sức có biểu hiện bất thường nhẹ. Cần điều chỉnh lối sống và tái khám định kỳ.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card-low">
                    <h3 style="color: #059669; margin-top: 0;">✅ NGUY CƠ THẤP (Xác suất: {risk_score:.1f}%)</h3>
                    <p style="font-size: 1.05rem; color: #1e293b; margin-bottom: 8px;">
                        Mô hình AI nhận định bệnh nhân có <strong>hệ tim mạch tương đối an toàn và khỏe mạnh</strong>.
                    </p>
                    <p style="font-size: 0.95rem; color: #475569;">
                        Các chỉ số sinh hiệu và kết quả gắng sức nằm trong phạm vi sinh lý bình thường. Tiếp tục duy trì chế độ sinh hoạt lành mạnh.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.write(f"**Thang đo nguy cơ:** {risk_score:.1f} / 100%")
            st.progress(int(risk_score))
            
            # Bảng đồng thuận 6 mô hình AI
            st.markdown("##### 🤝 Đồng thuận giữa 6 mô hình AI:")
            consensus_data = []
            for m_n, p_val in model_probas.items():
                status_icon = "🔴 Nguy cơ cao" if p_val >= 0.5 else "🟢 Nguy cơ thấp"
                consensus_data.append({
                    "Mô hình": m_n,
                    "Xác suất bệnh (%)": f"{p_val*100:.1f}%",
                    "Kết luận": status_icon
                })
            st.dataframe(pd.DataFrame(consensus_data), use_container_width=True, hide_index=True)

        with res_col2:
            st.markdown("##### 🔍 Giải thích đóng góp từng yếu tố (Local SHAP):")
            st.caption("Biểu đồ thể hiện mức độ đóng góp của từng chỉ số vào kết quả dự đoán của bệnh nhân này.")
            
            try:
                # Dùng TreeExplainer với Random Forest hoặc XGBoost
                rf_model = ALL_MODELS.get('Random Forest', {}).get('model', None)
                if rf_model is not None:
                    explainer = get_tree_explainer(rf_model)
                    if explainer is not None:
                        sv = explainer.shap_values(scaled_df)
                        if isinstance(sv, list):
                            shap_sample = sv[1][0]
                        elif len(sv.shape) == 3:
                            shap_sample = sv[0, :, 1]
                        else:
                            shap_sample = sv[0]
                    
                    vi_labels = [FEATURE_LABELS_VI.get(f, f) for f in FEATURE_NAMES]
                    shap_df = pd.DataFrame({
                        'Đặc trưng': vi_labels,
                        'Tác động (SHAP)': shap_sample,
                        'Giá trị thực': patient_row.iloc[0].values
                    })
                    shap_df['Độ lớn'] = shap_df['Tác động (SHAP)'].abs()
                    top_shap = shap_df.sort_values(by='Độ lớn', ascending=True).tail(7)
                    
                    fig, ax = plt.subplots(figsize=(8, 4.8))
                    bar_colors = ['#e11d48' if x > 0 else '#2563eb' for x in top_shap['Tác động (SHAP)']]
                    ax.barh(top_shap['Đặc trưng'], top_shap['Tác động (SHAP)'], color=bar_colors, edgecolor='black', linewidth=0.5)
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                    ax.set_xlabel('Mức đóng góp SHAP (Đỏ: Tăng nguy cơ | Xanh: Giảm nguy cơ)', fontsize=9)
                    ax.set_title('Top 7 chỉ số tác động mạnh nhất đến ca này', fontsize=11, fontweight='bold')
                    
                    for i, bar in enumerate(ax.patches):
                        orig_val = top_shap['Giá trị thực'].iloc[i]
                        v_str = f"{int(orig_val)}" if orig_val == int(orig_val) else f"{orig_val:.1f}"
                        x_pos = bar.get_width()
                        align = 'left' if x_pos > 0 else 'right'
                        ax.text(x_pos + (0.005 if x_pos > 0 else -0.005), bar.get_y() + bar.get_height()/2, 
                                f"{v_str}", va='center', ha=align, fontsize=8, fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.info(f"Biểu đồ SHAP chưa sẵn sàng cho ca này: {e}")

        # Lời khuyên lâm sàng & Tải báo cáo chẩn đoán
        st.markdown("---")
        rec_col1, rec_col2 = st.columns([1.5, 1])
        
        with rec_col1:
            st.markdown("##### 📋 Kế Hoạch Chăm Sóc & Đề Xuất Lâm Sàng:")
            if risk_score >= 65:
                st.error("""
                - **Chỉ định cận lâm sàng khẩn:** Đề nghị thực hiện Chụp mạch vành qua da (Coronary Angiography) hoặc Siêu âm tim gắng sức (Stress Echocardiography).
                - **Theo dõi:** Đặt Holter ECG 24 giờ để đánh giá các cơn rối loạn nhịp tim hoặc thiếu máu cơ tim thầm lặng.
                - **Điều trị:** Tham vấn bác sĩ chuyên khoa tim mạch về liệu pháp kháng kết tập tiểu cầu, nhóm thuốc Statin và ức chế men chuyển.
                """)
            elif risk_score >= 35:
                st.warning("""
                - **Theo dõi định kỳ:** Kiểm tra huyết áp và lipid máu mỗi 3-6 tháng.
                - **Điều chỉnh dinh dưỡng:** Áp dụng chế độ ăn Địa Trung Hải (DASH), giảm lượng muối (< 5g/ngày), hạn chế mỡ động vật.
                - **Vận động thể lực:** Tập thể dục vừa sức ít nhất 150 phút/tuần (đi bộ nhanh, bơi lội), tránh gắng sức đột ngột.
                """)
            else:
                st.success("""
                - **Duy trì sức khỏe:** Tiếp tục chế độ dinh dưỡng giàu chất xơ, duy trì chỉ số khối cơ thể (BMI) ở mức 18.5 - 23.
                - **Tầm soát:** Khám sức khỏe tim mạch định kỳ 1 năm/lần.
                - **Phòng ngừa:** Không hút thuốc lá, kiểm soát căng thẳng và ngủ đủ giấc 7-8 tiếng/ngày.
                """)
                
        with rec_col2:
            st.markdown("##### 📄 Xuất Báo Cáo Chẩn Đoán:")
            report_text = f"""
============================================================
BÁO CÁO KẾT QUẢ PHÂN TÍCH NGUY CƠ TIM MẠCH - CARDIOAI CDSS
============================================================
Thời gian phân tích: {time.strftime('%Y-%m-%d %H:%M:%S')}
Mô hình AI thực thi: {selected_model_name}

THÔNG TIN BỆNH NHÂN:
- Tuổi: {age} tuổi | Giới tính: {"Nam" if sex[0]==1 else "Nữ"}
- Huyết áp lúc nghỉ: {trestbps} mmHg
- Cholesterol toàn phần: {chol} mg/dL
- Đường huyết đói: {"> 120 mg/dL" if fbs[0]==1 else "<= 120 mg/dL"}
- Nhịp tim tối đa: {thalach} bpm
- ST Depression: {oldpeak} mm | Slope: {slope[0]}
- Số mạch máu hẹp (ca): {ca} | Thalassemia: {thal[0]}

KẾT QUẢ ĐÁNH GIÁ AI:
- Xác suất nguy cơ: {risk_score:.2f}%
- Phân loại rủi ro: {"NGUY CƠ CAO (CÓ BỆNH LÝ)" if risk_score>=50 else "NGUY CƠ THẤP (KHÔNG CÓ BỆNH LÝ)"}
- Khuyến nghị: {"Cần chuyển khám chuyên khoa tim mạch khẩn cấp" if risk_score>=65 else "Theo dõi định kỳ và điều chỉnh lối sống"}

* Lưu ý: Kết quả do AI tính toán chỉ mang tính chất tham khảo cho nhân viên y tế.
============================================================
            """
            st.download_button(
                label="📥 Tải Phiếu Báo Cáo (.txt)",
                data=report_text,
                file_name=f"CardioAI_Report_Patient_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True
            )

# ============================================================================
# TAB 2: DỰ ĐOÁN HÀNG LOẠT (BATCH PREDICTION)
# ============================================================================
with tab2:
    st.markdown('<p class="section-title">📁 Dự Đoán & Sàng Lọc Hàng Loạt (Batch Screening)</p>', unsafe_allow_html=True)
    st.write("Tải lên danh sách hồ sơ bệnh nhân từ tệp CSV để tiến hành quét nguy cơ tự động quy mô lớn cho bệnh viện hoặc phòng khám.")
    
    b_col1, b_col2 = st.columns([1.5, 1])
    
    with b_col1:
        uploaded_file = st.file_uploader("Chọn tệp CSV danh sách bệnh nhân:", type=["csv"])
    
    with b_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample_btn = st.button("🧪 Dùng Dữ Liệu Kiểm Tra Mẫu (10 ca có sẵn)", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Đã tải lên tệp: {len(batch_df)} dòng dữ liệu.")
        except Exception as e:
            st.error(f"Lỗi khi đọc file CSV: {e}")
    elif use_sample_btn or os.path.exists(os.path.join("data", "sample_patients.csv")):
        sample_path = os.path.join("data", "sample_patients.csv")
        if os.path.exists(sample_path):
            batch_df = pd.read_csv(sample_path)
            st.info(f"📂 Đang hiển thị dữ liệu kiểm tra mẫu ({len(batch_df)} bệnh nhân).")

    if batch_df is not None:
        # Kiểm tra xem có đủ cột không
        missing_cols = [c for c in FEATURE_NAMES if c not in batch_df.columns]
        if missing_cols:
            st.error(f"❌ File CSV thiếu các cột bắt buộc: {missing_cols}")
        else:
            st.markdown("##### 📋 Dữ liệu đầu vào:")
            st.dataframe(batch_df.head(5), use_container_width=True)
            
            if st.button("🚀 BẮT ĐẦU PHÂN TÍCH TOÀN BỘ DANH SÁCH", use_container_width=True):
                with st.spinner("Đang chạy dự đoán cho từng bệnh nhân..."):
                    results_list = []
                    scaled_batch = batch_df[FEATURE_NAMES].copy()
                    scaled_batch[NUMERIC_FEATURES] = SCALER.transform(scaled_batch[NUMERIC_FEATURES])
                    
                    if selected_model_name in ALL_MODELS:
                        target_model = ALL_MODELS[selected_model_name]['model']
                        probas = target_model.predict_proba(scaled_batch)[:, 1]
                    else:
                        # Chế độ Ensemble Voting: tính trung bình xác suất từ cả 6 mô hình
                        all_p = [m_info['model'].predict_proba(scaled_batch)[:, 1] for m_info in ALL_MODELS.values()]
                        probas = np.mean(all_p, axis=0)
                    
                    for idx, p in enumerate(probas):
                        risk_pct = p * 100
                        if risk_pct >= 65:
                            status = "🔴 Nguy cơ cao"
                        elif risk_pct >= 35:
                            status = "🟡 Nguy cơ trung bình"
                        else:
                            status = "🟢 An toàn (Nguy cơ thấp)"
                            
                        results_list.append({
                            'Mã BN': f"BN-{idx+1:03d}",
                            'Tuổi': int(batch_df['age'].iloc[idx]),
                            'Giới tính': "Nam" if batch_df['sex'].iloc[idx] == 1 else "Nữ",
                            'Huyết áp': int(batch_df['trestbps'].iloc[idx]),
                            'Cholesterol': int(batch_df['chol'].iloc[idx]),
                            'Xác suất (%)': f"{risk_pct:.1f}%",
                            'Phân loại': status
                        })
                        
                    res_df = pd.DataFrame(results_list)
                    
                    # KPI Metrics
                    st.markdown("---")
                    st.markdown("##### 📊 Thống Kê Tổng Quan Sàng Lọc:")
                    m1, m2, m3, m4 = st.columns(4)
                    
                    total_pts = len(res_df)
                    high_risk_count = sum(1 for r in results_list if "🔴" in r['Phân loại'])
                    med_risk_count = sum(1 for r in results_list if "🟡" in r['Phân loại'])
                    low_risk_count = sum(1 for r in results_list if "🟢" in r['Phân loại'])
                    
                    m1.metric("Tổng số bệnh nhân", f"{total_pts} ca")
                    m2.metric("Nguy cơ cao", f"{high_risk_count} ca", f"{high_risk_count/total_pts*100:.1f}%", delta_color="inverse")
                    m3.metric("Nguy cơ trung bình", f"{med_risk_count} ca", f"{med_risk_count/total_pts*100:.1f}%", delta_color="off")
                    m4.metric("An toàn (Thấp)", f"{low_risk_count} ca", f"{low_risk_count/total_pts*100:.1f}%")
                    
                    # Biểu đồ phân bổ
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
                        ax_pie.pie(
                            [high_risk_count, med_risk_count, low_risk_count],
                            labels=['Nguy cơ cao', 'Cảnh báo', 'An toàn'],
                            colors=['#f43f5e', '#f59e0b', '#10b981'],
                            autopct='%1.1f%%',
                            startangle=140
                        )
                        ax_pie.set_title('Tỷ lệ phân loại rủi ro', fontweight='bold')
                        st.pyplot(fig_pie)
                        plt.close()
                        
                    with c_col2:
                        fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
                        ax_sc.scatter(
                            batch_df['age'], probas * 100, 
                            c=['#f43f5e' if p >= 0.65 else ('#f59e0b' if p >= 0.35 else '#10b981') for p in probas],
                            s=60, edgecolors='black', linewidth=0.5
                        )
                        ax_sc.axhline(65, color='#f43f5e', linestyle='--', alpha=0.5, label='Ngưỡng cao')
                        ax_sc.axhline(35, color='#f59e0b', linestyle='--', alpha=0.5, label='Ngưỡng cảnh báo')
                        ax_sc.set_xlabel('Tuổi')
                        ax_sc.set_ylabel('Xác suất bệnh (%)')
                        ax_sc.set_title('Phân bổ nguy cơ theo độ tuổi', fontweight='bold')
                        ax_sc.legend(fontsize=8)
                        st.pyplot(fig_sc)
                        plt.close()

                    # Bảng kết quả và nút tải về
                    st.markdown("##### 📑 Bảng Chi Tiết Kết Quả Sàng Lọc:")
                    st.dataframe(res_df, use_container_width=True)
                    
                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải Kết Quả Sàng Lọc (.csv)",
                        data=csv_export,
                        file_name=f"CardioAI_Batch_Results_{int(time.time())}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ============================================================================
# TAB 3: SO SÁNH 6 MÔ HÌNH AI
# ============================================================================
with tab3:
    st.markdown('<p class="section-title">📊 Bảng Xếp Hạng & So Sánh 6 Mô Hình Machine Learning</p>', unsafe_allow_html=True)
    st.write("Đánh giá toàn diện 6 mô hình trên tập kiểm thử độc lập (Test Set 20% - Stratified) qua 5 chỉ số lâm sàng chuẩn.")
    
    if COMPARISON_DF is not None:
        metrics_cols = [c for c in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'CV Mean', 'Time (s)'] if c in COMPARISON_DF.columns]
        
        # Style Dataframe
        st.dataframe(
            COMPARISON_DF[metrics_cols].style.highlight_max(axis=0, color='#dcfce7').format("{:.4f}"),
            use_container_width=True
        )
        
        st.markdown("""
        > 💡 **Khuyến nghị Y khoa (Clinical Insight):**  
        > Trong chẩn đoán y tế, **Recall (Độ nhạy)** là chỉ số quan trọng bậc nhất vì việc bỏ sót một bệnh nhân có bệnh tim (Âm tính giả - False Negative) nguy hiểm hơn nhiều so với việc cảnh báo nhầm (Dương tính giả - False Positive).  
        > **Logistic Regression** và **Random Forest** đạt độ nhạy vượt trội **92.9%**, phát hiện hầu hết các ca bệnh tiềm ẩn.
        """)
        
        # Biểu đồ so sánh
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("##### 📈 Biểu đồ so sánh Metrics & ROC Curves:")
            roc_path = os.path.join("outputs", "model_comparison", "02_roc_curves.png")
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Đường cong ROC và diện tích dưới đường cong (AUC) của 6 mô hình", use_container_width=True)
                
        with chart_col2:
            st.markdown("##### 🎯 Ma trận nhầm lẫn (Confusion Matrices):")
            cm_path = os.path.join("outputs", "model_comparison", "03_confusion_matrices.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Ma trận nhầm lẫn chi tiết (True Positives, False Negatives)", use_container_width=True)
                
        st.markdown("##### 📊 So sánh tổng hợp đa chỉ số:")
        comp_path = os.path.join("outputs", "model_comparison", "01_model_comparison.png")
        if os.path.exists(comp_path):
            st.image(comp_path, caption="Biểu đồ so sánh toàn diện 6 thuật toán", use_container_width=True)

# ============================================================================
# TAB 4: GIẢI THÍCH QUYẾT ĐỊNH (SHAP & FEATURE IMPORTANCE)
# ============================================================================
with tab4:
    st.markdown('<p class="section-title">🔍 Giải Thích Thuật Toán AI (Explainable AI - XAI)</p>', unsafe_allow_html=True)
    st.write("Minh bạch hóa các quyết định của trí tuệ nhân tạo nhằm xây dựng sự tin cậy trong thực hành y khoa.")
    
    st.markdown("### 🏆 Top 5 Yếu Tố Nguy Cơ Tim Mạch Hàng Đầu Đã Được AI Xác Thực:")
    st.markdown("""
    1. **Số lượng mạch máu chính bị tắc (ca):** Yếu tố quan trọng nhất. Mỗi nhánh động mạch vành bị hẹp làm tăng vọt nguy cơ nhồi máu cơ tim.
    2. **Loại đau thắt ngực (cp):** Cơn đau điển hình khi gắng sức liên quan mật thiết đến thiếu máu cơ tim cục bộ.
    3. **Đoạn ST chênh xuống khi gắng sức (oldpeak):** Thể hiện mức độ thiếu oxy của tế bào cơ tim khi tim phải đập nhanh.
    4. **Nhịp tim tối đa đạt được (thalach):** Người có thể trạng tim kém thường không thể đạt được tần số tim gắng sức sinh lý.
    5. **Tưới máu cơ tim Thalassemia (thal):** Khiếm khuyết cố định hoặc có thể phục hồi phản ánh vùng mô tim bị hoại tử hoặc thiếu máu.
    """)
    
    xai_col1, xai_col2 = st.columns(2)
    
    with xai_col1:
        st.markdown("##### 📌 Bảng xếp hạng mức độ ảnh hưởng tổng hợp:")
        rank_path = os.path.join("outputs", "feature_importance", "10_overall_feature_ranking.png")
        if os.path.exists(rank_path):
            st.image(rank_path, caption="Xếp hạng trung bình từ 4 phương pháp phân tích độc lập", use_container_width=True)
            
    with xai_col2:
        st.markdown("##### 📌 Phân bố SHAP Values toàn cục (Random Forest):")
        shap_rf_path = os.path.join("outputs", "feature_importance", "05_shap_summary_rf.png")
        if os.path.exists(shap_rf_path):
            st.image(shap_rf_path, caption="Mỗi điểm biểu thị một bệnh nhân (Màu đỏ: giá trị cao, Màu xanh: giá trị thấp)", use_container_width=True)
            
    st.markdown("##### 📌 Tầm quan trọng hoán vị (Permutation Importance):")
    perm_path = os.path.join("outputs", "feature_importance", "04_permutation_importance.png")
    if os.path.exists(perm_path):
        st.image(perm_path, caption="Độ sụt giảm chính xác khi xáo trộn từng đặc trưng trên cả 6 mô hình", use_container_width=True)

# ============================================================================
# TAB 5: KHÁM PHÁ DỮ LIỆU (INTERACTIVE EDA)
# ============================================================================
with tab5:
    st.markdown('<p class="section-title">📈 Khám Phá Tập Dữ Liệu Huấn Luyện (UCI Cleveland)</p>', unsafe_allow_html=True)
    
    raw_data_path = os.path.join("data", "heart.csv")
    if os.path.exists(raw_data_path):
        df_heart = pd.read_csv(raw_data_path)
        
        st.markdown("##### 🔍 Lọc và tra cứu bệnh nhân trong tập dữ liệu gốc:")
        f_col1, f_col2, f_col3 = st.columns(3)
        
        with f_col1:
            age_filter = st.slider("Khoảng tuổi:", int(df_heart['age'].min()), int(df_heart['age'].max()), (40, 65))
        with f_col2:
            gender_filter = st.selectbox("Giới tính lọc:", ["Tất cả", "Nam", "Nữ"])
        with f_col3:
            disease_filter = st.selectbox("Tình trạng bệnh:", ["Tất cả", "Có bệnh tim (1)", "Không bệnh tim (0)"])
            
        filtered_df = df_heart[(df_heart['age'] >= age_filter[0]) & (df_heart['age'] <= age_filter[1])]
        if gender_filter == "Nam":
            filtered_df = filtered_df[filtered_df['sex'] == 1]
        elif gender_filter == "Nữ":
            filtered_df = filtered_df[filtered_df['sex'] == 0]
            
        if disease_filter == "Có bệnh tim (1)":
            filtered_df = filtered_df[filtered_df['target'] == 1]
        elif disease_filter == "Không bệnh tim (0)":
            filtered_df = filtered_df[filtered_df['target'] == 0]
            
        st.write(f"Tìm thấy **{len(filtered_df)}** bệnh nhân thỏa mãn tiêu chí:")
        st.dataframe(filtered_df.head(10), use_container_width=True)
        
        eda_col1, eda_col2 = st.columns(2)
        with eda_col1:
            st.markdown("##### 📌 Ma trận tương quan giữa các đặc trưng:")
            corr_path = os.path.join("outputs", "eda", "04_correlation_matrix.png")
            if os.path.exists(corr_path):
                st.image(corr_path, caption="Hệ số tương quan Pearson giữa 14 biến lâm sàng", use_container_width=True)
        with eda_col2:
            st.markdown("##### 📌 Phân tích theo độ tuổi và giới tính:")
            agesex_path = os.path.join("outputs", "eda", "07_age_sex_analysis.png")
            if os.path.exists(agesex_path):
                st.image(agesex_path, caption="Tỷ lệ mắc bệnh tim mạch theo lứa tuổi và giới tính", use_container_width=True)

# ============================================================================
# TAB 6: KIẾN TRÚC & HƯỚNG DẪN
# ============================================================================
with tab6:
    st.markdown('<p class="section-title">ℹ️ Kiến Trúc Hệ Thống & Hướng Dẫn Kỹ Thuật</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🏗️ Quy Trình Xử Lý Dữ Liệu Đầu - Cuối (End-to-End Pipeline)
    ```text
    [Dữ liệu UCI Cleveland (303 ca)] 
                 │
                 ▼
    [01_eda.py: Khám phá phân phối, phát hiện tương quan & outliers]
                 │
                 ▼
    [02_preprocessing.py: Xử lý missing values, chuẩn hóa StandardScaler, Stratified Split 80/20]
                 │
                 ▼
    [03_model_training.py: Huấn luyện 6 mô hình ML, tối ưu Hyperparameters với Stratified 5-Fold CV]
                 │
                 ▼
    [04_feature_importance.py: Trích xuất SHAP values, Permutation Importance, Gini & Gain]
                 │
                 ▼
    [05_report.py: Xuất báo cáo tổng kết, biểu đồ so sánh ROC/AUC & Confusion Matrix]
                 │
                 ▼
    [app.py: Ứng dụng Web CDSS tương tác trực quan với Streamlit & SHAP]
    ```
    
    ### 📚 Bảng Danh Mục 13 Thuộc Tính Lâm Sàng
    | Biến | Tên tiếng Việt | Đơn vị / Giá trị | Ý nghĩa lâm sàng |
    |---|---|---|---|
    | `age` | Tuổi | Năm (29 - 77) | Tuổi càng cao nguy cơ xơ vữa động mạch càng lớn |
    | `sex` | Giới tính | 1=Nam, 0=Nữ | Nam giới có tỷ lệ mắc bệnh mạch vành sớm hơn |
    | `cp` | Loại đau thắt ngực | 1, 2, 3, 4 | 1: Đau thắt ngực điển hình, 2: Không điển hình, 3: Không do tim, 4: Không triệu chứng |
    | `trestbps` | Huyết áp lúc nghỉ | mmHg | Tăng huyết áp gây tổn thương thành mạch |
    | `chol` | Cholesterol huyết thanh | mg/dl | Mức mỡ máu cao gây mảng xơ vữa |
    | `fbs` | Đường huyết lúc đói | >120 mg/dl (1/0) | Đái tháo đường là yếu tố nguy cơ tim mạch chính |
    | `restecg` | Điện tâm đồ lúc nghỉ | 0, 1, 2 | Phát hiện phì đại thất hoặc biến đổi sóng ST-T |
    | `thalach` | Nhịp tim tối đa | bpm | Đáp ứng nhịp tim khi gắng sức |
    | `exang` | Đau ngực gắng sức | 1=Có, 0=Không | Dấu hiệu điển hình của hẹp động mạch vành |
    | `oldpeak` | ST depression | mm (0.0 - 6.2) | Mức độ chênh xuống của đoạn ST khi gắng sức |
    | `slope` | Độ dốc đoạn ST | 1, 2, 3 | 1: Dốc lên (Upsloping), 2: Đi ngang (Flat), 3: Dốc xuống (Downsloping) |
    | `ca` | Số mạch máu chính | 0, 1, 2, 3 | Số nhánh động mạch vành bị hẹp qua soi huỳnh quang |
    | `thal` | Thalassemia | 3, 6, 7 | Khuyết tật tưới máu cơ tim qua xạ hình tim |

    ---
    ### ⚠️ Khuyến Cáo Y Khoa (Medical Disclaimer)
    > Hệ thống **CardioAI** được xây dựng nhằm mục đích nghiên cứu, học tập và hỗ trợ nhân viên y tế trong việc sàng lọc ban đầu.  
    > Kết quả dự đoán của mô hình **không thể thay thế** chẩn đoán chuyên môn, thăm khám lâm sàng hoặc kết luận từ bác sĩ chuyên khoa tim mạch.
    """)

# ============================================================================
# FOOTER HỆ THỐNG
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.9rem; padding: 15px 0;">
    ❤️ <strong>CardioAI Clinical Decision Support System</strong> | Bản quyền © 2026 AI Healthcare Project.<br>
    Được tối ưu hóa cho nghiên cứu y khoa, học máy lâm sàng và chuẩn đoán phân loại tự động.
</div>
""", unsafe_allow_html=True)
