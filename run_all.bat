@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ======================================================================
echo CHAY TOAN BO PIPELINE HE THONG TRI TUE NHAN TAO DU BAO BENH TIM
echo ======================================================================

echo [1/6] Running 01_eda.py...
python -u 01_eda.py
if %errorlevel% neq 0 (
    echo [ERROR] 01_eda.py failed with code %errorlevel%
    exit /b %errorlevel%
)

echo [2/6] Running 02_preprocessing.py...
python -u 02_preprocessing.py
if %errorlevel% neq 0 (
    echo [ERROR] 02_preprocessing.py failed with code %errorlevel%
    exit /b %errorlevel%
)

echo [3/6] Running 03_model_training.py...
python -u 03_model_training.py
if %errorlevel% neq 0 (
    echo [ERROR] 03_model_training.py failed with code %errorlevel%
    exit /b %errorlevel%
)

echo [4/6] Running 04_feature_importance.py...
python -u 04_feature_importance.py
if %errorlevel% neq 0 (
    echo [ERROR] 04_feature_importance.py failed with code %errorlevel%
    exit /b %errorlevel%
)

echo [5/6] Running 05_report.py...
python -u 05_report.py
if %errorlevel% neq 0 (
    echo [ERROR] 05_report.py failed with code %errorlevel%
    exit /b %errorlevel%
)

echo [6/6] Running test_system.py...
python -u test_system.py
if %errorlevel% neq 0 (
    echo [ERROR] test_system.py failed with code %errorlevel%
    exit /b %errorlevel%
)

echo ======================================================================
echo CHUC MUNG! TOAN BO HE THONG AI DA HOAN THANH VA KIEM THU 100%% THANH CONG!
echo De khoi chay giao dien Web Streamlit, vui long chay: run_app.bat
echo ======================================================================
pause
