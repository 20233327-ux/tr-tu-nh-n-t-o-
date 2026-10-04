@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ======================================================================
echo KHOI DONG GIAO DIEN WEB CardioAI - HE THONG CHAN DOAN BENH TIM
echo ======================================================================
echo Dang mo ung dung Streamlit tai trinh duyet...
echo Vui long giu cua so nay de duy tri server.
echo ======================================================================

streamlit run app.py
pause
