@echo off
echo ====================================
echo   راه‌اندازی سیستم جستجوی هوشمند PDF
echo ====================================

echo.
echo 📦 در حال نصب پکیج‌های مورد نیاز...
pip install -r requirements.txt

echo.
echo 📁 ایجاد پوشه‌های مورد نیاز...
if not exist "pdfs" mkdir pdfs
if not exist "vector_db" mkdir vector_db

echo.
echo ⚙️ تنظیمات اولیه:
echo 1. کلید API Gemini خود را در فایل app.py در خط GEMINI_API_KEY وارد کنید
echo 2. فایل‌های PDF خود را در پوشه pdfs قرار دهید
echo 3. فایل index.html را در همین پوشه قرار دهید

echo.
echo 🚀 برای اجرای سرور دستور زیر را اجرا کنید:
echo python app.py

echo.
pause