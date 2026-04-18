@echo off
REM ===================================================================
REM  Safety Zone Combined - one-click launcher
REM  Double-click to start the Streamlit app on http://localhost:8501
REM ===================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo   Safety Zone Combined
echo ============================================================
echo.

REM --- 1. Python check ----------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not on PATH. Install Python 3.10+ from python.org
    echo         and tick "Add Python to PATH".
    pause
    exit /b 1
)

REM --- 2. Create venv if missing ------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    python -m venv .venv || (echo [ERROR] venv creation failed. & pause & exit /b 1)
) else (
    echo [1/4] Virtual environment exists.
)

REM --- 3. Activate venv ---------------------------------------------
echo [2/4] Activating .venv...
call ".venv\Scripts\activate.bat"

REM --- 4. Verify / install deps -------------------------------------
".venv\Scripts\python.exe" -c "import streamlit, folium, faiss, openai" >nul 2>&1
if errorlevel 1 (
    echo [3/4] Installing dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo [ERROR] pip install failed. & pause & exit /b 1)
) else (
    echo [3/4] Dependencies already installed.
)

REM --- 5. Copy .env template if needed ------------------------------
if not exist ".env" if exist ".env.example" copy ".env.example" ".env" >nul

REM --- 6. Free port 8501 if something is holding it -----------------
set "PORT=8501"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    echo         Port %PORT% busy - killing PID %%P
    taskkill /F /PID %%P >nul 2>&1
)

REM --- 7. Launch Streamlit ------------------------------------------
echo [4/4] Launching at http://localhost:%PORT% ...
echo        ^(Close this window to stop the server^)
echo.
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.port %PORT%

pause
