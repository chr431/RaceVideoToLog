@echo off
cd /d "%~dp0"
chcp 65001 >nul
echo ========================================
echo   RaceVideoToLog - Setup venv
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    echo   Install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    echo Existing .venv found. Delete it first if you want a clean setup.
    echo.
    echo To rebuild: rmdir /s /q .venv ^&^& setup_venv.bat
    pause
    exit /b 0
)

echo Creating .venv ...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create .venv
    pause
    exit /b 1
)

echo Upgrading pip ...
.venv\Scripts\python -m pip install --upgrade pip

echo Installing dependencies ...
.venv\Scripts\python -m pip install -e .
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

echo Installing GPU Python bindings (DLLs loaded from system PATH) ...
.venv\Scripts\python -m pip install --no-deps "tensorrt>=10,<11" "tensorrt_cu13>=10,<11" "tensorrt_cu13_bindings>=10,<11"
.venv\Scripts\python -m pip install cuda-python
echo   To enable TRT inference: install CUDA Toolkit 12.x + TensorRT 10.x and add to PATH.
echo   Without system TRT, CPU will be used instead.

echo.
echo ========================================
echo   Setup complete.
echo   Run: .venv\Scripts\python RaceVideoToLog.py
echo ========================================
pause
