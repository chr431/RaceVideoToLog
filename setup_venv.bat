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
    echo Existing .venv found. To rebuild: rmdir /s /q .venv ^&^& setup_venv.bat
    echo.
    echo Run .venv\Scripts\activate to enter the venv.
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
.venv\Scripts\python -m pip install --upgrade pip -q

echo.
echo Installing project dependencies ...
.venv\Scripts\python -m pip install -e .
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   decord (self-built, GPU + memory fix)
echo ========================================
echo.
echo This project uses a self-built decord with NVDEC support and CPU
echo memory fixes.  PyPI decord is CPU-only and uses ~10 GB RAM.
echo.

set _DECORD_DIR=.venv\Lib\site-packages\decord
set _COPIED=0

if exist "_decord_build\decord.dll" (
    echo Found _decord_build\ - installing self-built decord ...
    for %%f in (_decord_build\*.dll _decord_build\ffprobe.exe) do (
        copy /Y "%%f" "%_DECORD_DIR%\" >nul
    )
    set _COPIED=1
) else if exist "..\decord\build\Release\decord.dll" (
    echo Found ..\decord\build\Release\ - copying from sibling repo ...
    copy /Y "..\decord\build\Release\decord.dll" "%_DECORD_DIR%\" >nul
    if exist "..\ffmpeg5\bin\*.dll" (
        copy /Y "..\ffmpeg5\bin\*.dll"      "%_DECORD_DIR%\" >nul
        copy /Y "..\ffmpeg5\bin\ffprobe.exe" "%_DECORD_DIR%\" >nul
    )
    set _COPIED=1
)

if %_COPIED%==1 (
    echo   Self-built decord installed - GPU decode ready.
) else (
    echo [WARNING] Self-built decord not found.
    echo   Falling back to PyPI decord (CPU-only, high memory).
    echo.
    echo   To enable GPU decode:
    echo     1. Build decord with -DUSE_CUDA=ON (see repo wiki)
    echo     2. Copy decord.dll + FFmpeg 5.x DLLs + ffprobe.exe to _decord_build\
    echo     3. Re-run this script
)

echo.
echo Installing GPU Python bindings (DLLs loaded from system PATH) ...
.venv\Scripts\python -m pip install --no-deps "tensorrt>=10,<11" "tensorrt_cu13>=10,<11" "tensorrt_cu13_bindings>=10,<11"
.venv\Scripts\python -m pip install cuda-python
echo   GPU video decode (NVDEC) is ready - only requires an NVIDIA GPU with drivers.
echo   GPU OCR (TensorRT) requires CUDA Toolkit + TensorRT installed and on PATH.
echo   Without them, CPU OCR will be used automatically.

echo.
echo ========================================
echo   Setup complete.
echo   Run: .venv\Scripts\python RaceVideoToLog.py
echo ========================================
pause
