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

rem 用 goto 分支替代 if 括号块内的 for 循环（cmd 的括号解析 bug
rem 会使嵌套 for 报 ". was unexpected at this time."）。
if exist "_decord_build\decord.dll" goto :use_decord_build
if exist "..\decord\build\Release\decord.dll" goto :use_decord_sibling
goto :decord_missing

:use_decord_build
echo Found _decord_build\ - installing self-built decord ...
rem 先清理 PyPI decord 自带的 FFmpeg 4.x DLL（避免与 FFmpeg 8 DLL 混杂）
del /q "%_DECORD_DIR%\avcodec-58.dll" "%_DECORD_DIR%\avformat-58.dll" "%_DECORD_DIR%\avutil-56.dll" 2>nul
del /q "%_DECORD_DIR%\avfilter-7.dll" "%_DECORD_DIR%\avdevice-58.dll" "%_DECORD_DIR%\swresample-3.dll" "%_DECORD_DIR%\swscale-5.dll" "%_DECORD_DIR%\postproc-55.dll" 2>nul
for %%f in (_decord_build\*.dll _decord_build\ffprobe.exe) do copy /Y "%%f" "%_DECORD_DIR%\" >nul
set _COPIED=1
goto :decord_done

:decord_missing
echo [WARNING] Self-built decord not found.
echo   Falling back to PyPI decord (CPU-only, high memory).
echo.
echo   To enable GPU decode:
echo     1. Build decord with -DUSE_CUDA=ON (see repo wiki)
echo     2. Copy decord.dll + FFmpeg 5.x DLLs + ffprobe.exe to _decord_build\
echo     3. Re-run this script
goto :decord_done

:decord_done
if %_COPIED%==1 (
    echo   Self-built decord installed - GPU decode ready.
) else (
    echo [WARNING] Self-built decord not found - using PyPI CPU version.
)

echo.
echo Installing GPU Python bindings (thin wrappers, ~few MB) ...
.venv\Scripts\python -m pip install -e ".[dev]"
echo.
echo Removing PySide6-Addons (qfluentwidgets 依赖 PySide6 meta 包会拉入 Addons，
echo 运行只需 Essentials. Addons 含 QtWebEngine 等 ~400MB 冗余) ...
.venv\Scripts\python -m pip uninstall -y PySide6-Addons >nul 2>&1
rem PySide6 打包缺陷：Addons RECORD 误含 Essentials 的 Qt6Core.dll，
rem 卸载后强制重装 Essentials 恢复（否则 QtCore 加载失败）。
.venv\Scripts\python -m pip install --force-reinstall --no-deps PySide6-Essentials -q

echo   GPU video decode (NVDEC) is ready - only requires an NVIDIA GPU with drivers.
echo   GPU OCR (TensorRT) requires CUDA Toolkit + TensorRT installed and on PATH.
echo   Without them, CPU OCR will be used automatically.

echo.
echo ========================================
echo   Setup complete.
echo   Run: .venv\Scripts\python RaceVideoToLog.py
echo ========================================
pause
