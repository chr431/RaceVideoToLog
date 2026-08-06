@echo off
cd /d "%~dp0"
chcp 65001 >nul
setlocal
rem --ci 模式（CI/自动化）：跳过所有 pause（GitHub Actions / 脚本调用）
set _NOPAUSE=0
if /i "%~1"=="--ci" set _NOPAUSE=1

echo ========================================
echo   RaceVideoToLog - Setup venv
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    echo   Install Python 3.11+ from https://python.org
    if not "%_NOPAUSE%"=="1" pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    echo Existing .venv found. To rebuild: rmdir /s /q .venv ^&^& setup_venv.bat
    echo.
    echo Run .venv\Scripts\activate to enter the venv.
    if not "%_NOPAUSE%"=="1" pause
    exit /b 0
)

echo Creating .venv ...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create .venv
    if not "%_NOPAUSE%"=="1" pause
    exit /b 1
)

echo Upgrading pip ...
.venv\Scripts\python -m pip install --upgrade pip -q

echo.
echo Installing project dependencies ...
.venv\Scripts\python -m pip install -e .
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    if not "%_NOPAUSE%"=="1" pause
    exit /b 1
)

echo.
echo ========================================
echo   decord (self-built fork, GPU + memory fix)
echo ========================================
echo.
echo This project requires the self-built decord fork (chr431/decord) with
echo NVDEC support, CPU memory fixes, and next_roi/get_codec APIs.
echo PyPI decord is NOT supported (CPU-only, ~10 GB RAM, no next_roi).
echo.

set _DECORD_DIR=.venv\Lib\site-packages\decord
set _COPIED=0

rem 用 goto 分支替代 if 括号块内的 for 循环（cmd 的括号解析 bug
rem 会使嵌套 for 报 ". was unexpected at this time."）。
rem 无 PyPI 回退：decord 必须来自自建 fork（_decord_build\ 或本地 decord 仓库构建）。
if exist "_decord_build\decord.dll" goto :use_decord_build
if exist "..\decord\build\Release\decord.dll" goto :use_decord_sibling
echo [ERROR] Self-built decord not found.
echo   Expected _decord_build\decord.dll or ..\decord\build\Release\decord.dll
echo   To obtain it:
echo     1. 运行 decord 仓库（chr431/decord）的 Release workflow，或本地 GPU 构建
echo     2. 将产物解压/复制为 _decord_build\（decord.dll + FFmpeg 8 DLLs
echo        + ffprobe.exe + python\decord\），setup_venv.bat 直接拷贝
echo   PyPI decord 不再支持（无 next_roi / get_codec、CPU 解码内存溢出）。
if not "%_NOPAUSE%"=="1" pause
exit /b 1

:use_decord_build
echo Found _decord_build\ - installing self-built decord ...
if not exist "%_DECORD_DIR%" mkdir "%_DECORD_DIR%"
for %%f in (_decord_build\*.dll _decord_build\ffprobe.exe) do (
    if exist "%%f" copy /Y "%%f" "%_DECORD_DIR%\" >nul
)
if exist "_decord_build\python\decord\video_reader.py" goto :py_layer_build
if exist "..\decord\python\decord\video_reader.py" goto :py_layer_sibling
echo [ERROR] decord Python 层未找到（_decord_build\python\decord\ 缺失）
echo   next_roi / get_codec 不可用。请重新获取完整 decord 发布产物。
if not "%_NOPAUSE%"=="1" pause
exit /b 1

:use_decord_sibling
echo Found ..\decord\build\Release\ - installing from sibling repo ...
if not exist "%_DECORD_DIR%" mkdir "%_DECORD_DIR%"
for %%f in (..\decord\build\Release\*.dll ..\decord\build\Release\ffprobe.exe) do (
    if exist "%%f" copy /Y "%%f" "%_DECORD_DIR%\" >nul
)
goto :py_layer_sibling

:py_layer_build
xcopy /E /Y /I "_decord_build\python\decord\*" "%_DECORD_DIR%\" >nul
echo   Fork decord Python layer installed.
set _COPIED=1
goto :decord_done

:py_layer_sibling
xcopy /E /Y /I "..\decord\python\decord\*" "%_DECORD_DIR%\" >nul
echo   Fork decord Python layer installed (from sibling repo).
set _COPIED=1
goto :decord_done

:decord_done
if %_COPIED%==1 (
    echo   Self-built decord installed - GPU decode ready.
) else (
    echo [ERROR] decord 安装失败
    if not "%_NOPAUSE%"=="1" pause
    exit /b 1
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
if not "%_NOPAUSE%"=="1" pause
