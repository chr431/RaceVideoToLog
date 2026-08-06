@echo off
cd /d "%~dp0"
chcp 65001 >nul
setlocal
rem --ci 模式（CI/自动化）：跳过所有 pause（GitHub Actions / 脚本调用）
set _NOPAUSE=0
if /i "%~1"=="--ci" set _NOPAUSE=1

echo ========================================
echo   RaceVideoToLog - Build EXE
echo ========================================
echo.

REM [1/4] Check / create venv
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] .venv not found, running setup_venv.bat ...
    call "%~dp0setup_venv.bat" %1
    if errorlevel 1 (
        echo [ERROR] venv setup failed.
        if not "%_NOPAUSE%"=="1" pause
        exit /b 1
    )
) else (
    echo [1/4] Using existing .venv.
)
set PY=.venv\Scripts\python

REM [2/4] Verify key deps + PyInstaller
echo [2/4] Checking dependencies ...
%PY% -c "import onnxruntime, numpy, PySide6, decord, qfluentwidgets, cuda"
if errorlevel 1 (
    echo   Some deps missing, reinstalling ...
    %PY% -m pip install -e .
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        if not "%_NOPAUSE%"=="1" pause
        exit /b 1
    )
)

%PY% -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo   Installing PyInstaller ...
    %PY% -m pip install pyinstaller
    if errorlevel 1 (
        echo [ERROR] Failed to install PyInstaller.
        if not "%_NOPAUSE%"=="1" pause
        exit /b 1
    )
)

REM [3/4] Version consistency check (single source: config.__version__)
echo.
echo [3/4] Checking version references ...
%PY% tools/version.py
if errorlevel 1 (
    echo.
    echo [ERROR] Version references inconsistent. Run: python tools/version.py bump X.Y.Z
    if not "%_NOPAUSE%"=="1" pause
    exit /b 1
)

REM [4/4] Build
echo.
echo [4/4] Building EXE ...
if exist "build" rmdir /s /q "build"
if exist "dist"  rmdir /s /q "dist"

%PY% -m PyInstaller RaceVideoToLog.spec --noconfirm
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.  Make sure the venv is active.
    if not "%_NOPAUSE%"=="1" pause
    exit /b 1
)

REM [5/4] Show result
echo.
echo [5/4] Build complete.
for /d %%d in (dist\*) do (
    echo   Output: %%d
    dir "%%d\RaceVideoToLog.exe" 2>nul
)
echo.
echo ========================================
echo   Done
echo ========================================
if not "%_NOPAUSE%"=="1" pause
