@echo off
cd /d "%~dp0"
chcp 65001 >nul
echo ========================================
echo   RaceVideoToLog - Build EXE
echo ========================================
echo.

REM [1/5] Check / create venv
if not exist ".venv\Scripts\python.exe" (
    echo [1/5] .venv not found, running setup_venv.bat ...
    call "%~dp0setup_venv.bat"
    if errorlevel 1 (
        echo [ERROR] venv setup failed.
        pause
        exit /b 1
    )
) else (
    echo [1/5] Using existing .venv.
)
set PY=.venv\Scripts\python

REM [2/5] Verify key deps + PyInstaller
echo [2/5] Checking dependencies ...
%PY% -c "import rapidocr, onnxruntime, cv2, numpy, PySide6, matplotlib, decord, qfluentwidgets, shapely, pyclipper, tensorrt, cuda"
if errorlevel 1 (
    echo   Some deps missing, reinstalling ...
    %PY% -m pip install -e .
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)


%PY% -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo   Installing PyInstaller ...
    %PY% -m pip install pyinstaller
    if errorlevel 1 (
        echo [ERROR] Failed to install PyInstaller.
        pause
        exit /b 1
    )
)

REM [3/5] Verify spec exists
if not exist "RaceVideoToLog.spec" (
    echo [ERROR] RaceVideoToLog.spec not found.
    echo   Run this script from the repository root.
    pause
    exit /b 1
)

REM [4/5] Clean + Build
echo [4/5] Cleaning old builds ...
if exist "build" rmdir /s /q "build"
if exist "dist"  rmdir /s /q "dist"

echo [4/5] Building with PyInstaller ...
%PY% -m PyInstaller RaceVideoToLog.spec --noconfirm
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

REM [5/5] Show result
echo.
echo [5/5] Build complete.
for /d %%d in (dist\*) do (
    echo   Output: %%d
    dir /s "%%d\RaceVideoToLog.exe" 2>nul
)
echo.
echo ========================================
echo   Done
echo ========================================
pause
