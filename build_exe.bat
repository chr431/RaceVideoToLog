@echo off
cd /d "%~dp0"
chcp 65001 >nul
echo ========================================
echo   RaceVideoToLog - Build EXE
echo ========================================
echo.

REM [1/4] Check / create venv
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] .venv not found, running setup_venv.bat ...
    call "%~dp0setup_venv.bat"
    if errorlevel 1 (
        echo [ERROR] venv setup failed.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Using existing .venv.
)
set PY=.venv\Scripts\python

REM [2/4] Verify key deps + PyInstaller
echo [2/4] Checking dependencies ...
%PY% -c "import onnxruntime, numpy, PySide6, decord, qfluentwidgets, shapely, pyclipper, cuda"
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

REM [3/4] Build
echo.
echo [3/4] Building EXE ...
if exist "build" rmdir /s /q "build"
if exist "dist"  rmdir /s /q "dist"

%PY% -m PyInstaller RaceVideoToLog.spec --noconfirm
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.  Make sure the venv is active.
    pause
    exit /b 1
)

REM [4/4] Show result
echo.
echo [4/4] Build complete.
for /d %%d in (dist\*) do (
    echo   Output: %%d
    dir "%%d\RaceVideoToLog.exe" 2>nul
)
echo.
echo ========================================
echo   Done
echo ========================================
pause
