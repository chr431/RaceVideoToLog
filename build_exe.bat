@echo off
chcp 65001 >nul
echo ========================================
echo   RaceVideoToLog - Build EXE
echo ========================================
echo.

REM Check / create venv
if not exist ".venv\Scripts\python.exe" (
    echo [0/4] Creating .venv...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create .venv
        pause
        exit /b 1
    )
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Ensure PyInstaller is installed
echo [1/4] Checking PyInstaller...
.venv\Scripts\python -c "import PyInstaller" 2>nul
if %ERRORLEVEL% neq 0 (
    echo   Installing PyInstaller...
    .venv\Scripts\python -m pip install pyinstaller -q
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install PyInstaller
        pause
        exit /b 1
    )
)

REM Clean old builds
if exist "build" (
    echo [2/4] Cleaning old build...
    rmdir /s /q "build"
)
if exist "dist" (
    echo [2/4] Cleaning old dist...
    rmdir /s /q "dist"
)

REM Build
echo [3/4] Building with PyInstaller...
python -m PyInstaller RaceVideoToLog.spec --noconfirm

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

REM Show result
echo.
echo [4/4] Build complete!
echo.
for /d %%d in (dist\*) do (
    echo   Output: %%d
    dir /s "%%d\RaceVideoToLog.exe" 2>nul
)
echo.
echo ========================================
echo   Done - press any key to exit
echo ========================================
pause
