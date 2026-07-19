@echo off
chcp 65001 >nul
echo ========================================
echo   RaceVideoToLog - Build EXE
echo ========================================
echo.

REM Activate venv
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] .venv not found. Run: python -m venv .venv
    pause
    exit /b 1
)

REM Clean old builds
if exist "build" (
    echo [1/3] Cleaning old build...
    rmdir /s /q "build"
)
if exist "dist" (
    echo [1/3] Cleaning old dist...
    rmdir /s /q "dist"
)

REM Build
echo [2/3] Building with PyInstaller...
python -m PyInstaller RaceVideoToLog.spec --noconfirm

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

REM Show result
echo.
echo [3/3] Build complete!
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
