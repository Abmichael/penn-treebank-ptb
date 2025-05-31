@echo off
REM Penn Treebank Project Cleanup Script for Windows
REM This batch file provides easy access to the Python cleanup script

echo.
echo ================================================
echo Penn Treebank Project Cleanup for Windows
echo ================================================
echo.

REM Check if we're in the right directory
if not exist "src" (
    echo Error: Please run this script from the project root directory
    echo (Directory should contain 'src' and 'config' folders)
    pause
    exit /b 1
)

if not exist "config" (
    echo Error: Please run this script from the project root directory
    echo (Directory should contain 'src' and 'config' folders)
    pause
    exit /b 1
)

echo Choose cleanup option:
echo.
echo 1. Preview cleanup (dry-run)
echo 2. Standard cleanup (keeps models)
echo 3. Aggressive cleanup (removes models too)
echo 4. Interactive cleanup
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Running preview mode...
    python scripts/cleanup_project.py --dry-run
) else if "%choice%"=="2" (
    echo.
    echo Running standard cleanup...
    python scripts/cleanup_project.py
) else if "%choice%"=="3" (
    echo.
    echo Running aggressive cleanup...
    python scripts/cleanup_project.py --aggressive
) else if "%choice%"=="4" (
    echo.
    echo Running interactive cleanup...
    python scripts/cleanup_project.py --interactive
) else if "%choice%"=="5" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice. Please run the script again.
)

echo.
pause
