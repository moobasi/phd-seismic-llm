@echo off
:: ============================================================================
:: PhD SEISMIC INTERPRETATION FRAMEWORK - MASTER LAUNCHER
:: ============================================================================
:: Author: Moses Ekene Obasi
:: University of Calabar, Nigeria
:: ============================================================================

title PhD Seismic Framework
cd /d "%~dp0"

:: Find Python
set PYTHON_CMD=
where py >nul 2>&1 && set PYTHON_CMD=py && goto :run
where python >nul 2>&1 && set PYTHON_CMD=python && goto :run
if exist "%USERPROFILE%\anaconda3\python.exe" set PYTHON_CMD="%USERPROFILE%\anaconda3\python.exe" && goto :run
if exist "%USERPROFILE%\Anaconda3\python.exe" set PYTHON_CMD="%USERPROFILE%\Anaconda3\python.exe" && goto :run
if exist "%USERPROFILE%\miniconda3\python.exe" set PYTHON_CMD="%USERPROFILE%\miniconda3\python.exe" && goto :run

echo ERROR: Python not found!
echo Please install Python from https://www.python.org/downloads/
pause
exit /b 1

:run
:: Run the startup script (does checks and launches GUI)
%PYTHON_CMD% startup_check.py

if errorlevel 1 (
    echo.
    echo ERROR: Application exited with an error. Check messages above.
    pause
)
