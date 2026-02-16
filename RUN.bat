@echo off
:: ============================================================================
:: PhD SEISMIC INTERPRETATION FRAMEWORK - MASTER LAUNCHER
:: Bornu Chad Basin - LLM-Assisted Interpretation
:: Author: Moses Ekene Obasi
:: University of Calabar, Nigeria
:: ============================================================================

title PhD Seismic Framework - Startup

:: Change to script directory
cd /d "%~dp0"

:: ============================================================================
:: FIND PYTHON
:: ============================================================================
set PYTHON_CMD=
set PYTHON_VERSION=

:: Check py launcher first (Windows default)
py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    for /f "tokens=2" %%i in ('py --version 2^>^&1') do set PYTHON_VERSION=%%i
    goto :found_python
)

:: Check python in PATH
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    goto :found_python
)

:: Check common Anaconda locations
if exist "%USERPROFILE%\anaconda3\python.exe" (
    set PYTHON_CMD=%USERPROFILE%\anaconda3\python.exe
    goto :found_python
)
if exist "%USERPROFILE%\Anaconda3\python.exe" (
    set PYTHON_CMD=%USERPROFILE%\Anaconda3\python.exe
    goto :found_python
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set PYTHON_CMD=%USERPROFILE%\miniconda3\python.exe
    goto :found_python
)

:: Python not found
cls
echo.
echo ============================================================================
echo    ERROR: Python not found!
echo ============================================================================
echo.
echo Please install Python from: https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.
echo Or install Anaconda from: https://www.anaconda.com/download
echo.
pause
exit /b 1

:found_python
:: ============================================================================
:: DISPLAY HEADER
:: ============================================================================
cls
echo.
echo ============================================================================
echo    PhD SEISMIC INTERPRETATION FRAMEWORK
echo    Bornu Chad Basin - LLM-Assisted Interpretation
echo ============================================================================
echo    Author: Moses Ekene Obasi
echo    Supervisor: Prof. Dominic Akam Obi
echo    Institution: University of Calabar, Nigeria
echo ============================================================================
echo.
echo [INFO] Python found: %PYTHON_CMD%
if defined PYTHON_VERSION echo [INFO] Version: %PYTHON_VERSION%
echo.

:: ============================================================================
:: CHECK CORE DEPENDENCIES
:: ============================================================================
echo [STEP 1/4] Checking core dependencies...

set MISSING_CORE=0
%PYTHON_CMD% -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo   [X] tkinter - MISSING ^(comes with Python^)
    set MISSING_CORE=1
) else (
    echo   [OK] tkinter
)

%PYTHON_CMD% -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo   [X] numpy - MISSING
    set MISSING_CORE=1
) else (
    echo   [OK] numpy
)

%PYTHON_CMD% -c "import segyio" >nul 2>&1
if errorlevel 1 (
    echo   [X] segyio - MISSING
    set MISSING_CORE=1
) else (
    echo   [OK] segyio
)

%PYTHON_CMD% -c "import lasio" >nul 2>&1
if errorlevel 1 (
    echo   [X] lasio - MISSING
    set MISSING_CORE=1
) else (
    echo   [OK] lasio
)

%PYTHON_CMD% -c "import matplotlib" >nul 2>&1
if errorlevel 1 (
    echo   [X] matplotlib - MISSING
    set MISSING_CORE=1
) else (
    echo   [OK] matplotlib
)

%PYTHON_CMD% -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo   [X] pandas - MISSING
    set MISSING_CORE=1
) else (
    echo   [OK] pandas
)

%PYTHON_CMD% -c "import scipy" >nul 2>&1
if errorlevel 1 (
    echo   [X] scipy - MISSING
    set MISSING_CORE=1
) else (
    echo   [OK] scipy
)

if %MISSING_CORE%==1 (
    echo.
    echo [ACTION] Installing missing core packages...
    %PYTHON_CMD% -m pip install --quiet numpy scipy matplotlib segyio lasio scikit-learn scikit-image tqdm rich pandas requests seaborn pillow h5py
    echo [DONE] Core packages installed.
)
echo.

:: ============================================================================
:: CHECK DEEP LEARNING DEPENDENCIES
:: ============================================================================
echo [STEP 2/4] Checking deep learning dependencies...

set DL_AVAILABLE=1
%PYTHON_CMD% -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo   [!] PyTorch - NOT INSTALLED
    echo       Deep learning features will be disabled.
    echo       To enable: pip install torch torchvision
    set DL_AVAILABLE=0
) else (
    echo   [OK] PyTorch
    %PYTHON_CMD% -c "import torch; print('       CUDA:', 'Available' if torch.cuda.is_available() else 'Not available (CPU mode)')"
)

:: Check for pre-trained model weights
if exist "deep_learning\models\faultseg3d\faultseg3d_pytorch.pth" (
    echo   [OK] FaultSeg3D model weights found
) else (
    echo   [!] FaultSeg3D weights not found
    echo       Run model conversion or download weights
)
echo.

:: ============================================================================
:: CHECK OLLAMA (LLM Features)
:: ============================================================================
echo [STEP 3/4] Checking Ollama LLM service...

set OLLAMA_AVAILABLE=0
ollama list >nul 2>&1
if errorlevel 1 (
    echo   [!] Ollama - NOT RUNNING
    echo       LLM interpretation features will be disabled.
    echo       To enable: Install from https://ollama.ai
    echo       Then run: ollama pull llama3.2
) else (
    set OLLAMA_AVAILABLE=1
    echo   [OK] Ollama is running

    :: Check for recommended models
    ollama list 2>nul | findstr /i "llava" >nul
    if errorlevel 1 (
        echo   [!] llava model not found - Image interpretation disabled
        echo       To enable: ollama pull llava:13b
    ) else (
        echo   [OK] llava model available
    )

    ollama list 2>nul | findstr /i "qwen\|llama\|mixtral" >nul
    if errorlevel 1 (
        echo   [!] No chat model found
        echo       Recommended: ollama pull qwen3:32b or ollama pull llama3.2
    ) else (
        echo   [OK] Chat model available
    )
)
echo.

:: ============================================================================
:: CHECK SAVED STATE
:: ============================================================================
echo [STEP 4/4] Checking saved progress...

if exist "project_config.json" (
    echo   [OK] Project configuration found
    %PYTHON_CMD% -c "import json; c=json.load(open('project_config.json')); print('       Project:', c.get('project_name', 'Unknown'))"
) else (
    echo   [i] No saved project - Will create new configuration
)

if exist "processing_state.json" (
    echo   [OK] Processing state found
    %PYTHON_CMD% -c "import json; s=json.load(open('processing_state.json')); steps=s.get('completed_steps',[]); print('       Completed steps:', len(steps), 'of 9' if steps else 'None yet')"
) else (
    echo   [i] No processing state - Starting fresh
)
echo.

:: ============================================================================
:: LAUNCH GUI
:: ============================================================================
echo ============================================================================
echo    STARTING PhD WORKFLOW GUI
echo ============================================================================
echo.
echo  WORKFLOW STEPS:
echo  ---------------
echo   [1] Exploratory Data Analysis
echo   [2] Dead Trace Detection and Repair
echo   [3] Well Log Integration
echo   [4] 2D Seismic QC
echo   [5] Horizon Interpretation
echo   [6] Seismic Attributes
echo   [7] Acoustic Impedance Inversion
echo   [8] 2D-3D Integration
echo   [9] Deep Learning Interpretation
echo.
echo  Your progress is automatically saved.
echo  Configure your data paths in the Project tab.
echo.
echo ============================================================================
echo.

:: Launch the new unified GUI
%PYTHON_CMD% phd_workflow_gui.py

:: Keep window open if error
if errorlevel 1 (
    echo.
    echo ============================================================================
    echo    ERROR: Application exited with an error
    echo ============================================================================
    echo.
    echo Check the messages above for details.
    echo Common issues:
    echo   - Missing dependencies: Run 'pip install -r requirements.txt'
    echo   - File not found: Ensure phd_workflow_gui.py exists
    echo   - Import errors: Check Python version (3.8+ recommended)
    echo.
    pause
)
