@echo off
setlocal

echo ============================================
echo  Project Installer
echo ============================================

:: --- Custom wheel definitions ---
set "VLLM_WHEEL=vllm-0.33.0-cp313-cp313-win_amd64.whl"
set "VLLM_URL=https://github.com/aivrar/vllm-windows-build/releases/download/v0.33.0-win-cu130/vllm-0.33.0-cp313-cp313-win_amd64.whl"

set "FLASH_WHEEL=flash_attn-2.8.3+cu130torch2.14-cp313-cp313-win_amd64.whl"
set "FLASH_URL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.10.2/flash_attn-2.8.3+cu130torch2.14-cp313-cp313-win_amd64.whl"

:: --- Check and download wheels ---
if not exist "%VLLM_WHEEL%" (
    echo [▶] Downloading vLLM wheel...
    curl -L -o "%VLLM_WHEEL%" "%VLLM_URL%"
    if errorlevel 1 (
        echo [X] Failed to download vLLM wheel.
        exit /b 1
    )
) else (
    echo [●] vLLM wheel already present.
)

if not exist "%FLASH_WHEEL%" (
    echo [▶] Downloading flash-attn wheel...
    curl -L -o "%FLASH_WHEEL%" "%FLASH_URL%"
    if errorlevel 1 (
        echo [X] Failed to download flash-attn wheel.
        exit /b 1
    )
) else (
    echo [●] flash-attn wheel already present.
)

:: --- Create virtual environment ---
echo [▶] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [X] Failed to create virtual environment.
    exit /b 1
)

:: --- Activate virtual environment ---
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [X] Failed to activate virtual environment.
    exit /b 1
)

:: --- Install dependencies ---
echo [▶] Installing Torch...
pip install torch==2.14.0 torchvision==0.29.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 (
    echo [X] Failed to install Torch.
    exit /b 1
)

echo [▶] Installing vLLM...
pip install "%VLLM_WHEEL%"
if errorlevel 1 (
    echo [X] Failed to install vLLM.
    exit /b 1
)

echo [▶] Installing flash-attn...
pip install "%FLASH_WHEEL%"
if errorlevel 1 (
    echo [X] Failed to install flash-attn.
    exit /b 1
)

echo [▶] Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo [X] Failed to install requirements.
    exit /b 1
)

:: --- Cleanup: remove installed wheels ---
echo [▶] Cleaning up wheel files...
if exist "%VLLM_WHEEL%" del "%VLLM_WHEEL%"
if exist "%FLASH_WHEEL%" del "%FLASH_WHEEL%"
echo [✔] Wheel files removed.

echo ============================================
echo  Installation complete!
echo  PxCaption.bat to run.
echo ============================================

echo.
pause

endlocal
