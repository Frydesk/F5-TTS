@echo off
echo Checking if f5-tts environment exists...

call conda env list | findstr /C:"f5-tts" > nul
if errorlevel 1 (
    echo Creating conda environment for F5-TTS...
    call conda create -n f5-tts python=3.10 -y
) else (
    echo f5-tts environment already exists
)

echo Activating f5-tts environment...
call conda activate f5-tts

echo Installing PyTorch with CUDA support...
pip install --pre torch==2.8.0.dev20250324 torchvision==0.22.0.dev20250325+cu128 torchaudio==2.6.0.dev20250325+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128

echo Installing required packages...
pip install pyyaml safetensors huggingface_hub fastapi uvicorn
pip install ninja

echo Installing ffmpeg...
call conda install -c conda-forge ffmpeg -y

echo Installing F5-TTS in editable mode...
pip install -e .

echo Creating models directory structure...
if not exist "models\vocos" mkdir "models\vocos"

echo Downloading Vocos model...
python -c "from huggingface_hub import snapshot_download; snapshot_download('charactr/vocos-mel-24khz', local_dir='models/vocos')"

echo Building Flash Attention for Windows...
if exist "flash-attention" rmdir /s /q "flash-attention"
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
git checkout -b v2.7.0.post2 v2.7.0.post2

echo Downloading WindowsWhlBuilder_cuda.bat...
powershell -Command "Invoke-WebRequest -Uri 'https://huggingface.co/lldacing/flash-attention-windows-wheel/raw/main/WindowsWhlBuilder_cuda.bat' -OutFile 'WindowsWhlBuilder_cuda.bat'"

if not exist "WindowsWhlBuilder_cuda.bat" (
    echo Failed to download WindowsWhlBuilder_cuda.bat. Please check your internet connection and try again.
    exit /b 1
)

echo Building Flash Attention (this may take around 30 minutes)...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo Failed to initialize Visual Studio environment. Please make sure Visual Studio 2022 is installed.
    exit /b 1
)

call WindowsWhlBuilder_cuda.bat
if errorlevel 1 (
    echo Flash Attention build failed. Please check the error messages above.
    exit /b 1
)

echo Installing built Flash Attention wheel...
if not exist "dist\*.whl" (
    echo No wheel file found in dist directory. Build may have failed.
    exit /b 1
)

for %%f in (dist\*.whl) do pip install %%f

cd ..

echo Installation complete! You can now activate the environment using: conda activate f5-tts
pause 