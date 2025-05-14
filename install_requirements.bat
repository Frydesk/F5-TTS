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
pip install pyyaml safetensors huggingface_hub

echo Installing F5-TTS in editable mode...
pip install -e .

echo Installing f5-tts for inference
pip install f5-tts

echo Creating models directory structure...
if not exist "models\vocos" mkdir "models\vocos"

echo Downloading Vocos model...
python -c "from huggingface_hub import snapshot_download; snapshot_download('charactr/vocos-mel-24khz', local_dir='models/vocos')"

echo Installation complete! You can now activate the environment using: conda activate f5-tts
pause 