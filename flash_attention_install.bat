@echo off

echo Activating f5-tts environment...
call conda activate f5-tts

echo This installation process can take several hours to complete.
echo Installing dependencies...
pip install ninja
echo Installing flash-attention...
pip install flash-attn --no-build-isolation
echo Installing Triton from source...
git clone https://github.com/triton-lang/triton.git
cd triton
pip install cmake wheel
pip install -e .
echo Flash-attention and Triton installation completed.
pause 