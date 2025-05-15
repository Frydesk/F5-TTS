@echo off
setlocal enabledelayedexpansion

echo Activating Anaconda environment 'f5-tts'...
call conda activate f5-tts

echo Preloading TTS models into RAM...
python -c "from f5_tts.api import F5TTS; model = F5TTS(model='F5TTS_v1_Base', device='cuda' if __import__('torch').cuda.is_available() else 'cpu'); print('Models loaded successfully!')"

echo Starting FastAPI server...
echo Press Ctrl+C to stop the server gracefully
echo.

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Start the server with proper error handling and logging
python -m uvicorn app:app --reload --host 0.0.0.0 --port 9000 --log-level info --log-config logging.conf 2> logs\error.log

REM Check if the server exited with an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server stopped with an error. Check logs\error.log for details.
    echo.
    type logs\error.log
) else (
    echo.
    echo Server stopped gracefully.
)

pause 