@echo off
setlocal enabledelayedexpansion

echo Activating Anaconda environment 'f5-tts'...
call conda activate f5-tts
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate f5-tts environment. Please make sure it exists.
    pause
    exit /b 1
)

echo Checking Flash Attention status...
python -c "import flash_attn; print('Flash Attention detected - Version:', flash_attn.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Flash Attention is not installed or not working properly
    echo The system will still work but may be slower
    echo.
)

echo Verifying required packages...
python -c "import uvicorn; import fastapi" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Missing required packages. Please run setup.bat first.
    pause
    exit /b 1
)

echo Starting FastAPI server...
echo Press Ctrl+C to stop the server gracefully
echo.

REM Start the server without background flag to see immediate errors
python -m uvicorn app:app --reload --host 0.0.0.0 --port 9000 --log-level info --log-config logging.conf

REM Check if the server exited with an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server failed to start or stopped with an error.
    echo Please check the error messages above.
    echo Common issues:
    echo - Port 9000 already in use
    echo - Missing or incorrect logging.conf file
    echo - Issues with app.py file
    echo.
) else (
    echo.
    echo Server stopped gracefully.
)

pause 