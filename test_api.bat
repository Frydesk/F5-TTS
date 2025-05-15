@echo off
setlocal enabledelayedexpansion

echo Activating Anaconda environment 'f5-tts'...
call conda activate f5-tts

echo Testing TTS API...
echo.

REM Set paths from custom.toml
set CONFIG_FILE=custom.toml
set API_URL=http://localhost:8000/tts

REM Check if server is running
echo Checking if server is running...
curl -s "%API_URL%" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Cannot connect to FastAPI server!
    echo Please ensure the server is running using run_api.bat
    echo and check if it's accessible at %API_URL%
    pause
    exit /b 1
)

echo.
echo Running TTS inference with custom.toml...
echo.

REM Run the inference with the custom config
f5-tts_infer-cli -c custom.toml

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: TTS inference failed
    echo.
    echo Please check the logs for more details
    pause
    exit /b 1
) else (
    echo.
    echo Test completed successfully!
    echo Output saved to: output/spanish_test_output.wav
)

echo.
echo Press any key to exit...
pause > nul 