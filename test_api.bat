@echo off
setlocal enabledelayedexpansion

echo Activating Anaconda environment 'f5-tts'...
call conda activate f5-tts

echo Testing TTS API...
echo.

REM Check for tomli_w package
python -c "import tomli_w" 2>nul
if errorlevel 1 (
    echo Installing tomli_w package...
    pip install tomli_w
    if errorlevel 1 (
        echo Error: Failed to install tomli_w package
        pause
        exit /b 1
    )
)

REM Set paths and URLs
set API_URL=http://localhost:9000/tts
set REFERENCE_AUDIO=reference/reference.wav
set OUTPUT_DIR=output
set CONFIGS_DIR=configs
set CONFIG_FILE=%CONFIGS_DIR%\test_config.toml

REM Create necessary directories
if not exist "%CONFIGS_DIR%" (
    echo Creating configs directory...
    mkdir "%CONFIGS_DIR%"
)
if not exist "%OUTPUT_DIR%" (
    echo Creating output directory...
    mkdir "%OUTPUT_DIR%"
)
if not exist "reference" (
    echo Creating reference directory...
    mkdir "reference"
)

REM Check if server is running
echo Checking if server is running...
curl -s "%API_URL%" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Cannot connect to FastAPI server!
    echo Please ensure the server is running using tts-api.bat
    echo and check if it's accessible at %API_URL%
    pause
    exit /b 1
)

:test_loop
echo.
echo ===========================================
echo TTS Inference Test
echo ===========================================
echo.

REM Ask user for configuration method
set /p CONFIG_METHOD="Choose configuration method (1 for custom.toml, 2 for manual input, default: 2): "
if "!CONFIG_METHOD!"=="1" (
    set /p CUSTOM_TOML="Enter path to custom.toml file: "
    if not exist "!CUSTOM_TOML!" (
        echo Error: File not found: !CUSTOM_TOML!
        goto test_loop
    )
    set CONFIG_FILE=!CUSTOM_TOML!
    echo Using configuration from: !CUSTOM_TOML!
) else (
    echo.
    echo Manual configuration mode
    echo ------------------------
    
    REM Get user input for inference parameters
    set /p MODEL="Enter model name (default: F5TTS_v1_Base): "
    if "!MODEL!"=="" set MODEL=F5TTS_v1_Base

    set /p REF_AUDIO="Enter reference audio path (default: %REFERENCE_AUDIO%): "
    if "!REF_AUDIO!"=="" set REF_AUDIO=%REFERENCE_AUDIO%

    set /p REF_TEXT="Enter reference text (optional): "

    set /p GEN_TEXT="Enter text to generate: "
    if "!GEN_TEXT!"=="" (
        echo Error: Text to generate is required
        goto test_loop
    )

    set /p REMOVE_SILENCE="Remove silence? (y/n, default: n): "
    if /i "!REMOVE_SILENCE!"=="y" (
        set REMOVE_SILENCE=True
    ) else (
        set REMOVE_SILENCE=False
    )

    set /p OUTPUT_FILE="Enter output filename (default: generated_speech.wav): "
    if "!OUTPUT_FILE!"=="" set OUTPUT_FILE=generated_speech.wav

    REM Create the TOML file using Python
    echo Creating configuration file...
    echo Configuration will be saved to: %CONFIG_FILE%
    
    REM Create a Python script to write the TOML file
    (
        echo # -*- coding: utf-8 -*-
        echo import tomli_w
        echo import os
        echo import pathlib
        echo.
        echo # Create configs directory if it doesn't exist
        echo os.makedirs('configs', exist_ok=True^)
        echo.
        echo # Create configuration dictionary
        echo config = {
        echo     'model': '!MODEL!',
        echo     'ref_audio': '!REF_AUDIO!',
    ) > create_toml.py

    if not "!REF_TEXT!"=="" (
        echo     'ref_text': '!REF_TEXT!', >> create_toml.py
    )

    (
        echo     'gen_text': '!GEN_TEXT!',
        echo     'remove_silence': !REMOVE_SILENCE!,
        echo     'output_dir': '%OUTPUT_DIR%',
        echo     'output_file': '!OUTPUT_FILE!'
        echo }
        echo.
        echo # Write TOML file
        echo config_path = pathlib.Path('configs/test_config.toml'^)
        echo with open(config_path, 'wb'^) as f:
        echo     tomli_w.dump(config, f^)
    ) >> create_toml.py

    REM Run the Python script with UTF-8 encoding
    python -X utf8 create_toml.py
    if errorlevel 1 (
        echo Error: Failed to create configuration file
        echo Please ensure you have write permissions in the current directory
        pause
        goto test_loop
    )
    del create_toml.py

    if not exist "%CONFIG_FILE%" (
        echo Error: Failed to create configuration file
        pause
        goto test_loop
    )

    echo.
    echo Running TTS inference with configuration:
    type "%CONFIG_FILE%"
)

echo.
echo Running inference...
f5-tts_infer-cli -c "%CONFIG_FILE%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: TTS inference failed
    echo.
    echo Please check the logs for more details
) else (
    echo.
    echo Test completed successfully!
    if "!CONFIG_METHOD!"=="1" (
        echo Output saved according to custom.toml configuration
    ) else (
        echo Output saved to: %OUTPUT_DIR%\!OUTPUT_FILE!
    )
)

echo.
set /p CONTINUE="Run another test? (y/n, default: y): "
if /i not "!CONTINUE!"=="n" goto test_loop

echo.
echo Testing complete. Press any key to exit...
pause > nul 