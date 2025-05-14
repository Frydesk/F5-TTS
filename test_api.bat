@echo off
echo Testing TTS API...
echo.

REM Set the API endpoint
set API_URL=http://localhost:8000/tts

REM Set the text to convert (without quotes)
set TEXT=This is a test of the TTS API

REM Set the reference audio file path
set REF_AUDIO=reference/reference.wav

echo Sending request to %API_URL%
echo Text: %TEXT%
echo Reference Audio: %REF_AUDIO%
echo.

REM Create output directory if it doesn't exist
if not exist output mkdir output

REM Send the request using curl with verbose output
curl -v -X POST ^
  -F "text=%TEXT%" ^
  -F "reference_audio=@%REF_AUDIO%" ^
  -o output/test_output.wav ^
  %API_URL%

echo.
if %ERRORLEVEL% EQU 0 (
    echo Test completed successfully!
    echo Output saved to: output/test_output.wav
    
    REM Check if the output file exists and has content
    if exist output\test_output.wav (
        for %%A in (output\test_output.wav) do set size=%%~zA
        if !size! gtr 0 (
            echo File size: !size! bytes
        ) else (
            echo Warning: Output file is empty!
        )
    ) else (
        echo Error: Output file was not created!
    )
) else (
    echo Test failed with error code: %ERRORLEVEL%
)

pause 