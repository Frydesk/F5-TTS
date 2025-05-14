@echo off
call conda activate f5-tts

f5-tts_infer-cli --model F5TTS_Base ^
--ref_audio "reference/reference.wav" ^
--ref_text "reference/transcription.txt" ^
--gen_text "reference/texto.txt" ^
--vocoder_name "vocos" ^
--load_vocoder_from_local

if errorlevel 1 (
    echo Error: Inference failed
    exit /b 1
    pause
) 
pause