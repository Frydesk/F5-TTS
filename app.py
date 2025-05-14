from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from inference_api import F5TTSAPI
import torchaudio
import tempfile
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/tts")
async def tts(text: str = Form(...), reference_audio: UploadFile = File(...)):
    logger.info(f"Received TTS request for text: {text}")
    logger.info(f"Reference audio filename: {reference_audio.filename}")
    
    # Save uploaded reference audio to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_ref:
        ref_path = tmp_ref.name
        content = await reference_audio.read()
        tmp_ref.write(content)
        logger.info(f"Saved reference audio to: {ref_path}")

    try:
        # Generate speech
        api = F5TTSAPI.get_instance()
        logger.info("Generating speech...")
        audio, sr = api.generate_speech(text, ref_path)
        logger.info(f"Speech generated successfully. Audio shape: {audio.shape}")

        # Save generated audio to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_out:
            out_path = tmp_out.name
            torchaudio.save(out_path, audio.cpu(), sr)
            logger.info(f"Saved generated audio to: {out_path}")

        # Return the generated audio file
        return FileResponse(out_path, media_type='audio/wav', filename='output.wav')
    except Exception as e:
        logger.error(f"Error during TTS generation: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Clean up temp reference audio
        if os.path.exists(ref_path):
            os.remove(ref_path)
            logger.info(f"Cleaned up temporary reference audio: {ref_path}")

@app.get("/")
async def root():
    return {"message": "F5-TTS API is running. Use POST /tts endpoint for text-to-speech conversion."} 