from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import torchaudio
import tempfile
import os
import logging
import traceback
import sys
from pathlib import Path
import torch
from f5_tts.api import F5TTS

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize TTS model
logger.info("Initializing TTS model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = F5TTS(model='F5TTS_v1_Base', device=device)
logger.info("TTS model loaded successfully")

@app.post("/tts")
async def tts(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    transcription: UploadFile = File(None)
):
    logger.info("="*50)
    logger.info("Received new TTS request")
    logger.info(f"Text to synthesize: {text}")
    logger.info(f"Reference audio filename: {reference_audio.filename}")
    logger.info(f"Reference audio content type: {reference_audio.content_type}")
    if transcription:
        logger.info(f"Transcription filename: {transcription.filename}")
        logger.info(f"Transcription content type: {transcription.content_type}")
    
    # Save uploaded reference audio to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_ref:
        ref_path = tmp_ref.name
        content = await reference_audio.read()
        tmp_ref.write(content)
        logger.info(f"Saved reference audio to: {ref_path}")
        logger.info(f"Reference audio size: {len(content)} bytes")

    # Save transcription if provided
    ref_text = ""
    if transcription:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_trans:
            trans_path = tmp_trans.name
            content = await transcription.read()
            tmp_trans.write(content)
            with open(trans_path, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()
            logger.info(f"Saved transcription to: {trans_path}")
            logger.info(f"Transcription content: {ref_text}")
            os.remove(trans_path)

    try:
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")
        
        # Generate speech
        logger.info("Starting speech generation...")
        
        # Generate speech using the preloaded model
        logger.info("Running inference...")
        audio_segment = tts_model.generate_speech(text, ref_path)
        logger.info(f"Generated audio shape: {audio_segment.shape}")
        
        # Save the generated audio
        output_path = output_dir / "generated_speech.wav"
        tts_model.export_wav(audio_segment, str(output_path))
        logger.info(f"Saved generated audio to: {output_path}")
        
        # Verify the output file
        if not output_path.exists() or output_path.stat().st_size < 1000:  # Less than 1KB is suspicious
            logger.error(f"Generated audio file is too small or doesn't exist: {output_path}")
            raise ValueError("Generated audio file is too small or doesn't exist")

        # Return the generated audio file
        logger.info("Request completed successfully")
        return FileResponse(str(output_path), media_type='audio/wav', filename='output.wav')
    except Exception as e:
        error_msg = f"Error during TTS generation: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
    finally:
        # Clean up temp files
        if os.path.exists(ref_path):
            os.remove(ref_path)
            logger.info(f"Cleaned up temporary reference audio: {ref_path}")
        logger.info("="*50)

@app.get("/")
async def root():
    return {"message": "F5-TTS API is running. Use POST /tts endpoint for text-to-speech conversion."} 