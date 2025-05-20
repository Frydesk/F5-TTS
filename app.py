from fastapi import FastAPI, UploadFile, File, Form, WebSocket
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
import datetime
import json
import tomli
import soundfile as sf

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

# Load custom configuration
try:
    with open('custom.toml', 'rb') as f:  # tomli requires binary mode
        custom_config = tomli.load(f)
    logger.info("Successfully loaded custom configuration")
except Exception as e:
    logger.error(f"Error loading custom configuration: {str(e)}")
    raise

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/")
async def root():
    return {
        "message": "F5-TTS API is running",
        "endpoints": {
            "POST /tts": "Generate speech from text",
            "GET /health": "Health check endpoint"
        }
    }

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for the start message
            message = await websocket.receive_text()
            
            if message == "start":
                await websocket.send_text("ready")
                
                # Receive the text to synthesize
                text = await websocket.receive_text()
                logger.info(f"Received text for synthesis: {text}")
                
                # Update config with the received text
                inference_config = custom_config.copy()
                inference_config['gen_text'] = text
                
                # Generate speech using the model
                logger.info("Starting speech generation...")
                audio_segment = tts_model.generate_speech(
                    text=text,
                    ref_path=inference_config['ref_audio']
                )
                
                # Save the generated audio temporarily
                output_path = Path("output") / "temp_generated.wav"
                output_path.parent.mkdir(exist_ok=True)
                tts_model.export_wav(audio_segment, str(output_path))
                
                # Get audio duration in milliseconds
                audio_info = sf.info(str(output_path))
                duration_ms = int(audio_info.duration * 1000)
                
                # Send duration to client
                await websocket.send_json({
                    "type": "duration",
                    "duration_ms": duration_ms
                })
                
                # Play the audio (you might want to implement actual audio playback here)
                logger.info(f"Audio duration: {duration_ms}ms")
                
                # Send completion message
                await websocket.send_text("done")
                
            elif message == "close":
                break
                
    except Exception as e:
        error_msg = f"Error during WebSocket TTS: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        await websocket.close() 