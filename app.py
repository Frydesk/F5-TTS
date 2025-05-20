from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import logging
import traceback
from pathlib import Path
import torch
from f5_tts.api import F5TTS
import tomli
import soundfile as sf
import datetime

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_device": device,
        "model_loaded": tts_model is not None
    }

@app.websocket("/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for the message
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
                wav, sr, spec = tts_model.infer(
                    ref_file=inference_config['ref_audio'],
                    ref_text="",  # Empty string for auto-transcription
                    gen_text=text,
                    target_rms=0.1,
                    cross_fade_duration=0.15,
                    nfe_step=32,
                    cfg_strength=2.0,
                    sway_sampling_coef=-1.0,
                    speed=1.0
                )
                
                # Save the generated audio temporarily
                output_path = Path("output") / "temp_generated.wav"
                output_path.parent.mkdir(exist_ok=True)
                tts_model.export_wav(wav, str(output_path))
                
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
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except RuntimeError:
            # Connection might already be closed
            logger.warning("Could not send error message - connection may be closed")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            # Connection might already be closed
            pass 