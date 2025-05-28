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
import asyncio
import sounddevice as sd
import numpy as np
import os
import sys
import codecs

# Force UTF-8 encoding for stdout and stderr
if sys.platform == 'win32':
    # Python UTF-8 Mode
    if hasattr(sys, 'set_utf8_mode'):
        sys.set_utf8_mode(True)
    
    # Force console to use UTF-8
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCP(65001)
    kernel32.SetConsoleOutputCP(65001)

# Create a custom StreamHandler that forces UTF-8
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.stream = codecs.getwriter('utf-8')(sys.stdout.buffer) if stream is None else stream

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Set up logging with UTF-8 encoding
os.makedirs('logs', exist_ok=True)  # Ensure logs directory exists
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        UTF8StreamHandler(),
        logging.FileHandler('logs/f5_tts_api.log', encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger("fastapi")

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
logger.info("Initializing TTS model (F5TTS_Base)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

try:
    tts_model = F5TTS(
        model='F5TTS_Base',
        device=device,
        use_ema=True,  # Use EMA for better quality
        ode_method='euler',  # Default ODE solver
        hf_cache_dir=os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')  # Explicit cache dir
    )
    logger.info("F5TTS_Base model loaded successfully")
    logger.info("Model configuration loaded and verified")
except Exception as e:
    logger.error(f"Failed to load F5TTS_Base model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

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
                logger.info("➡️ Sent to client: 'ready'")
                
                # Receive the text to synthesize and clean it
                raw_text = await websocket.receive_text()
                # Remove JSON formatting if present
                text = raw_text.replace('{"text": "', '').replace('"}', '').strip()
                logger.info(f"Received text for synthesis: {text}")
                
                # Update config with the received text
                inference_config = custom_config.copy()
                inference_config['gen_text'] = text
                
                # Generate speech using the model
                logger.info("Starting speech generation...")
                try:
                    wav, sr, spec = tts_model.infer(
                        ref_file=inference_config.get('ref_audio', ''),  # Reference audio path
                        ref_text="",  # Empty string for auto-transcription
                        gen_text=text,
                        target_rms=0.1,  # Default RMS value for good volume
                        cross_fade_duration=0.15,  # Smooth transitions
                        nfe_step=32,  # Number of flow steps
                        cfg_strength=2.0,  # Classifier-free guidance strength
                        sway_sampling_coef=-1.0,  # Sway sampling for better quality
                        speed=1.0,  # Normal speed
                        show_info=logger.info,  # Use logger for model info
                        progress=None  # Disable progress bar in API context
                    )
                    logger.info("Speech generation completed successfully")
                except Exception as e:
                    error_msg = f"Speech generation failed: {str(e)}"
                    logger.error(error_msg)
                    await websocket.send_json({
                        "type": "error",
                        "message": error_msg
                    })
                    continue
                
                # Save the generated audio temporarily
                output_path = Path("output") / "temp_generated.wav"
                output_path.parent.mkdir(exist_ok=True)
                tts_model.export_wav(wav, str(output_path))
                
                # Get audio duration in milliseconds
                audio_info = sf.info(str(output_path))
                duration_ms = int(audio_info.duration * 1000)
                logger.info(f"Audio duration: {duration_ms}ms ({duration_ms/1000:.2f} seconds)")
                
                # Send duration to client
                await websocket.send_json({
                    "type": "duration",
                    "duration_ms": duration_ms
                })
                logger.info(f"➡️ Sent to client: duration info - {duration_ms}ms")
                
                # Log start time
                start_time = datetime.datetime.now()
                logger.info(f"Starting audio playback at: {start_time.strftime('%H:%M:%S.%f')[:-3]}")
                
                # Play the audio in a non-blocking way
                audio_data, sample_rate = sf.read(str(output_path))
                
                # Start playback in a non-blocking way
                sd.play(audio_data, sample_rate)
                
                # Wait for the exact duration of the audio
                await asyncio.sleep(duration_ms / 1000.0)  # Convert ms to seconds
                
                # Stop playback
                sd.stop()
                
                # Log end time
                end_time = datetime.datetime.now()
                actual_duration = (end_time - start_time).total_seconds() * 1000
                logger.info(f"Finished audio playback at: {end_time.strftime('%H:%M:%S.%f')[:-3]}")
                logger.info(f"Actual playback duration: {actual_duration:.2f}ms (expected: {duration_ms}ms)")
                
                # Send completion message
                await websocket.send_json({
                    "type": "complete",
                    "message": "Audio playback completed"
                })
                logger.info("➡️ Sent to client: completion message")
                
                # Add timeout handler after completion
                try:
                    logger.info("Waiting for next client message (30s timeout)...")
                    await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning("⚠️ ====================================")
                    logger.warning("⚠️ Connection timed out after 30 seconds")
                    logger.warning("⚠️ Closing WebSocket connection")
                    logger.warning("⚠️ ====================================")
                    await websocket.close(code=1000, reason="Inactivity timeout")
                    break
                
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
            logger.info(f"➡️ Sent to client: error message - {str(e)}")
        except RuntimeError:
            logger.warning("Could not send error message - connection may be closed")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass 