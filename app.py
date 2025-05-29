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
import pyaudio

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
        if stream is None:
            if sys.platform == 'win32':
                # On Windows, use a custom stream that handles UTF-8 properly
                self.stream = codecs.getwriter('utf-8')(sys.stdout.buffer)
            else:
                self.stream = sys.stdout
        else:
            self.stream = stream

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Ensure the message is properly encoded and handle emoji characters
            if isinstance(msg, str):
                # Replace problematic characters with ASCII alternatives
                msg = msg.replace('➡️', '->')
                msg = msg.encode('utf-8', errors='replace').decode('utf-8')
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

# Helper function for safe logging
def safe_log(logger, level, message):
    """Safely log messages by replacing problematic characters"""
    if isinstance(message, str):
        message = message.replace('➡️', '->')
    getattr(logger, level)(message)

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
                safe_log(logger, "info", "➡️ Sent to client: 'ready'")
                
                # Receive the text to synthesize and clean it
                raw_text = await websocket.receive_text()
                # Remove JSON formatting if present
                text = raw_text.replace('{"text": "', '').replace('"}', '').strip()
                safe_log(logger, "info", f"Received text for synthesis: {text}")
                
                # Update config with the received text
                inference_config = custom_config.copy()
                inference_config['gen_text'] = text
                
                # Generate speech using the model
                safe_log(logger, "info", "Starting speech generation...")
                try:
                    wav, sr, spec = tts_model.infer(
                        ref_file=inference_config.get('ref_audio', ''),  # Reference audio path
                        ref_text=inference_config.get('ref_text', ''),  # Reference text from config
                        gen_text=text,
                        show_info=logger.info,  # Use logger for model info
                    )
                    safe_log(logger, "info", "Speech generation completed successfully")
                except Exception as e:
                    error_msg = f"Speech generation failed: {str(e)}"
                    safe_log(logger, "error", error_msg)
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
                safe_log(logger, "info", f"Audio duration: {duration_ms}ms ({duration_ms/1000:.2f} seconds)")
                
                # Send duration to client
                await websocket.send_json({
                    "type": "duration",
                    "duration_ms": duration_ms
                })
                safe_log(logger, "info", f"➡️ Sent to client: duration info - {duration_ms}ms")
                
                # Read audio data
                audio_data, sample_rate = sf.read(str(output_path))
                
                # Ensure audio data is in the correct format (float32)
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # Create an Event for synchronization
                event = asyncio.Event()
                
                # Initialize variables for the callback
                current_frame = 0
                
                def callback(outdata, frames, time, status):
                    nonlocal current_frame
                    if status:
                        safe_log(logger, "info", f"Status: {status}")
                    
                    remaining = len(audio_data) - current_frame
                    if remaining == 0:
                        # No more data to play
                        raise sd.CallbackStop()
                    
                    # Calculate how many frames to write
                    valid_frames = min(remaining, frames)
                    outdata[:valid_frames] = audio_data[current_frame:current_frame + valid_frames].reshape(-1, 1)
                    if valid_frames < frames:
                        outdata[valid_frames:] = 0
                        raise sd.CallbackStop()
                    current_frame += valid_frames

                # Create and start the stream
                try:
                    stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        callback=callback,
                        finished_callback=lambda: event.set(),
                        dtype=np.float32  # Explicitly set dtype to float32
                    )
                    
                    # Log start time
                    start_time = datetime.datetime.now()
                    safe_log(logger, "info", f"Starting audio playback at: {start_time.strftime('%H:%M:%S.%f')[:-3]}")
                    
                    with stream:
                        # Wait for playback to finish
                        await event.wait()
                    
                    # Log end time
                    end_time = datetime.datetime.now()
                    actual_duration = (end_time - start_time).total_seconds() * 1000
                    safe_log(logger, "info", f"Finished audio playback at: {end_time.strftime('%H:%M:%S.%f')[:-3]}")
                    safe_log(logger, "info", f"Actual playback duration: {actual_duration:.2f}ms (expected: {duration_ms}ms)")
                    
                    # Send completion message
                    await websocket.send_json({
                        "type": "complete",
                        "message": "Audio playback completed"
                    })
                    safe_log(logger, "info", "➡️ Sent to client: completion message")
                
                except Exception as e:
                    error_msg = f"Error during audio playback: {str(e)}"
                    safe_log(logger, "error", error_msg)
                    await websocket.send_json({
                        "type": "error",
                        "message": error_msg
                    })
                
                # Add timeout handler after completion
                try:
                    safe_log(logger, "info", "Waiting for next client message (30s timeout)...")
                    await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                except asyncio.TimeoutError:
                    safe_log(logger, "warning", "⚠️ ====================================")
                    safe_log(logger, "warning", "⚠️ Connection timed out after 30 seconds")
                    safe_log(logger, "warning", "⚠️ Closing WebSocket connection")
                    safe_log(logger, "warning", "⚠️ ====================================")
                    await websocket.close(code=1000, reason="Inactivity timeout")
                    break
                
            elif message == "close":
                break
                
    except Exception as e:
        error_msg = f"Error during WebSocket TTS: {str(e)}\n{traceback.format_exc()}"
        safe_log(logger, "error", error_msg)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
            safe_log(logger, "info", f"➡️ Sent to client: error message - {str(e)}")
        except RuntimeError:
            safe_log(logger, "warning", "Could not send error message - connection may be closed")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass 