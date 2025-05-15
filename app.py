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
import yaml
from hydra.utils import get_class
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text
)

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize model and vocoder
logger.info("Initializing model and vocoder...")
model_dir = Path("checkpoints")
logger.info(f"Model directory: {model_dir.absolute()}")

vocoder = load_vocoder(
    vocoder_name="vocos",
    is_local=True,
    local_path="vocos-mel-24khz",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
logger.info("Vocoder loaded successfully")

# Load model configuration
model_cfg_path = Path("src/f5_tts/configs/F5TTS_Base.yaml")
logger.info(f"Loading model config from: {model_cfg_path.absolute()}")
with open(model_cfg_path, 'r') as f:
    model_cfg = yaml.safe_load(f)

# Get the model class from the configuration
model_cls = get_class(f"f5_tts.model.{model_cfg['model']['backbone']}")
model_arc = model_cfg['model']['arch']
logger.info(f"Model class: {model_cls.__name__}")

# Set up model paths
ckpt_path = str(model_dir / "model_1200000.safetensors")
vocab_path = str(model_dir / "vocab.txt")
logger.info(f"Checkpoint path: {ckpt_path}")
logger.info(f"Vocabulary path: {vocab_path}")

model = load_model(
    model_cls=model_cls,
    model_cfg=model_arc,
    ckpt_path=ckpt_path,
    mel_spec_type="vocos",
    vocab_file=vocab_path,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
logger.info("Model loaded successfully")

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
        
        # Preprocess reference audio
        logger.info("Preprocessing reference audio...")
        ref_audio, ref_text = preprocess_ref_audio_text(ref_path, ref_text)
        logger.info(f"Reference audio shape: {ref_audio.shape}")
        logger.info(f"Reference text: {ref_text}")
        
        # Generate speech
        logger.info("Running inference...")
        audio_segment, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            text,
            model,
            vocoder,
            mel_spec_type="vocos",
            target_rms=0.1,
            cross_fade_duration=0.15,
            nfe_step=32,
            cfg_strength=1.0,
            sway_sampling_coef=-1,
            speed=1.0,
            fix_duration=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Generated audio shape: {audio_segment.shape}")
        logger.info(f"Sample rate: {final_sample_rate}")

        # Save the generated audio
        output_path = output_dir / "generated_speech.wav"
        torchaudio.save(str(output_path), torch.from_numpy(audio_segment).unsqueeze(0), final_sample_rate)
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