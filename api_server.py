from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import tempfile
import os
from pathlib import Path
from inference_api import F5TTSAPI
import torchaudio
import uvicorn

app = FastAPI(title="F5-TTS API", description="Text-to-Speech API using F5-TTS")

# Initialize the TTS API
tts_api = F5TTSAPI.get_instance()

@app.post("/generate")
async def generate_speech(
    text: str,
    reference_audio: UploadFile = File(...),
):
    """
    Generate speech from text using a reference audio file.
    
    Args:
        text: The text to convert to speech
        reference_audio: The reference audio file (WAV format)
    
    Returns:
        The generated audio file
    """
    try:
        # Save uploaded reference audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_ref:
            content = await reference_audio.read()
            temp_ref.write(content)
            temp_ref_path = temp_ref.name

        # Generate speech
        audio, sr = tts_api.generate_speech(text, temp_ref_path)

        # Save to temporary file
        output_path = Path("output") / "generated_speech.wav"
        output_path.parent.mkdir(exist_ok=True)
        torchaudio.save(str(output_path), audio.cpu(), sr)

        # Clean up temporary reference file
        os.unlink(temp_ref_path)

        # Return the generated audio file
        return FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename="generated_speech.wav"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 