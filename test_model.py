from inference_api import F5TTSAPI
import torchaudio
import os

def test_model():
    print("Starting model test...")
    
    # Initialize the API
    api = F5TTSAPI.get_instance()
    
    # Test parameters
    test_text = "This is a test of the F5-TTS model."
    ref_audio = "reference/reference.wav"  # Make sure this file exists
    output_file = "output/test_output.wav"
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    print(f"\nGenerating speech for text: '{test_text}'")
    print(f"Using reference audio: {ref_audio}")
    
    try:
        # Generate speech
        audio, sr = api.generate_speech(test_text, ref_audio)
        
        # Save the generated audio
        torchaudio.save(output_file, audio.cpu(), sr)
        
        print(f"\nTest successful!")
        print(f"Generated audio saved to: {output_file}")
        print(f"Audio shape: {audio.shape}")
        print(f"Sample rate: {sr}")
        
        return True
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    test_model() 