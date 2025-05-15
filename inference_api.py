import torch
import torchaudio
from pathlib import Path
import yaml
from safetensors.torch import load_file
import argparse
import time
from f5_tts.infer.utils_infer import load_vocoder

class F5TTSAPI:
    _instance = None
    _model = None
    _device = None
    _config = None
    _vocoder = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if F5TTSAPI._model is None:
            self._load_model()

    def _load_model(self):
        # Load configuration
        config_path = Path("config.txt")
        if not config_path.exists():
            raise FileNotFoundError("config.txt not found")
        
        # Read config values
        self._config = {}
        with open(config_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    self._config[key.strip()] = value.strip()

        # Set device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._device = torch.device("cpu")
            print("Warning: CUDA not available, using CPU instead")
        
        # Load model configuration
        model_dir = Path(self._config['MODEL_DIR'])
        with open(model_dir / "transformer_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Load the model using safetensors
        model_path = str(model_dir / "model_1200000.safetensors")
        print(f"Loading model from: {model_path}")
        
        # Load state dict directly to GPU if available
        if torch.cuda.is_available():
            state_dict = load_file(model_path, device="cuda")
        else:
            state_dict = load_file(model_path, device="cpu")
        
        # Create model from config and load state dict
        from f5_tts.model import CFM
        from f5_tts.model.backbones.dit import DiT
        
        # Create transformer with exact config values
        transformer = DiT(
            dim=config['dim'],  # 1024
            depth=config['depth'],  # 22
            heads=config['heads'],  # 16
            dim_head=64,
            dropout=0.1,
            ff_mult=config['ff_mult'],  # 2
            mel_dim=80,
            text_num_embeds=256,
            text_dim=config['text_dim'],  # 512
            text_mask_padding=True,
            qk_norm=None,
            conv_layers=config['conv_layers'],  # 4
            pe_attn_head=None,
            long_skip_connection=True,
            checkpoint_activations=False
        ).to(self._device)
        
        # Create CFM model with exact parameters
        model = CFM(
            transformer=transformer,
            sigma=0.0,
            audio_drop_prob=0.3,
            cond_drop_prob=0.2,
            num_channels=80,
            mel_spec_kwargs={
                'n_mel_channels': 80,
                'target_sample_rate': int(self._config['SAMPLE_RATE']),
                'hop_length': 256,
                'win_length': 1024,
                'n_fft': 1024,
                'mel_spec_type': 'vocos'
            }
        ).to(self._device)
        
        # Load state dict and handle any missing keys
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print("\nDetailed state dict analysis:")
            print("---------------------------")
            
            # Get model state dict
            model_state = model.state_dict()
            
            # Find missing keys
            missing_keys = set(model_state.keys()) - set(state_dict.keys())
            if missing_keys:
                print("\nMissing keys in state dict:")
                for key in sorted(missing_keys):
                    print(f"- {key}")
            
            # Find unexpected keys
            unexpected_keys = set(state_dict.keys()) - set(model_state.keys())
            if unexpected_keys:
                print("\nUnexpected keys in state dict:")
                for key in sorted(unexpected_keys):
                    print(f"- {key}")
            
            print("\nAttempting to load with strict=False...")
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully with strict=False")
        
        model.eval()
        F5TTSAPI._model = model

        # Load vocoder
        print("Loading vocoder...")
        self._vocoder = load_vocoder(
            vocoder_name="vocos",
            is_local=True,
            local_path=str(model_dir / "vocos-mel-24khz"),
            device=self._device
        )
        print("Vocoder loaded successfully")
        
        print("\nModel and vocoder loaded and ready for inference")

    def generate_speech(self, text, reference_audio):
        """
        Generate speech from text using reference audio
        
        Args:
            text (str): Text to convert to speech
            reference_audio (str): Path to reference audio file
            
        Returns:
            tuple: (audio_tensor, sample_rate)
        """
        start_time = time.time()
        
        try:
            # Load reference audio
            print(f"Loading reference audio from: {reference_audio}")
            ref_audio, ref_sr = torchaudio.load(reference_audio)
            print(f"Reference audio loaded. Shape: {ref_audio.shape}, Sample rate: {ref_sr}")
            
            # Normalize reference audio
            rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
            target_rms = 0.1  # Default from infer_cli.py
            if rms < target_rms:
                ref_audio = ref_audio * target_rms / rms
            
            # Resample if needed
            if ref_sr != int(self._config['SAMPLE_RATE']):
                resampler = torchaudio.transforms.Resample(ref_sr, int(self._config['SAMPLE_RATE']))
                ref_audio = resampler(ref_audio)
            
            ref_audio = ref_audio.to(self._device)
            print(f"Reference audio moved to device: {self._device}")
            
            # Convert text to list of strings for proper tokenization
            text_list = [text]
            print(f"Text to synthesize: {text}")
            
            # Generate speech
            print("Starting speech generation...")
            with torch.no_grad():
                mel_spec = self._model.sample(
                    cond=ref_audio,
                    text=text_list,
                    duration=len(text) * 20,
                    steps=32,
                    cfg_strength=1.0
                )
            
            print(f"Mel spectrogram generated. Shape: {mel_spec[0].shape if isinstance(mel_spec, tuple) else mel_spec.shape}")
            
            # Convert mel spectrogram to audio using vocoder
            print("Converting mel spectrogram to audio...")
            if isinstance(mel_spec, tuple):
                mel_spec = mel_spec[0]
            
            # Process mel spectrogram
            mel_spec = mel_spec.to(torch.float32)
            mel_spec = mel_spec.permute(0, 2, 1)  # Change to [batch, channels, time]
            
            # Generate audio using vocoder
            audio = self._vocoder.decode(mel_spec)
            
            # Normalize output audio
            if rms < target_rms:
                audio = audio * rms / target_rms
            
            print(f"Audio generated. Shape: {audio.shape}")
            
            generation_time = time.time() - start_time
            print(f"Generation time: {generation_time:.2f} seconds")
            
            return audio, int(self._config['SAMPLE_RATE'])
            
        except Exception as e:
            print(f"Error in generate_speech: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

def main():
    parser = argparse.ArgumentParser(description='F5-TTS Inference API')
    parser.add_argument('--ref_audio', type=str, required=True,
                      help='Path to reference audio file')
    parser.add_argument('--gen_text', type=str, required=True,
                      help='Text to generate speech for')
    parser.add_argument('--output', type=str, default='output.wav',
                      help='Output audio file name')
    
    args = parser.parse_args()
    
    # Get API instance (model will be loaded only once)
    api = F5TTSAPI.get_instance()
    
    # Generate speech
    audio, sr = api.generate_speech(args.gen_text, args.ref_audio)
    
    # Save the generated audio
    torchaudio.save(args.output, audio.cpu(), sr)
    print(f"Audio generated and saved to {args.output}")

if __name__ == "__main__":
    main() 