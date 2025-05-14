import torch
import torchaudio
from pathlib import Path
import yaml
from safetensors.torch import load_file
import argparse
import time

class F5TTSAPI:
    _instance = None
    _model = None
    _device = None
    _config = None

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
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self._device}")
        
        # Load model configuration
        model_dir = Path(self._config['MODEL_DIR'])
        with open(model_dir / "transformer_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Load the model using safetensors
        model_path = str(model_dir / "model_1200000.safetensors")
        print(f"Loading model from: {model_path}")
        state_dict = load_file(model_path, device=self._device)
        
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
        )
        
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
        )
        
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
        print("\nModel loaded and ready for inference")

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
        
        # Load reference audio
        ref_audio, _ = torchaudio.load(reference_audio)
        ref_audio = ref_audio.to(self._device)
        
        # Generate speech
        with torch.no_grad():
            audio = self._model.sample(
                cond=ref_audio,
                text=text,
                duration=len(text) * 20,
                steps=32,
                cfg_strength=1.0
            )
        
        generation_time = time.time() - start_time
        print(f"Generation time: {generation_time:.2f} seconds")
        
        return audio, int(self._config['SAMPLE_RATE'])

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