import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from f5_tts.model import CFM
from f5_tts.model.backbones.dit import DiT
import yaml
import os

class TritonPythonModel:
    def initialize(self, args):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model configuration
        model_dir = os.path.join(os.environ.get("EXECUTION_ENV_PATH", "/workspace/F5-TTS"), "checkpoints")
        with open(os.path.join(model_dir, "transformer_config.yaml"), 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        transformer = DiT(
            dim=config['dim'],
            depth=config['depth'],
            heads=config['heads'],
            dim_head=64,
            dropout=0.1,
            ff_mult=config['ff_mult'],
            mel_dim=80,
            text_num_embeds=256,
            text_dim=config['text_dim'],
            text_mask_padding=True,
            qk_norm=None,
            conv_layers=config['conv_layers'],
            pe_attn_head=None,
            long_skip_connection=True,
            checkpoint_activations=False
        )
        
        self.model = CFM(
            transformer=transformer,
            sigma=0.0,
            audio_drop_prob=0.3,
            cond_drop_prob=0.2,
            num_channels=80,
            mel_spec_kwargs={
                'n_mel_channels': 80,
                'target_sample_rate': 24000,
                'hop_length': 256,
                'win_length': 1024,
                'n_fft': 1024,
                'mel_spec_type': 'vocos'
            }
        )
        
        # Load model weights
        self.model.load_state_dict(torch.load(
            os.path.join(model_dir, "model_1200000.safetensors"),
            map_location=self.device
        ))
        self.model.eval()
        self.model.to(self.device)

    def execute(self, requests):
        responses = []
        
        for request in requests:
            # Get inputs
            text = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()
            ref_audio = pb_utils.get_input_tensor_by_name(request, "reference_audio").as_numpy()
            
            # Convert to torch tensors
            text = torch.from_numpy(text).to(self.device)
            ref_audio = torch.from_numpy(ref_audio).to(self.device)
            
            # Generate speech
            with torch.no_grad():
                audio = self.model.sample(
                    cond=ref_audio,
                    text=text,
                    duration=len(text) * 20,
                    steps=32,
                    cfg_strength=1.0
                )
            
            # Convert to numpy and create response
            audio_np = audio.cpu().numpy()
            out_tensor = pb_utils.Tensor("audio", audio_np)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        self.model = None 