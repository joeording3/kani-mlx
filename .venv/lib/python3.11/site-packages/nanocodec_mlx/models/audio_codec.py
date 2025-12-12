"""
Main AudioCodec model for MLX.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import json
from pathlib import Path

from .encoder import HiFiGANEncoder
from .decoder import CausalHiFiGANDecoder
from .quantizer_fixed import GroupFiniteScalarQuantizer
from ..utils import load_safetensors_weights

class AudioCodecModel(nn.Module):
    """
    NanoCodec Model in MLX.

    Combines encoder, quantizer, and decoder for audio compression.

    Args:
        sample_rate: Audio sample rate (22050 for nano-codec)
        encoder_config: Configuration dict for encoder
        decoder_config: Configuration dict for decoder
        quantizer_config: Configuration dict for quantizer
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        encoder_config: Optional[dict] = None,
        decoder_config: Optional[dict] = None,
        quantizer_config: Optional[dict] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate

        # Default configurations matching nanocodec
        if encoder_config is None:
            encoder_config = {
                "down_sample_rates": [2, 3, 6, 7, 7],
                "encoded_dim": 16,
                "base_channels": 24,
                "activation": "lrelu",
                "pad_mode": "replicate",
            }

        if decoder_config is None:
            decoder_config = {
                "up_sample_rates": [7, 7, 6, 3, 2],
                "input_dim": 16,
                "base_channels": 864,
                "activation": "half_snake",
                "output_activation": "clamp",
                "pad_mode": "zeros",
                "n_groups_equal_to_out_channels": True,
            }

        if quantizer_config is None:
            quantizer_config = {
                "num_groups": 4,
                "num_levels_per_group": [9, 8, 8, 7],
            }

        self.audio_encoder = HiFiGANEncoder(**encoder_config)
        self.audio_decoder = CausalHiFiGANDecoder(**decoder_config)
        self.vector_quantizer = GroupFiniteScalarQuantizer(**quantizer_config)

        total_downsample = 1
        for rate in encoder_config["down_sample_rates"]:
            total_downsample *= rate
        self.samples_per_frame = total_downsample  # 1764 for nanocodec

    def encode(
        self,
        audio: mx.array,
        audio_len: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array]:
        """
        Encode audio to quantized tokens.

        Args:
            audio: Input audio [batch, time] or [batch, 1, time]
            audio_len: Length of each audio [batch]

        Returns:
            tokens: Quantized token indices [batch, num_groups, time/downsample]
            tokens_len: Length of token sequence [batch]
        """
        if audio.ndim == 2:
            audio = mx.expand_dims(audio, axis=1)
        elif audio.ndim == 1:
            audio = mx.expand_dims(mx.expand_dims(audio, axis=0), axis=0)

        encoded, encoded_len = self.audio_encoder(audio, audio_len)
        tokens = self.vector_quantizer.encode(encoded, encoded_len)

        return tokens, encoded_len

    def decode(
        self,
        tokens: mx.array,
        tokens_len: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array]:
        """
        Decode quantized tokens to audio.

        Args:
            tokens: Quantized token indices [batch, num_groups, time]
            tokens_len: Length of token sequence [batch]

        Returns:
            audio: Reconstructed audio [batch, 1, time * upsample]
            audio_len: Length of audio sequence [batch]
        """
        encoded = self.vector_quantizer.decode(tokens, tokens_len)
        audio, audio_len = self.audio_decoder(encoded, tokens_len)

        return audio, audio_len

    def __call__(
        self,
        audio: mx.array,
        audio_len: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array, mx.array]:
        """
        Full encode-decode pass (for validation/testing).

        Args:
            audio: Input audio [batch, time] or [batch, 1, time]
            audio_len: Length of each audio [batch]

        Returns:
            reconstructed_audio: Reconstructed audio [batch, 1, time]
            tokens: Quantized tokens [batch, num_groups, time/downsample]
            audio_len: Length of output audio [batch]
        """
        tokens, tokens_len = self.encode(audio, audio_len)
        reconstructed_audio, audio_len = self.decode(tokens, tokens_len)

        return reconstructed_audio, tokens, audio_len

    @staticmethod
    def from_pretrained(
        weights_path: str,
        sample_rate: Optional[int] = None,
        **kwargs
    ):
        """
        Load a pretrained model from HuggingFace Hub.

        Args:
            weights_path: Path to HuggingFace repo ID (e.g., "username/model-name")
            sample_rate: Audio sample rate (if None, will try to load from config)
            **kwargs: Additional arguments passed to hf_hub_download

        Returns:
            model: Loaded AudioCodecModel instance

        Examples:
            # Load from HuggingFace Hub
            model = AudioCodecModel.from_pretrained("username/nanocodec-model")
        """
        
        if '/' in weights_path:
            # This looks like a HuggingFace repo ID
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required to download models from HuggingFace. "
                    "Install it with: pip install huggingface-hub"
                )

            print(f"Downloading model from HuggingFace Hub: {weights_path}")

            # Download config.json first to get sample_rate and model config
            config_path = None
            try:
                config_path = hf_hub_download(
                    repo_id=weights_path,
                    filename="config.json",
                    **kwargs
                )
            except Exception:
                print("Warning: config.json not found in repo, using defaults")

            # Download model.safetensors
            try:
                weights_file = hf_hub_download(
                    repo_id=weights_path,
                    filename="model.safetensors",
                    **kwargs
                )
            except Exception:
                # Fallback to .npz if safetensors not available
                print("model.safetensors not found, trying .npz format...")
                weights_file = hf_hub_download(
                    repo_id=weights_path,
                    filename="nemo_codec_weights.npz",
                    **kwargs
                )

            # Load config if available
            config = {}
            if config_path:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if sample_rate is None:
                        sample_rate = config.get('sample_rate', 22050)

            weights_path = weights_file

        # Default sample rate if not specified
        if sample_rate is None:
            sample_rate = 22050
            print(f"Using default sample_rate={sample_rate}")

        # Create model instance
        model = AudioCodecModel(sample_rate=sample_rate)

        # Load weights based on file extension
        if weights_path.endswith('.safetensors'):
            load_safetensors_weights(model, weights_path, verbose=True)
        else:
            raise ValueError(f"Unsupported weight format. Only .safetensors files are supported, got: {weights_path}")

        return model

    def get_info(self) -> dict:
        """Get model information."""
        
        encoder_params = sum(x.size for x in self.audio_encoder.parameters().values() if hasattr(x, 'size'))
        decoder_params = sum(x.size for x in self.audio_decoder.parameters().values() if hasattr(x, 'size'))

        return {
            "sample_rate": self.sample_rate,
            "samples_per_frame": self.samples_per_frame,
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
            "quantizer_codebook_size": self.vector_quantizer.codebook_size,
        }
