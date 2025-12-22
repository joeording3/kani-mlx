"""
Safetensors weight loader for MLX NanoCodec.

"""

from typing import Dict
from safetensors import safe_open


def load_safetensors_weights(
    model,
    weights_path: str,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Load weights from safetensors format into MLX model.

    The safetensors file contains raw PyTorch weights including weight
    normalization parameters.

    Args:
        model: MLX AudioCodecModel instance
        weights_path: Path to .safetensors file
        verbose: Print progress

    Returns:
        Dict with counts of loaded parameters per component
    """
    if verbose:
        print(f"Loading weights from safetensors: {weights_path}...")

    # Load all tensors from safetensors
    pytorch_weights = {}
    with safe_open(weights_path, framework="numpy") as f:
        metadata = f.metadata()
        if verbose and metadata:
            print(f"Model metadata:")
            for key, value in metadata.items():
                if key != 'config':  # config might be long, skip it
                    print(f"  {key}: {value}")

        for key in f.keys():
            pytorch_weights[key] = f.get_tensor(key)

    if verbose:
        print(f"Found {len(pytorch_weights)} weight tensors in safetensors file")

    # Use loading functions from weight_loader_v2
    from .weight_loader_v2 import (
        load_encoder_weights,
        load_decoder_weights,
        load_quantizer_weights
    )

    counts = {}
    if verbose:
        print("\nLoading encoder weights...")
    counts['encoder'] = load_encoder_weights(model, pytorch_weights)
    if verbose:
        print(f"  Loaded {counts['encoder']} encoder parameter groups")

    if verbose:
        print("\nLoading decoder weights...")
    counts['decoder'] = load_decoder_weights(model, pytorch_weights)
    if verbose:
        print(f"  Loaded {counts['decoder']} decoder parameter groups")

    if verbose:
        print("\nLoading quantizer configuration...")
    counts['quantizer'] = load_quantizer_weights(model, pytorch_weights)
    if verbose:
        print(f"  Quantizer uses fixed configuration (no trainable weights)")

    if verbose:
        print(f"\n✓ Successfully loaded weights from safetensors!")
        print(f"  Total parameter groups loaded: {sum(counts.values())}")

    return counts
