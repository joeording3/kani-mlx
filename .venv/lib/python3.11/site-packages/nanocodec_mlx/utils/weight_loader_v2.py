"""
Complete weight loader for MLX AudioCodec from PyTorch weights.
"""

import mlx.core as mx
import numpy as np
from typing import Dict, Any
import re


def compute_weight_from_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    """
    Compute actual weight from weight normalization parameters.

    PyTorch stores: weight = weight_g * (weight_v / ||weight_v||)
    """
    if weight_v.ndim == 3:  # Conv1d: [out_ch, in_ch, kernel]
        v_norm = np.linalg.norm(weight_v, axis=(1, 2), keepdims=True)
    elif weight_v.ndim == 2:
        v_norm = np.linalg.norm(weight_v, axis=1, keepdims=True)
    else:
        v_norm = np.linalg.norm(weight_v)

    v_norm = np.maximum(v_norm, 1e-8)
    weight = weight_g * (weight_v / v_norm)

    # Transpose for MLX: PyTorch uses [out_ch, in_ch, kernel] but MLX uses [out_ch, kernel, in_ch]
    if weight.ndim == 3:
        weight = np.transpose(weight, (0, 2, 1))

    return weight


def load_conv1d_weights(
    pytorch_weights: Dict[str, np.ndarray],
    pytorch_prefix: str
) -> Dict[str, mx.array]:
    """
    Load Conv1D weights (with weight norm) from PyTorch format.

    Returns dict with 'weight' and 'bias' keys.
    """
    result = {}
    g_key = f"{pytorch_prefix}.parametrizations.weight.original0"
    v_key = f"{pytorch_prefix}.parametrizations.weight.original1"
    bias_key = f"{pytorch_prefix}.bias"

    if g_key in pytorch_weights and v_key in pytorch_weights:
        weight_g = pytorch_weights[g_key]
        weight_v = pytorch_weights[v_key]
        weight = compute_weight_from_norm(weight_g, weight_v)
    elif f"{pytorch_prefix}.weight" in pytorch_weights:
        weight = pytorch_weights[f"{pytorch_prefix}.weight"]
    else:
        raise KeyError(f"Could not find weight for {pytorch_prefix}")

    result['weight'] = mx.array(weight)

    if bias_key in pytorch_weights:
        result['bias'] = mx.array(pytorch_weights[bias_key])

    return result


def load_activation_weights(
    pytorch_weights: Dict[str, np.ndarray],
    pytorch_prefix: str
) -> Dict[str, mx.array]:
    """
    Load activation parameters (e.g., snake_act.alpha).
    """
    result = {}
    alpha_key = f"{pytorch_prefix}.activation.snake_act.alpha"

    if alpha_key in pytorch_weights:
        result['alpha'] = mx.array(pytorch_weights[alpha_key])

    return result


def load_encoder_weights(
    model,
    pytorch_weights: Dict[str, np.ndarray]
) -> int:
    """
    Load all encoder weights from PyTorch.

    Returns number of parameters loaded.
    """
    count = 0
    pre_conv_weights = load_conv1d_weights(
        pytorch_weights,
        "audio_encoder.pre_conv.conv"
    )
    model.audio_encoder.pre_conv.conv.weight = pre_conv_weights['weight']
    if 'bias' in pre_conv_weights:
        model.audio_encoder.pre_conv.conv.bias = pre_conv_weights['bias']
    count += 2

    for i in range(len(model.audio_encoder.down_sample_conv_layers)):
        conv_weights = load_conv1d_weights(
            pytorch_weights,
            f"audio_encoder.down_sample_conv_layers.{i}.conv"
        )
        model.audio_encoder.down_sample_conv_layers[i].conv.weight = conv_weights['weight']
        if 'bias' in conv_weights:
            model.audio_encoder.down_sample_conv_layers[i].conv.bias = conv_weights['bias']
        count += 2

    # res_layers[i].res_blocks[j].res_blocks[k]
    for i in range(len(model.audio_encoder.res_layers)):
        res_layer = model.audio_encoder.res_layers[i]
        for j in range(len(res_layer.res_blocks)):
            res_block = res_layer.res_blocks[j]
            for k in range(len(res_block.res_blocks)):
                residual_block = res_block.res_blocks[k]
                input_conv_weights = load_conv1d_weights(
                    pytorch_weights,
                    f"audio_encoder.res_layers.{i}.res_blocks.{j}.res_blocks.{k}.input_conv.conv"
                )
                residual_block.input_conv.conv.weight = input_conv_weights['weight']
                if 'bias' in input_conv_weights:
                    residual_block.input_conv.conv.bias = input_conv_weights['bias']
                count += 2

                skip_conv_weights = load_conv1d_weights(
                    pytorch_weights,
                    f"audio_encoder.res_layers.{i}.res_blocks.{j}.res_blocks.{k}.skip_conv.conv"
                )
                residual_block.skip_conv.conv.weight = skip_conv_weights['weight']
                if 'bias' in skip_conv_weights:
                    residual_block.skip_conv.conv.bias = skip_conv_weights['bias']
                count += 2

    post_conv_weights = load_conv1d_weights(
        pytorch_weights,
        "audio_encoder.post_conv.conv"
    )
    model.audio_encoder.post_conv.conv.weight = post_conv_weights['weight']
    if 'bias' in post_conv_weights:
        model.audio_encoder.post_conv.conv.bias = post_conv_weights['bias']
    count += 2

    return count


def load_decoder_weights(
    model,
    pytorch_weights: Dict[str, np.ndarray]
) -> int:
    """
    Load all decoder weights from PyTorch.

    Returns number of parameters loaded.
    """
    count = 0

    pre_conv_weights = load_conv1d_weights(
        pytorch_weights,
        "audio_decoder.pre_conv.conv"
    )
    model.audio_decoder.pre_conv.conv.weight = pre_conv_weights['weight']
    if 'bias' in pre_conv_weights:
        model.audio_decoder.pre_conv.conv.bias = pre_conv_weights['bias']
    count += 2

    for i in range(len(model.audio_decoder.activations)):
        alpha_weights = load_activation_weights(
            pytorch_weights,
            f"audio_decoder.activations.{i}"
        )
        if 'alpha' in alpha_weights:
            model.audio_decoder.activations[i].alpha = alpha_weights['alpha']
            count += 1

    for i in range(len(model.audio_decoder.up_sample_conv_layers)):
        # Load weights for grouped ConvTranspose1d
        # PyTorch stores as [in_ch, out_ch/groups, kernel] with groups=out_ch
        # After transpose: [in_ch, kernel, out_ch/groups]
        # Since groups=out_ch, out_ch/groups=1, so final shape is [in_ch, kernel, 1]
        conv_weights = load_conv1d_weights(
            pytorch_weights,
            f"audio_decoder.up_sample_conv_layers.{i}.conv"
        )
        model.audio_decoder.up_sample_conv_layers[i].weight = conv_weights['weight']
        if 'bias' in conv_weights:
            model.audio_decoder.up_sample_conv_layers[i].bias = conv_weights['bias']
        count += 2

    for i in range(len(model.audio_decoder.res_layers)):
        res_layer = model.audio_decoder.res_layers[i]
        for j in range(len(res_layer.res_blocks)):
            res_block = res_layer.res_blocks[j]
            for k in range(len(res_block.res_blocks)):
                residual_block = res_block.res_blocks[k]
                input_act_weights = load_activation_weights(
                    pytorch_weights,
                    f"audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{k}.input_activation"
                )
                if 'alpha' in input_act_weights:
                    residual_block.input_activation.alpha = input_act_weights['alpha']
                    count += 1

                skip_act_weights = load_activation_weights(
                    pytorch_weights,
                    f"audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{k}.skip_activation"
                )
                if 'alpha' in skip_act_weights:
                    residual_block.skip_activation.alpha = skip_act_weights['alpha']
                    count += 1
                input_conv_weights = load_conv1d_weights(
                    pytorch_weights,
                    f"audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{k}.input_conv.conv"
                )
                residual_block.input_conv.conv.weight = input_conv_weights['weight']
                if 'bias' in input_conv_weights:
                    residual_block.input_conv.conv.bias = input_conv_weights['bias']
                count += 2
                skip_conv_weights = load_conv1d_weights(
                    pytorch_weights,
                    f"audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{k}.skip_conv.conv"
                )
                residual_block.skip_conv.conv.weight = skip_conv_weights['weight']
                if 'bias' in skip_conv_weights:
                    residual_block.skip_conv.conv.bias = skip_conv_weights['bias']
                count += 2

    post_act_weights = load_activation_weights(
        pytorch_weights,
        "audio_decoder.post_activation"
    )
    if 'alpha' in post_act_weights:
        model.audio_decoder.post_activation.alpha = post_act_weights['alpha']
        count += 1

    post_conv_weights = load_conv1d_weights(
        pytorch_weights,
        "audio_decoder.post_conv.conv"
    )
    model.audio_decoder.post_conv.conv.weight = post_conv_weights['weight']
    if 'bias' in post_conv_weights:
        model.audio_decoder.post_conv.conv.bias = post_conv_weights['bias']
    count += 2

    return count


def load_quantizer_weights(
    model,
    pytorch_weights: Dict[str, np.ndarray]
) -> int:
    """
    Load quantizer configuration from PyTorch weights.
    """
    count = 0

    for i in range(len(model.vector_quantizer.fsqs)):
        num_levels_key = f"vector_quantizer.fsqs.{i}.num_levels"
        if num_levels_key in pytorch_weights:
            num_levels = pytorch_weights[num_levels_key].squeeze()  # [4]
            model.vector_quantizer.fsqs[i].num_levels = mx.array(num_levels, dtype=mx.int32)
            count += 1

        dim_base_index_key = f"vector_quantizer.fsqs.{i}.dim_base_index"
        if dim_base_index_key in pytorch_weights:
            dim_base_index = pytorch_weights[dim_base_index_key].squeeze()  # [4]
            model.vector_quantizer.fsqs[i].dim_base_index = mx.array(dim_base_index, dtype=mx.int32)
            count += 1

    return count


def load_pretrained_weights(
    model,
    weights_path: str,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Load pretrained PyTorch weights into MLX model.

    Args:
        model: MLX AudioCodecModel instance
        weights_path: Path to .npz file with PyTorch weights
        verbose: Print progress

    Returns:
        Dict with counts of loaded parameters per component
    """
    if verbose:
        print(f"Loading pretrained weights from {weights_path}...")

    pytorch_weights_npz = np.load(weights_path)
    pytorch_weights = {k: pytorch_weights_npz[k] for k in pytorch_weights_npz.files}

    if verbose:
        print(f"Found {len(pytorch_weights)} weight tensors in checkpoint")

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
        print(f"\n✓ Successfully loaded pretrained weights!")
        print(f"  Total parameter groups loaded: {sum(counts.values())}")

    return counts
