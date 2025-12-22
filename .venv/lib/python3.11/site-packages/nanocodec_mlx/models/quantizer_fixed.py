"""
Finite Scalar Quantizer (FSQ) implementation in MLX.

Based on https://arxiv.org/abs/2309.15505v1
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class FiniteScalarQuantizer(nn.Module):
    """
    Finite Scalar Quantization for a single group.

    Args:
        num_levels: List of quantization levels per dimension, e.g., [9, 8, 8, 7]
        eps: Small regularization constant for compression (default: 1e-3)
    """

    def __init__(self, num_levels: list[int], eps: float = 1e-3):
        super().__init__()

        self.num_levels = mx.array(num_levels, dtype=mx.int32)  # [D]
        self.dim = len(num_levels)
        self.eps = eps

        # Example: [9, 8, 8, 7] -> [1, 9, 72, 576]
        base_indices = [1]
        for i in range(len(num_levels) - 1):
            base_indices.append(base_indices[-1] * num_levels[i])
        self.dim_base_index = mx.array(base_indices, dtype=mx.int32)  # [D]

        self.codebook_size = int(mx.prod(self.num_levels).item())

    def compress(self, inputs: mx.array) -> mx.array:
        """
        Apply tanh-based compression to limit values.

        Args:
            inputs: Input tensor [B, D, T]

        Returns:
            Compressed output [B, D, T]
        """
        num_levels = self.num_levels[None, :, None].astype(mx.float32)
        output_scale = (num_levels - 1) / 2
        output_scale = output_scale * (1 - self.eps)
        output_offset = mx.where(self.num_levels[None, :, None] % 2 == 0, 0.5, 0.0)
        input_shift = mx.tan(output_offset / output_scale)
        compressed = output_scale * mx.tanh(inputs + input_shift) - output_offset

        return compressed

    def round_ste(self, inputs: mx.array) -> mx.array:
        """
        Round to nearest integer with straight-through estimator.

        Args:
            inputs: Input tensor [B, D, T]

        Returns:
            Rounded tensor [B, D, T]
        """
        return mx.round(inputs)

    def inputs_to_codes(self, inputs: mx.array) -> mx.array:
        """
        Convert continuous inputs to quantized codes in [-1, 1] range.

        Args:
            inputs: Input tensor [B, D, T]

        Returns:
            Quantized codes [B, D, T] with values in [-1, 1]
        """
        compressed = self.compress(inputs)
        codes_int = self.round_ste(compressed)
        scale = (self.num_levels[None, :, None] // 2).astype(mx.float32)
        codes = codes_int / scale

        return codes

    def codes_to_nonnegative(self, codes: mx.array) -> mx.array:
        """
        Convert codes centered around zero to nonnegative indices.

        Args:
            codes: Code tensor [B, D, T] with values in [-1, 1]

        Returns:
            Nonnegative indices [B, D, T] with values in [0, num_levels-1]
        """
        scale = offset = (self.num_levels[None, :, None] // 2).astype(mx.float32)
        return scale * codes + offset

    def nonnegative_to_codes(self, codes_nonnegative: mx.array) -> mx.array:
        """
        Convert nonnegative indices to codes centered around zero.

        Args:
            codes_nonnegative: Nonnegative indices [B, D, T] in [0, num_levels-1]

        Returns:
            Codes [B, D, T] with values in [-1, 1]
        """
        scale = offset = (self.num_levels[None, :, None] // 2).astype(mx.float32)
        return (codes_nonnegative - offset) / scale

    def codes_to_indices(self, codes: mx.array) -> mx.array:
        """
        Convert code vectors to flat indices.

        Args:
            codes: Code tensor [B, D, T] with values in [-1, 1]

        Returns:
            Flat indices [B, T] with values in [0, codebook_size-1]
        """
        indices_per_dim = self.codes_to_nonnegative(codes)
        dim_base = self.dim_base_index[None, :, None].astype(mx.float32)
        flat_indices = mx.sum(indices_per_dim * dim_base, axis=1)

        return flat_indices.astype(mx.int32)

    def indices_to_codes(self, indices: mx.array) -> mx.array:
        """
        Convert flat indices to code vectors.

        Args:
            indices: Flat indices [B, T] with values in [0, codebook_size-1]

        Returns:
            Codes [B, D, T] with values in [-1, 1]
        """
        indices_expanded = indices[:, None, :]  # [B, 1, T]
        dim_base = self.dim_base_index[None, :, None].astype(mx.int32)
        num_levels = self.num_levels[None, :, None]
        codes_nonnegative = (indices_expanded // dim_base) % num_levels
        codes = self.nonnegative_to_codes(codes_nonnegative.astype(mx.float32))

        return codes

    def encode(self, inputs: mx.array) -> mx.array:
        """
        Encode continuous inputs to discrete indices.

        Args:
            inputs: Input tensor [B, D, T]

        Returns:
            Flat indices [B, T] with values in [0, codebook_size-1]
        """
        codes = self.inputs_to_codes(inputs)
        indices = self.codes_to_indices(codes)

        return indices

    def decode(self, indices: mx.array) -> mx.array:
        """
        Decode discrete indices to continuous codes.

        Args:
            indices: Flat indices [B, T] with values in [0, codebook_size-1]

        Returns:
            Dequantized codes [B, D, T] with values in [-1, 1]
        """
        codes = self.indices_to_codes(indices)
        return codes

    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        """
        Quantize and return both codes and indices.

        Args:
            inputs: Input tensor [B, D, T]

        Returns:
            codes: Quantized codes [B, D, T]
            indices: Flat indices [B, T]
        """
        codes = self.inputs_to_codes(inputs)
        indices = self.codes_to_indices(codes)
        return codes, indices


class GroupFiniteScalarQuantizer(nn.Module):
    """
    Grouped Finite Scalar Quantization.

    Args:
        num_groups: Number of groups (4 for nanocodec)
        num_levels_per_group: List of levels for each dimension within a group
                              E.g., [9, 8, 8, 7] means each group has 4 dimensions
    """

    def __init__(
        self,
        num_groups: int = 4,
        num_levels_per_group: list[int] = [9, 8, 8, 7],
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_levels_per_group = num_levels_per_group
        self.fsqs = [
            FiniteScalarQuantizer(num_levels=num_levels_per_group)
            for _ in range(num_groups)
        ]

        fsq_codebook_size = 1
        for levels in num_levels_per_group:
            fsq_codebook_size *= levels
        self.codebook_size = fsq_codebook_size ** num_groups

    def encode(self, x: mx.array, input_len: Optional[mx.array] = None) -> mx.array:
        """
        Encode input to quantized indices.

        Args:
            x: Input tensor [B, C, T]
            input_len: Optional length tensor [B] (not used in inference)

        Returns:
            tokens: Quantized token indices [B, num_groups, T]
        """
        batch, channels, time = x.shape
        channels_per_group = channels // self.num_groups
        all_indices = []
        for i in range(self.num_groups):
            start_idx = i * channels_per_group
            end_idx = start_idx + channels_per_group
            group_input = x[:, start_idx:end_idx, :]  # [B, channels_per_group, T]
            indices = self.fsqs[i].encode(group_input)  # [B, T]
            all_indices.append(indices)

        # Stack to [B, num_groups, T]
        tokens = mx.stack(all_indices, axis=1)

        return tokens

    def decode(self, tokens: mx.array, input_len: Optional[mx.array] = None) -> mx.array:
        """
        Decode quantized indices back to continuous values.

        Args:
            tokens: Quantized token indices [B, num_groups, T]
            input_len: Optional length tensor [B] (not used in inference)

        Returns:
            x: Dequantized tensor [B, C, T]
        """
        batch, num_groups, time = tokens.shape
        assert num_groups == self.num_groups
        channels_per_group = len(self.num_levels_per_group)
        all_groups = []
        for i in range(self.num_groups):
            group_indices = tokens[:, i, :]  # [B, T]
            group_codes = self.fsqs[i].decode(group_indices)  # [B, D, T]
            all_groups.append(group_codes)

        x = mx.concatenate(all_groups, axis=1)  # [B, C, T]

        return x

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """
        Quantize and dequantize in one pass.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            tokens: Quantized token indices [num_groups, B, T]
            x_quantized: Dequantized tensor [B, C, T]
        """
        tokens = self.encode(x)
        x_quantized = self.decode(tokens)

        return tokens, x_quantized
