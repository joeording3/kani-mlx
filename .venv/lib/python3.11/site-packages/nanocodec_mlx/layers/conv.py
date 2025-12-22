"""
Convolutional layers for MLX NanoCodec.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Literal

from .activations import CodecActivation, HalfSnakeActivation


class Conv1dNorm(nn.Module):
    """
    Conv1D layer with weight normalization support.

    Supports two padding strategies:
    - replicate: Symmetric padding (for encoder)
    - zeros: Causal padding (left-only, for decoder)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding_mode: Literal["replicate", "zeros"] = "replicate",
        padding: int = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding_mode = padding_mode

        if padding is not None:
            self.padding_left = padding
            self.padding_right = padding
        elif padding_mode == "zeros":
            total_padding = (kernel_size - 1) * dilation
            self.padding_left = total_padding
            self.padding_right = 0
        else:
            total_padding = (kernel_size - 1) * dilation
            self.padding_left = total_padding // 2
            self.padding_right = total_padding - self.padding_left

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

    def _apply_padding(self, x: mx.array) -> mx.array:
        """Apply custom padding based on padding_mode."""
        if self.padding_left == 0 and self.padding_right == 0:
            return x

        if self.padding_mode == "replicate":
            if self.padding_left > 0:
                left_pad = mx.repeat(x[:, :, :1], self.padding_left, axis=2)
                x = mx.concatenate([left_pad, x], axis=2)
            if self.padding_right > 0:
                right_pad = mx.repeat(x[:, :, -1:], self.padding_right, axis=2)
                x = mx.concatenate([x, right_pad], axis=2)
            return x
        else:  # zeros
            pad_width = [(0, 0), (0, 0), (self.padding_left, self.padding_right)]
            return mx.pad(x, pad_width)

    def __call__(self, x: mx.array) -> mx.array:
        x = self._apply_padding(x)
        x = mx.transpose(x, (0, 2, 1))
        x = self.conv(x)
        x = mx.transpose(x, (0, 2, 1))

        return x


class ResidualBlock(nn.Module):
    """
    Residual block with dilated convolutions used in HiFiGAN encoder/decoder.

    For decoder (padding_mode="zeros"), uses HalfSnakeActivation.
    For encoder (padding_mode="replicate"), uses CodecActivation (LeakyReLU).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int = 1,
        padding_mode: Literal["replicate", "zeros"] = "replicate",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.dropout_p = dropout

        if padding_mode == "zeros":
            self.input_activation = HalfSnakeActivation(channels)
            self.skip_activation = HalfSnakeActivation(channels)
        else:
            self.input_activation = CodecActivation()
            self.skip_activation = CodecActivation()

        self.input_conv = Conv1dNorm(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding_mode=padding_mode,
        )

        self.skip_conv = Conv1dNorm(
            channels,
            channels,
            kernel_size,
            dilation=1,
            padding_mode=padding_mode,
        )

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def __call__(self, x: mx.array) -> mx.array:
        original_size = x.shape[2]
        out = self.input_activation(x)
        out = self.input_conv(out)
        out = self.skip_activation(out)
        out = self.skip_conv(out)

        if out.shape[2] != original_size:
            if out.shape[2] > original_size:
                out = out[:, :, :original_size]
            else:
                pad_amount = original_size - out.shape[2]
                pad_width = [(0, 0), (0, 0), (0, pad_amount)]
                out = mx.pad(out, pad_width)

        if self.dropout is not None:
            out = self.dropout(out)

        return x + out


class HiFiGANResBlock(nn.Module):
    """
    HiFiGAN residual block containing multiple residual blocks with different dilations.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: list[int],
        padding_mode: Literal["replicate", "zeros"] = "replicate",
    ):
        super().__init__()
        self.res_blocks = [
            ResidualBlock(channels, kernel_size, dilation, padding_mode)
            for dilation in dilations
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.res_blocks:
            x = block(x)
        return x


class HiFiGANResLayer(nn.Module):
    """
    Layer containing multiple HiFiGAN residual blocks with different kernel sizes.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: list[int] = [3, 7, 11],
        dilations: list[int] = [1, 3, 5],
        padding_mode: Literal["replicate", "zeros"] = "replicate",
    ):
        super().__init__()
        self.res_blocks = [
            HiFiGANResBlock(channels, kernel_size, dilations, padding_mode)
            for kernel_size in kernel_sizes
        ]

    def __call__(self, x: mx.array) -> mx.array:
        import mlx.core as mx
        residuals = []
        for i, block in enumerate(self.res_blocks):
            res = block(x)
            mx.eval(res)
            residuals.append(res)

            # Clear cache after processing each residual block to reduce memory
            if i < len(self.res_blocks) - 1:  # Don't clear on last iteration
                mx.metal.clear_cache()

        out = sum(residuals) / len(residuals)
        return out
