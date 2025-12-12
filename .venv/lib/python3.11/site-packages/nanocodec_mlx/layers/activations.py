"""
Activation functions for MLX NanoCodec.
"""

import mlx.core as mx
import mlx.nn as nn


class CodecActivation(nn.Module):
    """Simple wrapper for LeakyReLU activation used in encoder."""

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def __call__(self, x: mx.array) -> mx.array:
        return nn.leaky_relu(x, negative_slope=self.negative_slope)


class SnakeActivation(nn.Module):
    """
    Snake activation function: x + (1/alpha) * sin(alpha * x)^2
    """

    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = mx.ones((1, channels, 1)) * alpha_init

    def __call__(self, x: mx.array) -> mx.array:
        sin_term = mx.sin(self.alpha * x)
        return x + (1.0 / self.alpha) * sin_term ** 2


class HalfSnakeActivation(nn.Module):
    """
    Half-Snake activation: combines Snake with LeakyReLU.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0, negative_slope: float = 0.01):
        super().__init__()
        self.channels = channels
        self.split_point = channels // 2
        self.alpha = mx.ones((1, self.split_point, 1)) * alpha_init
        self.negative_slope = negative_slope

    def __call__(self, x: mx.array) -> mx.array:
        x1 = x[:, :self.split_point, :]
        x2 = x[:, self.split_point:, :]
        sin_term = mx.sin(self.alpha * x1)
        x1_out = x1 + (1.0 / self.alpha) * sin_term ** 2
        x2_out = nn.leaky_relu(x2, negative_slope=self.negative_slope)

        return mx.concatenate([x1_out, x2_out], axis=1)
