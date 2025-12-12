"""
HiFiGAN Encoder implementation in MLX.
"""

import mlx.core as mx
import mlx.nn as nn

from ..layers import Conv1dNorm, HiFiGANResLayer, CodecActivation


class HiFiGANEncoder(nn.Module):
    """
    HiFiGAN-based audio encoder.

    Args:
        down_sample_rates: List of downsampling rates for each layer [2, 3, 6, 7, 7]
        encoded_dim: Output dimension (16 for nanocodec)
        base_channels: Base number of channels (24 for nanocodec)
        activation: Activation function type ('lrelu')
        pad_mode: Padding mode ('replicate' or 'zeros')
    """

    def __init__(
        self,
        down_sample_rates: list[int] = [2, 3, 6, 7, 7],
        encoded_dim: int = 16,
        base_channels: int = 24,
        activation: str = "lrelu",
        pad_mode: str = "replicate",
    ):
        super().__init__()
        self.down_sample_rates = down_sample_rates
        self.encoded_dim = encoded_dim
        self.base_channels = base_channels

        self.pre_conv = Conv1dNorm(
            in_channels=1,
            out_channels=base_channels,
            kernel_size=7,
            stride=1,
            padding_mode=pad_mode,
            padding=3,  # PyTorch uses kernel=7, padding=3
        )

        self.activations = [
            CodecActivation() for _ in range(len(down_sample_rates))
        ]

        # Using padding values from PyTorch
        # Strides: [2, 3, 6, 7, 7] Padding: [1, 2, 3, 4, 4]
        pytorch_padding = [1, 2, 3, 4, 4]

        self.down_sample_conv_layers = []
        in_ch = base_channels
        for i, rate in enumerate(down_sample_rates):
            out_ch = in_ch * 2
            if rate == 2:
                kernel_size = 4
            elif rate == 3:
                kernel_size = 6
            elif rate == 6:
                kernel_size = 12
            elif rate == 7:
                kernel_size = 14
            else:
                kernel_size = rate * 2

            self.down_sample_conv_layers.append(
                Conv1dNorm(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=rate,
                    padding_mode=pad_mode,
                    padding=pytorch_padding[i],  # Use PyTorch padding
                )
            )
            in_ch = out_ch

        self.res_layers = []
        in_ch = base_channels
        for i, rate in enumerate(down_sample_rates):
            self.res_layers.append(
                HiFiGANResLayer(
                    channels=in_ch,  # process at current channel count
                    kernel_sizes=[3, 7, 11],
                    dilations=[1, 3, 5],
                    padding_mode=pad_mode,
                )
            )
            in_ch = in_ch * 2

        self.final_activation = CodecActivation()
        final_channels = base_channels * (2 ** len(down_sample_rates))
        self.post_conv = Conv1dNorm(
            in_channels=final_channels,
            out_channels=encoded_dim,
            kernel_size=7,
            stride=1,
            padding_mode=pad_mode,
            padding=3,  # PyTorch uses kernel=7, padding=3
        )

    def __call__(self, x: mx.array, audio_len: mx.array = None) -> tuple[mx.array, mx.array]:
        """
        Encode audio to latent representation.

        Args:
            x: Input audio [batch, channels, time] or [batch, time]
            audio_len: Length of each audio in batch [batch]

        Returns:
            encoded: Encoded representation [batch, encoded_dim, time/downsample_factor]
            encoded_len: Length of encoded sequence [batch]
        """
        if x.ndim == 2:
            x = mx.expand_dims(x, axis=1)

        x = self.pre_conv(x)
        for i, (res_layer, activation, down_conv) in enumerate(
            zip(self.res_layers, self.activations, self.down_sample_conv_layers)
        ):
            x = res_layer(x)
            x = activation(x)
            x = down_conv(x)

            mx.eval(x)

            # Clear cache every 2 layers
            if i % 2 == 1:
                mx.metal.clear_cache()

        x = self.final_activation(x)
        x = self.post_conv(x)

        if audio_len is not None:
            total_downsample = 1
            for rate in self.down_sample_rates:
                total_downsample *= rate
            encoded_len = audio_len // total_downsample
        else:
            encoded_len = mx.array([x.shape[2]] * x.shape[0])

        return x, encoded_len
