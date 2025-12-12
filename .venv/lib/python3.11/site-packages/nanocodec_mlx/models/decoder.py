"""
Causal HiFiGAN Decoder implementation in MLX.
"""

import mlx.core as mx
import mlx.nn as nn

from ..layers import Conv1dNorm, HiFiGANResLayer
from ..layers.activations import HalfSnakeActivation


class CausalConv1d(nn.Module):
    """
    Causal Conv1D layer for streaming applications.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.padding > 0:
            pad_width = [(0, 0), (0, 0), (self.padding, 0)]
            x = mx.pad(x, pad_width)

        x = mx.transpose(x, (0, 2, 1))
        x = self.conv(x)
        x = mx.transpose(x, (0, 2, 1))

        return x


class CausalConvTranspose1d(nn.Module):
    """
    Causal grouped transposed convolution for upsampling.

    NanoCodec uses groups=out_channels, meaning each output channel
    is produced by a separate group with in_channels/out_channels inputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = out_channels  # NanoCodec pattern: groups = out_channels
        self.trim = kernel_size - stride
        self.channels_per_group = in_channels // out_channels
        assert self.channels_per_group * out_channels == in_channels, \
            f"in_channels ({in_channels}) must be divisible by out_channels ({out_channels})"

        # Create placeholder that will be replaced by weight loader
        self.weight = mx.zeros((in_channels, kernel_size, 1))
        self.bias = mx.zeros((out_channels,))

    def _upsample_with_zeros(self, x: mx.array, stride: int) -> mx.array:
        """
        Upsample by inserting zeros between samples.
        Optimized to reduce memory allocations.

        Args:
            x: Input [batch, channels, time]
            stride: Upsampling factor

        Returns:
            Upsampled tensor [batch, channels, upsampled_time]
        """
        if stride == 1:
            return x

        batch, channels, time = x.shape
        x_expanded = mx.expand_dims(x, axis=-1)
        pad_width = [(0, 0), (0, 0), (0, 0), (0, stride - 1)]
        x_padded = mx.pad(x_expanded, pad_width, constant_values=0)
        x_flat = mx.reshape(x_padded, (batch, channels, time * stride))
        upsampled_time = time + (time - 1) * (stride - 1)
        return x_flat[:, :, :upsampled_time]

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply grouped transposed convolution.

        Args:
            x: Input tensor [batch, in_channels, time]

        Returns:
            Output tensor [batch, out_channels, upsampled_time]
        """
        batch, in_ch, time = x.shape
        assert in_ch == self.in_channels, f"Expected {self.in_channels} input channels, got {in_ch}"
        x_grouped = mx.reshape(x, (batch, self.groups, self.channels_per_group, time))
        weight_grouped = mx.reshape(self.weight, (self.groups, self.channels_per_group, self.kernel_size, 1))
        weight_flipped = weight_grouped[:, :, ::-1, :]

        # Process each group separately to minimize peak memory
        output_list = []
        for g in range(self.groups):
            x_g = x_grouped[:, g, :, :]
            x_up = self._upsample_with_zeros(x_g, self.stride)
            padding = self.kernel_size - 1
            if padding > 0:
                pad_width = [(0, 0), (0, 0), (padding, padding)]
                x_up = mx.pad(x_up, pad_width)

            x_conv = mx.transpose(x_up, (0, 2, 1))
            w_g = mx.transpose(weight_flipped[g], (2, 1, 0))
            y_g = mx.conv1d(x_conv, w_g, stride=1, padding=0)
            y_g = mx.transpose(y_g, (0, 2, 1))
            
            mx.eval(y_g)
            output_list.append(y_g)

            # Clear cache after each group
            mx.metal.clear_cache()

        output = mx.concatenate(output_list, axis=1)

        # Add bias
        bias_expanded = mx.reshape(self.bias, (1, self.out_channels, 1))
        output = output + bias_expanded

        if self.trim > 0 and output.shape[2] > self.trim:
            output = output[:, :, :-self.trim]

        return output


class CausalHiFiGANDecoder(nn.Module):
    """
    Causal HiFiGAN decoder for audio reconstruction.

    Args:
        up_sample_rates: List of upsampling rates [7, 7, 6, 3, 2]
        input_dim: Input dimension (16 for nanocodec)
        base_channels: Base number of channels (864 for nanocodec)
        activation: Activation function ('half_snake')
        output_activation: Output activation ('clamp')
        pad_mode: Padding mode ('zeros' for causal)
        n_groups_equal_to_out_channels: Use group norm with groups=channels
    """

    def __init__(
        self,
        up_sample_rates: list[int] = [7, 7, 6, 3, 2],
        input_dim: int = 16,
        base_channels: int = 864,
        activation: str = "half_snake",
        output_activation: str = "clamp",
        pad_mode: str = "zeros",
        n_groups_equal_to_out_channels: bool = True,
    ):
        super().__init__()
        self.up_sample_rates = up_sample_rates
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.output_activation_type = output_activation
        
        self.pre_conv = CausalConv1d(
            in_channels=input_dim,
            out_channels=base_channels,
            kernel_size=7,
            stride=1,
        )

        self.up_sample_conv_layers = []
        self.activations = []
        self.res_layers = []
        in_ch = base_channels

        for i, rate in enumerate(up_sample_rates):
            out_ch = in_ch // 2

            if activation == "half_snake":
                self.activations.append(HalfSnakeActivation(in_ch))
            else:
                from ..layers import CodecActivation
                self.activations.append(CodecActivation())

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

            self.up_sample_conv_layers.append(
                CausalConvTranspose1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=rate,
                )
            )

            self.res_layers.append(
                HiFiGANResLayer(
                    channels=out_ch,
                    kernel_sizes=[3, 7, 11],
                    dilations=[1, 3, 5],
                    padding_mode=pad_mode,
                )
            )

            in_ch = out_ch

        if activation == "half_snake":
            self.post_activation = HalfSnakeActivation(in_ch)
        else:
            from ..layers import CodecActivation
            self.post_activation = CodecActivation()

        self.post_conv = CausalConv1d(
            in_channels=in_ch,
            out_channels=1,
            kernel_size=3,  # PyTorch uses kernel_size=3 for post_conv
            stride=1,
        )

    def __call__(self, x: mx.array, tokens_len: mx.array = None) -> tuple[mx.array, mx.array]:
        """
        Decode latent representation to audio.

        Args:
            x: Encoded representation [batch, input_dim, time]
            tokens_len: Length of encoded sequence [batch]

        Returns:
            audio: Reconstructed audio [batch, 1, time * upsample_factor]
            audio_len: Length of audio sequence [batch]
        """
        x = self.pre_conv(x)

        for i, (activation, up_layer, res_layer) in enumerate(
            zip(self.activations, self.up_sample_conv_layers, self.res_layers)
        ):
            x = activation(x)
            x = up_layer(x)  # This already calls mx.eval() and clear_cache() internally
            x = res_layer(x)
            mx.eval(x)

            # Clear cache every 2 layers to reduce peak memory
            if i % 2 == 1:
                mx.metal.clear_cache()

        x = self.post_activation(x)
        x = self.post_conv(x)

        if self.output_activation_type == "clamp":
            x = mx.clip(x, -1.0, 1.0)
        elif self.output_activation_type == "tanh":
            x = mx.tanh(x)

        if tokens_len is not None:
            total_upsample = 1
            for rate in self.up_sample_rates:
                total_upsample *= rate
            audio_len = tokens_len * total_upsample
        else:
            audio_len = mx.array([x.shape[2]] * x.shape[0])

        return x, audio_len
