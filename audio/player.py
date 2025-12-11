"""Audio player for LLM-generated speech tokens."""

from __future__ import annotations

from collections.abc import Sequence
import logging

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError:  # pragma: no cover - Linux CI image
    mx = None
try:
    from nanocodec_mlx.models.audio_codec import AudioCodecModel
except ImportError:  # pragma: no cover - nanocodec-mlx is macOS-only
    AudioCodecModel = None

from config import (
    AUDIO_TOKENS_START,
    CODEBOOK_SIZE,
    CODEC_MODEL_NAME,
    END_OF_AI,
    END_OF_HUMAN,
    END_OF_SPEECH,
    END_OF_TEXT,
    PAD_TOKEN,
    START_OF_AI,
    START_OF_HUMAN,
    START_OF_SPEECH,
    START_OF_TEXT,
    TOKENIZER_LENGTH,
)

from .tokenizer_types import TokenizerLike

logger = logging.getLogger(__name__)


class _PlaceholderAudioCodecModel:
    """Fallback codec that emits silence when nanocodec-mlx is unavailable."""

    def decode(
        self,
        audio_codes: mx.ndarray,
        lengths: mx.ndarray | None,
    ) -> tuple[mx.ndarray, mx.ndarray]:
        try:
            num_frames = max(1, int(audio_codes.shape[-1]))
        except Exception:
            num_frames = 1
        if mx is None:
            reconstructed = np.zeros((1, 1, num_frames), dtype=np.float32)
            recon_len = np.array([num_frames], dtype=np.int32)
        else:
            reconstructed = mx.zeros((1, 1, num_frames), dtype=mx.float32)
            recon_len = mx.array([num_frames], dtype=mx.int32)
        return reconstructed, recon_len


class AudioValidationError(ValueError):
    """Base for MLX audio decoding validation failures."""

    message = "Audio validation failed"

    def __init__(self) -> None:
        super().__init__(self.message)


class MissingSpeechTokensError(AudioValidationError):
    """Raised when the special speech markers are absent."""

    message = "Special speech tokens not exist!"


class InvalidAudioCodesSequenceError(AudioValidationError):
    """Raised when the start/end speech markers are out of order."""

    message = "Invalid audio codes sequence!"


class InvalidAudioLengthError(AudioValidationError):
    """Raised when the decoded sequence length is malformed."""

    message = "The length of the sequence must be a multiple of 4!"


class InvalidAudioTokensError(AudioValidationError):
    """Raised when decoded tokens fall below the expected range."""

    message = "Invalid audio tokens!"


class LLMAudioPlayer:
    """Helper for decoding audio tokens emitted by the LLM."""

    def __init__(self, tokenizer: TokenizerLike) -> None:
        self.codec_model = self._build_codec_model()

        self.tokenizer: TokenizerLike = tokenizer

        self.tokeniser_length = TOKENIZER_LENGTH
        self.start_of_text = START_OF_TEXT
        self.end_of_text = END_OF_TEXT
        self.start_of_speech = START_OF_SPEECH
        self.end_of_speech = END_OF_SPEECH
        self.start_of_human = START_OF_HUMAN
        self.end_of_human = END_OF_HUMAN
        self.start_of_ai = START_OF_AI
        self.end_of_ai = END_OF_AI
        self.pad_token = PAD_TOKEN
        self.audio_tokens_start = AUDIO_TOKENS_START
        self.codebook_size = CODEBOOK_SIZE

    def _build_codec_model(self) -> AudioCodecModel | _PlaceholderAudioCodecModel:
        if AudioCodecModel is None:
            logger.warning(
                "nanocodec-mlx is unavailable; Kani MLX audio will be silent",
            )
            return _PlaceholderAudioCodecModel()

        return AudioCodecModel.from_pretrained(CODEC_MODEL_NAME)

    def output_validation(self, out_ids: mx.ndarray) -> None:
        """Ensure the speech token markers are present before decoding."""
        start_of_speech_flag = self.start_of_speech in out_ids
        end_of_speech_flag = self.end_of_speech in out_ids
        if not (start_of_speech_flag and end_of_speech_flag):
            raise MissingSpeechTokensError()

    def get_nano_codes(self, out_ids: mx.ndarray) -> tuple[mx.ndarray, mx.ndarray]:
        """Extract the encoded audio frames and length for the codec model."""
        start_a_idx = (out_ids == self.start_of_speech).tolist().index(True)
        end_a_idx   = (out_ids == self.end_of_speech).tolist().index(True)
        if start_a_idx >= end_a_idx:
            raise InvalidAudioCodesSequenceError()

        audio_codes = out_ids[start_a_idx+1 : end_a_idx]
        if len(audio_codes) % 4:
            raise InvalidAudioLengthError()
        audio_codes = audio_codes.reshape(-1, 4)
        audio_codes = audio_codes - mx.array([self.codebook_size * i for i in range(4)])

        audio_codes = audio_codes - self.audio_tokens_start
        if (audio_codes < 0).sum().item() > 0:
            raise InvalidAudioTokensError()

        audio_codes = mx.expand_dims(audio_codes.T, axis=0)
        len_ = mx.array([audio_codes.shape[-1]])
        return audio_codes, len_

    def get_text(self, out_ids: mx.ndarray) -> str | None:
        """Decode the text span from the token sequence."""
        try:
            start_t_idx = (out_ids == self.start_of_text).tolist().index(True)
            end_t_idx   = (out_ids == self.end_of_text).tolist().index(True)
        except ValueError:
            return None

        txt_tokens = out_ids[start_t_idx : end_t_idx+1]
        return self.tokenizer.decode(txt_tokens, skip_special_tokens=True)

    def get_waveform(self, out_ids: mx.ndarray) -> tuple[np.ndarray, str | None]:
        """Return the decoded waveform and optional text for a token batch."""
        out_ids = out_ids.flatten()
        self.output_validation(out_ids)
        audio_codes, len_ = self.get_nano_codes(out_ids)

        # Decode using MLX model (audio_codes and len_ are already MLX arrays)
        reconstructed_audio, recon_len = self.codec_model.decode(audio_codes, len_)
        text = self.get_text(out_ids)
        return np.array(reconstructed_audio[0, 0, :int(recon_len[0])]), text

    def decode_audio_chunk(self, audio_codes: Sequence[int]) -> np.ndarray | None:
        """Decode a chunk of audio codes (shape: [num_frames, 4])."""
        if len(audio_codes) == 0:
            return None

        # Process audio codes: subtract offsets for each codebook
        audio_codes = mx.array(audio_codes)
        audio_codes = audio_codes - mx.array([self.codebook_size * i for i in range(4)])
        audio_codes = audio_codes - self.audio_tokens_start

        if (audio_codes < 0).sum().item() > 0:
            return None  # Invalid tokens, skip

        # Shape: (1, 4, num_frames) - batch_size=1, num_codebooks=4, num_frames
        audio_codes = mx.expand_dims(audio_codes.T, axis=0)
        len_ = mx.array([audio_codes.shape[-1]])

        # Decode using MLX model (audio_codes and len_ are already MLX arrays)
        reconstructed_audio, recon_len = self.codec_model.decode(audio_codes, len_)
        return np.array(reconstructed_audio[0, 0, :int(recon_len[0])])
