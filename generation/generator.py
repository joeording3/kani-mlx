"""Text-to-speech generation logic."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from audio.tokenizer_types import TokenizerLike

try:
    import mlx.core as mx
except ModuleNotFoundError:  # pragma: no cover - Linux CI image
    mx = None
try:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
except ImportError:  # pragma: no cover - optional Darwin dependency
    load = None
    stream_generate = None
    make_logits_processors = None
    make_sampler = None

from config import (
    END_OF_AI,
    END_OF_HUMAN,
    END_OF_TEXT,
    MAX_TOKENS,
    MODEL_NAME,
    REPETITION_CONTEXT_SIZE,
    REPETITION_PENALTY,
    START_OF_HUMAN,
    TEMPERATURE,
    TOP_P,
)

logger = logging.getLogger(__name__)

MLX_RUNTIME_AVAILABLE = all(
    [
        mx is not None,
        load is not None,
        stream_generate is not None,
        make_logits_processors is not None,
        make_sampler is not None,
    ]
)


class AudioTokenSink(Protocol):
    """Protocol describing minimal interface for audio token consumers."""

    def add_token(self, token_id: int) -> None:
        """Enqueue a token for downstream audio decoding."""
        ...


class MLXUnavailableError(RuntimeError):
    """Raised when the MLX backend cannot be initialized."""

    def __init__(self) -> None:
        super().__init__("MLX runtime unavailable")


class TTSGenerator:
    """High-level orchestrator for Kani TTS generation."""

    def __init__(self) -> None:
        if not MLX_RUNTIME_AVAILABLE:
            raise MLXUnavailableError()
        self.model, tokenizer = load(MODEL_NAME)
        self.tokenizer: TokenizerLike = tokenizer
        self.sampler = make_sampler(temp=TEMPERATURE, top_p=TOP_P)
        self.logits_processors = make_logits_processors(
            repetition_penalty=REPETITION_PENALTY,
            repetition_context_size=REPETITION_CONTEXT_SIZE,
        )

    def prepare_input(self, prompt: str) -> list[int]:
        """Build custom input_ids with special tokens."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = mx.array([input_ids])

        start_token = mx.array([[START_OF_HUMAN]], dtype=mx.int64)
        end_tokens = mx.array([[END_OF_TEXT, END_OF_HUMAN]], dtype=mx.int64)
        modified_input_ids = mx.concatenate([start_token, input_ids, end_tokens], axis=1)

        # Flatten to 1D list for generate function
        return modified_input_ids[0].tolist()

    def generate(
        self,
        prompt: str,
        audio_writer: AudioTokenSink,
        max_tokens: int = MAX_TOKENS,
    ) -> dict[str, Any]:
        """Generate speech tokens from a text prompt."""
        modified_input_ids = self.prepare_input(prompt)

        point_1 = time.time()

        # Stream tokens from LLM
        generated_text = ""
        all_token_ids = []

        for response in stream_generate(
            self.model,
            self.tokenizer,
            modified_input_ids,
            max_tokens=max_tokens,
            sampler=self.sampler,
            logits_processors=self.logits_processors,
        ):
            generated_text += response.text

            # Use token ID directly from response if available
            if hasattr(response, 'token') and response.token is not None:
                token_id = response.token
                all_token_ids.append(token_id)
                # print(f"[LLM] Token {len(all_token_ids)}: {token_id}")
                audio_writer.add_token(token_id)

                # Stop after END_OF_AI to avoid generating multiple turns
                if token_id == END_OF_AI:
                    logger.info("[LLM] END_OF_AI detected, stopping generation")
                    break
            else:
                # Fallback to encoding (shouldn't happen with proper stream_generate)
                logger.warning("[LLM] Warning: No token ID in response, using text encoding")

        point_2 = time.time()

        logger.debug(
            "[MAIN] Generation complete. Total tokens: %d", len(all_token_ids)
        )
        logger.debug("[MAIN] Generated text length: %d chars", len(generated_text))

        return {
            'generated_text': generated_text,
            'all_token_ids': all_token_ids,
            'generation_time': point_2 - point_1,
            'point_1': point_1,
            'point_2': point_2
        }
