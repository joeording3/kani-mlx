"""Kani TTS - Text to Speech Generation."""

from __future__ import annotations

import logging
import time

from audio import LLMAudioPlayer, StreamingAudioWriter
from generation import TTSGenerator

from config import CHUNK_SIZE, LOOKBACK_FRAMES

logger = logging.getLogger(__name__)


def time_report(point_1: float, point_2: float, point_3: float) -> str:
    """Return a formatted timing report for the generation run."""
    model_request = point_2 - point_1
    player_time = point_3 - point_2
    total_time = point_3 - point_1
    return (
        f"SPEECH TOKENS: {model_request:.2f}\n"
        f"CODEC: {player_time:.2f}\n"
        f"TOTAL: {total_time:.2f}"
    )


def main() -> None:
    """Run a quick local demonstration of the Kani TTS pipeline."""
    # Initialize generator and audio player
    generator = TTSGenerator()
    player = LLMAudioPlayer(generator.tokenizer)

    # Set prompt
    prompt = (
        "katie: Oh, yeah. I mean did you want to get a quick snack together "
        "or maybe something before you go?"
    )

    # Create streaming audio writer with sliding window decoder
    # Uses lookback context from previous frames to maintain codec continuity
    audio_writer = StreamingAudioWriter(
        player,
        'output.wav',
        chunk_size=CHUNK_SIZE,        # Output 25 new frames (2.0s) per iteration
        lookback_frames=LOOKBACK_FRAMES    # Include 15 previous frames (1.2s) for context
    )
    audio_writer.start()

    # Generate speech
    result = generator.generate(prompt, audio_writer)

    # Finalize and write audio file
    audio_writer.finalize()

    point_3 = time.time()

    # Print results
    logger.info(
        "%s",
        time_report(result['point_1'], result['point_2'], point_3),
    )
    # print(f"\n[DEBUG] First 100 chars of generated text: {result['generated_text'][:100]}")
    # print(f"[DEBUG] Last 100 chars of generated text: {result['generated_text'][-100:]}")


if __name__ == "__main__":
    main()

