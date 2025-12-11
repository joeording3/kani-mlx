"""Streaming audio writer with sliding window decoder."""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np
from scipy.io.wavfile import write

from config import CHUNK_SIZE, LOOKBACK_FRAMES, SAMPLE_RATE

from .player import LLMAudioPlayer

logger = logging.getLogger(__name__)


class StreamingAudioWriter:
    """Decode streaming tokens into audio chunks with sliding-window context."""

    def __init__(
        self,
        player: LLMAudioPlayer,
        output_file: str | None,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = CHUNK_SIZE,
        lookback_frames: int = LOOKBACK_FRAMES,
    ) -> None:
        """Initialize the sliding-window decoder state."""
        self.player = player
        self.output_file = output_file
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.lookback_frames = lookback_frames
        self.token_queue: queue.Queue[int] = queue.Queue()
        self.audio_chunks: list[np.ndarray] = []
        self.running = True
        self.inside_speech = False
        self.audio_token_buffer: list[int] = []
        self.all_tokens: list[int] = []
        self.frames_decoded = 0

    def decoder_worker(self) -> None:
        """Background thread that decodes audio chunks as they arrive."""
        speech_ended = False

        while self.running or not self.token_queue.empty():
            try:
                token_id = self.token_queue.get(timeout=0.1)

                if token_id == self.player.start_of_speech:
                    logger.debug("[DECODER] START_OF_SPEECH detected")
                    self.inside_speech = True
                    speech_ended = False
                    self.audio_token_buffer = []
                    continue

                if token_id == self.player.end_of_speech:
                    if speech_ended:
                        logger.debug(
                            "[DECODER] Warning: Duplicate END_OF_SPEECH detected, ignoring"
                        )
                        continue

                    self._handle_end_of_speech()
                    self.inside_speech = False
                    speech_ended = True
                    self.audio_token_buffer = []
                    continue

                if self.inside_speech and not speech_ended:
                    self._handle_accumulated_token(token_id)

            except queue.Empty:
                continue

    def _handle_end_of_speech(self) -> None:
        total_frames = len(self.all_tokens) // 4
        remaining_frames = total_frames - self.frames_decoded

        if remaining_frames < 1:
            return

        start_frame = max(0, self.frames_decoded - self.lookback_frames)
        start_token = start_frame * 4
        tokens_to_decode = self.all_tokens[start_token:]
        num_frames = len(tokens_to_decode) // 4

        if num_frames <= 0:
            return

        audio_chunk = self._decode_tokens_to_chunk(tokens_to_decode, num_frames)
        if audio_chunk is None:
            return

        self._append_audio_from_chunk(audio_chunk, num_frames, remaining_frames)
        logger.debug(
            "[DECODER] Final chunk: %d frames (%.2fs audio)",
            remaining_frames,
            remaining_frames / 12.5,
        )

    def _handle_accumulated_token(self, token_id: int) -> None:
        self.audio_token_buffer.append(token_id)
        self.all_tokens.append(token_id)

        total_frames = len(self.all_tokens) // 4
        new_frames = total_frames - self.frames_decoded

        if new_frames < self.chunk_size:
            return

        start_frame = max(0, self.frames_decoded - self.lookback_frames)
        start_token = start_frame * 4
        tokens_to_decode = self.all_tokens[start_token:]
        num_frames = len(tokens_to_decode) // 4

        if num_frames <= 0:
            return

        audio_chunk = self._decode_tokens_to_chunk(tokens_to_decode, num_frames)
        if audio_chunk is None:
            return

        self._append_audio_from_chunk(audio_chunk, num_frames, self.chunk_size)
        self.frames_decoded += self.chunk_size
        logger.debug(
            "[DECODER] Decoded %d frames (%.2fs audio)"
            " with %d-frame lookback context",
            self.chunk_size,
            self.chunk_size / 12.5,
            self.lookback_frames,
        )
        self.audio_token_buffer = []

    def _decode_tokens_to_chunk(self, tokens: list[int], num_frames: int) -> np.ndarray | None:
        if num_frames <= 0:
            return None

        codes = np.array(tokens[: num_frames * 4]).reshape(-1, 4)
        return self.player.decode_audio_chunk(codes)

    def _append_audio_from_chunk(
        self,
        audio_chunk: np.ndarray,
        num_frames: int,
        frames_to_collect: int,
    ) -> None:
        if num_frames <= 0 or frames_to_collect <= 0:
            return

        samples_per_frame = len(audio_chunk) // num_frames
        if samples_per_frame == 0:
            return

        lookback_skip = min(self.frames_decoded, self.lookback_frames)
        available_frames = num_frames - lookback_skip
        if available_frames <= 0:
            return

        frames_to_collect = min(frames_to_collect, available_frames)
        skip_samples = lookback_skip * samples_per_frame
        new_samples = frames_to_collect * samples_per_frame
        new_audio = audio_chunk[skip_samples : skip_samples + new_samples]
        self.audio_chunks.append(new_audio)
    def add_token(self, token_id: int) -> None:
        """Add a token to the processing queue."""
        self.token_queue.put(token_id)

    def finalize(self) -> np.ndarray | None:
        """Stop the decoder thread and write the final audio file."""
        self.running = False
        self.decoder_thread.join()

        if self.audio_chunks:
            full_audio = np.concatenate(self.audio_chunks)

            if self.output_file:
                write(self.output_file, self.sample_rate, full_audio)
                logger.info(
                    "[WRITER] Wrote %.2fs of audio to %s",
                    len(full_audio) / self.sample_rate,
                    self.output_file,
                )

            return full_audio

        return None

    def start(self) -> None:
        """Start the decoder thread."""
        self.decoder_thread = threading.Thread(target=self.decoder_worker)
        self.decoder_thread.start()
