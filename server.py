"""FastAPI server for Kani TTS with streaming support."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import io
import logging
import queue
import struct
import threading

from audio import LLMAudioPlayer, StreamingAudioWriter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from generation import TTSGenerator
import numpy as np
from pydantic import BaseModel
from scipy.io.wavfile import write as wav_write

from config import CHUNK_SIZE, LOOKBACK_FRAMES, MAX_TOKENS, TEMPERATURE, TOP_P

logger = logging.getLogger(__name__)

app = FastAPI(title="Kani TTS API", version="1.0.0")
app.state.init_error = None

# Add CORS middleware to allow client.html to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def require_models() -> tuple[TTSGenerator, LLMAudioPlayer]:
    """Return initialized TTS generator and player or raise 503."""
    generator = getattr(app.state, "generator", None)
    player = getattr(app.state, "player", None)
    if not generator or not player:
        detail = getattr(app.state, "init_error", None) or "TTS models not initialized"
        raise HTTPException(status_code=503, detail=detail)

    return generator, player


class TTSRequest(BaseModel):
    """Payload describing options for TTS generation."""

    text: str
    temperature: float | None = TEMPERATURE
    max_tokens: int | None = MAX_TOKENS
    top_p: float | None = TOP_P
    chunk_size: int | None = CHUNK_SIZE
    lookback_frames: int | None = LOOKBACK_FRAMES


class HealthResponse(BaseModel):
    status: bool
    tts_initialized: bool
    error: str | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize models on startup."""
    logger.info("Initializing TTS models...")
    try:
        generator = TTSGenerator()
        player = LLMAudioPlayer(generator.tokenizer)
    except Exception as exc:
        app.state.init_error = f"MLX runtime unavailable: {exc}"
        logger.warning("Kani MLX initialization skipped: %s", exc)
        return

    app.state.generator = generator
    app.state.player = player
    app.state.init_error = None
    logger.info("TTS models initialized successfully!")


@app.get("/health")
async def health_check() -> HealthResponse:
    """Check if the server and TTS models are ready."""
    generator = getattr(app.state, "generator", None)
    player = getattr(app.state, "player", None)
    init_error = getattr(app.state, "init_error", None)
    return HealthResponse(
        status=True,
        tts_initialized=generator is not None and player is not None,
        error=init_error,
    )


@app.post("/tts")
async def generate_speech(request: TTSRequest) -> Response:
    """Generate a complete audio file (non-streaming)."""
    generator, player = require_models()

    audio_writer = StreamingAudioWriter(
        player,
        output_file=None,
        chunk_size=request.chunk_size,
        lookback_frames=request.lookback_frames,
    )
    audio_writer.start()

    try:
        result = generator.generate(
            request.text,
            audio_writer,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:
        logger.exception("Failed to generate speech")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        audio_writer.finalize()

    if not audio_writer.audio_chunks:
        raise HTTPException(status_code=500, detail="No audio generated")

    full_audio = np.concatenate(audio_writer.audio_chunks)

    wav_buffer = io.BytesIO()
    wav_write(wav_buffer, 22050, full_audio)
    wav_buffer.seek(0)

    logger.info(
        "Generated %d tokens in %.2fs for /tts",
        len(result["all_token_ids"]),
        result["generation_time"],
    )

    return Response(
        content=wav_buffer.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


@app.post("/stream-tts")
async def stream_speech(request: TTSRequest) -> StreamingResponse:
    """Stream audio chunks as they are generated for immediate playback."""
    generator, player = require_models()

    async def audio_chunk_generator() -> AsyncGenerator[bytes, None]:
        """Yield audio chunks as raw PCM data with a length prefix."""
        chunk_queue: queue.Queue[tuple[str, np.ndarray | str | None]] = queue.Queue()

        class ChunkList(list):
            def append(self, chunk: np.ndarray) -> None:
                super().append(chunk)
                chunk_queue.put(("chunk", chunk))

        audio_writer = StreamingAudioWriter(
            player,
            output_file=None,
            chunk_size=request.chunk_size,
            lookback_frames=request.lookback_frames,
        )
        audio_writer.audio_chunks = ChunkList()

        def generate() -> None:
            try:
                audio_writer.start()
                generator.generate(
                    request.text,
                    audio_writer,
                    max_tokens=request.max_tokens,
                )
                audio_writer.finalize()
                chunk_queue.put(("done", None))
            except Exception as exc:
                logger.exception("Generation error during stream")
                chunk_queue.put(("error", str(exc)))

        gen_thread = threading.Thread(target=generate)
        gen_thread.start()

        try:
            while True:
                msg_type, data = chunk_queue.get(timeout=30)

                if msg_type == "chunk":
                    pcm_data = (data * 32767).astype(np.int16)
                    chunk_bytes = pcm_data.tobytes()
                    length_prefix = struct.pack("<I", len(chunk_bytes))
                    yield length_prefix + chunk_bytes
                elif msg_type == "done":
                    yield struct.pack("<I", 0)
                    break
                elif msg_type == "error":
                    logger.error("Streaming generation error: %s", data)
                    yield struct.pack("<I", 0xFFFFFFFF)
                    break
        finally:
            gen_thread.join()

    return StreamingResponse(
        audio_chunk_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": "22050",
            "X-Channels": "1",
            "X-Bit-Depth": "16",
        },
    )


@app.get("/")
async def root() -> dict[str, object]:
    """Root endpoint with API info."""
    return {
        "name": "Kani TTS API",
        "version": "1.0.0",
        "endpoints": {
            "/tts": "POST - Generate complete audio",
            "/stream-tts": "POST - Stream audio chunks",
            "/health": "GET - Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Kani TTS Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
