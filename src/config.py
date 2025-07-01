import os
from typing import Optional

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_MS = 4000  # 4 seconds
OVERLAP_MS = 500  # 0.5 seconds overlap
BLOCKSIZE = 64000  # sounddevice blocksize (4 seconds at 16kHz)

# VAD configuration
VAD_FRAME_MS = 10  # 10ms frames for VAD
VAD_AGGRESSIVENESS = 0  # 0-3, higher = more aggressive (0 = less aggressive)

# MLX Whisper configuration
MODEL_NAME = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
LANGUAGE = "ja"
BEAM_SIZE = 5

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Speaker Diarization configuration
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
MIN_SPEAKERS = 1
MAX_SPEAKERS = 5

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")


def get_chunk_size() -> int:
    """Get chunk size in samples"""
    return int(SAMPLE_RATE * CHUNK_MS / 1000)


def get_overlap_size() -> int:
    """Get overlap size in samples"""
    return int(SAMPLE_RATE * OVERLAP_MS / 1000)


def get_vad_frame_size() -> int:
    """Get VAD frame size in samples"""
    return int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
