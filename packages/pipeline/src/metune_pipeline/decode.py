import subprocess
from pathlib import Path

import numpy as np

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac", ".webm"}


class DecodeError(Exception):
    """Unreadable / unsupported audio. Not retryable. str() is shown to users; `detail` (ffmpeg output, server
    paths) is for logs only."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


def decode(src: Path, sr: int = 16000) -> np.ndarray:
    """Any ffmpeg-readable file → mono float32 at `sr`. Stereo is averaged (calls are fake stereo)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(src), "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise DecodeError("Could not decode audio: the file is corrupt or not a supported audio format.",
                          detail=proc.stderr.decode(errors="replace").strip()[-300:] or "ffmpeg failed")
    x = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if len(x) < sr // 2:
        raise DecodeError(f"Audio too short ({len(x) / sr:.2f} s); at least 0.5 s is needed.")
    return x


def encode_flac(x: np.ndarray, sr: int = 16000) -> bytes:
    """Lossless and ~2x smaller than WAV → smaller LLM upload (inline limit ~20 MB)."""
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-", "-f", "flac", "-"],
        input=pcm, capture_output=True, check=True,
    )
    return proc.stdout
