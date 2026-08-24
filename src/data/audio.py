"""
Audio cache builder for UTOPYA (log-mel spectrogram, Section 3.2).

Wires the "Audio" dataset modality (11_..._Audio, segmented WAV recordings
of the plant environment) into a per-experiment fixed-size log-mel summary
that AudioEncoder (src/models/encoders.py) can consume.

Segments for one experiment (e.g. "10-59-11__SEG0000001.wav",
"10-59-52__SEG0000002.wav", ...) are concatenated in filename order (which
sorts by recording start time) into a single waveform, converted to a
log-mel spectrogram (n_mels=64, hop_length=512, per paper Section 3.2), then
average-pooled along time to a fixed number of frames so every experiment
yields the same (1, N_MELS, AUDIO_FRAMES) shape regardless of recording
length — mirroring how tabular.py/molecular.py cache one fixed-size static
feature per experiment (audio here is treated as an experiment-level
summary, not per-window, matching how GC composition and tabular features
are already applied uniformly across a window).

Uses scipy.io.wavfile (not torchaudio.load, which requires the optional
torchcodec backend and fails on this environment) to decode PCM WAV.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

AUDIO_ROOT_DIRNAME = "11_Batch_Distillation_Plant_M-202210_Audio"

N_MELS = 64
HOP_LENGTH = 512
AUDIO_FRAMES = 128   # fixed output width after average-pooling


def _iter_segment_paths(data_root: str, system: str, op: str, name: str) -> List[str]:
    exp_dir = os.path.join(data_root, AUDIO_ROOT_DIRNAME, "Operation", system, op, name)
    if not os.path.isdir(exp_dir):
        return []
    return [
        os.path.join(exp_dir, f)
        for f in sorted(os.listdir(exp_dir))
        if f.lower().endswith(".wav")
    ]


def _load_waveform(paths: List[str]) -> Optional[Tuple[np.ndarray, int]]:
    """Concatenate all segment waveforms for one experiment. Mono-mixed, float32 [-1, 1]."""
    from scipy.io import wavfile

    chunks, sr = [], None
    for p in paths:
        try:
            file_sr, data = wavfile.read(p)
        except Exception:
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        else:
            data = data.astype(np.float32)
        sr = sr or file_sr
        if file_sr != sr:
            continue  # skip segments with mismatched sample rate (rare/corrupt files)
        chunks.append(data)

    if not chunks:
        return None
    return np.concatenate(chunks), sr


def _log_mel(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Return (N_MELS, T) log-mel spectrogram, average-pooled to AUDIO_FRAMES."""
    import torchaudio

    wav_t = torch.from_numpy(waveform).unsqueeze(0)   # (1, n_samples)
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=N_MELS, hop_length=HOP_LENGTH,
    )
    mel = mel_fn(wav_t)                                # (1, N_MELS, T)
    log_mel = torch.log(mel.clamp_min(1e-8)).squeeze(0).numpy()   # (N_MELS, T)

    T = log_mel.shape[1]
    if T == 0:
        return np.zeros((N_MELS, AUDIO_FRAMES), dtype=np.float32)
    if T == AUDIO_FRAMES:
        return log_mel.astype(np.float32)
    # Average-pool (or repeat-pad) to a fixed number of frames.
    idx = np.linspace(0, T, AUDIO_FRAMES + 1).astype(int)
    pooled = np.stack([
        log_mel[:, idx[i]:max(idx[i] + 1, idx[i + 1])].mean(axis=1)
        for i in range(AUDIO_FRAMES)
    ], axis=1)
    return pooled.astype(np.float32)


def load_experiment_audio(
    data_root: str, system: str, op: str, name: str,
) -> Optional[np.ndarray]:
    """Return (1, N_MELS, AUDIO_FRAMES) log-mel summary, or None if no audio found."""
    paths = _iter_segment_paths(data_root, system, op, name)
    if not paths:
        return None
    loaded = _load_waveform(paths)
    if loaded is None:
        return None
    waveform, sr = loaded
    log_mel = _log_mel(waveform, sr)
    return log_mel[np.newaxis, :, :]   # (1, N_MELS, AUDIO_FRAMES)


def build_audio_feature_cache(
    data_root: str,
    system: str,
    experiments: List[Dict],   # dicts with keys "op" and "name"
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Build per-experiment log-mel summaries.

    Returns dict (op, name) → np.ndarray (1, N_MELS, AUDIO_FRAMES).
    Experiments with no audio recording fall back to an all-zero array
    (same convention as the tabular/molecular caches).
    """
    cache: Dict[Tuple[str, str], np.ndarray] = {}
    missing = 0
    for meta in experiments:
        op, name = meta["op"], meta["name"]
        feat = load_experiment_audio(data_root, system, op, name)
        if feat is None:
            feat = np.zeros((1, N_MELS, AUDIO_FRAMES), dtype=np.float32)
            missing += 1
        cache[(op, name)] = feat
    if missing:
        print(f"[Audio] {missing}/{len(experiments)} experiments had no usable "
              f"WAV recording — zero-filled fallback.")
    print(f"[Audio] Built log-mel cache for {len(cache)} experiments "
          f"({N_MELS}x{AUDIO_FRAMES}).")
    return cache
