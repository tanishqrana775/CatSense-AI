"""
Shared audio feature extraction module for MeowMind AI.
Used across both training and inference to guarantee zero train/serve skew.
"""
import io
import numpy as np
import librosa
from backend.config import SAMPLE_RATE, DURATION, TARGET_LEN, N_MELS, HOP_LENGTH, TIME_STEPS

def load_audio_waveform(source, sr=SAMPLE_RATE):
    """
    Loads audio from either a file path or in-memory bytes/buffer.
    Returns waveform y as float32 numpy array and sample rate.
    """
    if isinstance(source, (bytes, bytearray)):
        buffer = io.BytesIO(source)
        y, orig_sr = librosa.load(buffer, sr=sr, mono=True)
    elif isinstance(source, io.BytesIO):
        source.seek(0)
        y, orig_sr = librosa.load(source, sr=sr, mono=True)
    else:
        y, orig_sr = librosa.load(str(source), sr=sr, mono=True)
    return y.astype(np.float32), orig_sr

def audio_to_logmel(y, sr=SAMPLE_RATE):
    """
    Converts audio waveform y into a standardized Log-Mel Spectrogram.
    Guarantees exact return shape of (130, 64, 1).
    
    Processing Steps:
    1. Resample to SAMPLE_RATE if needed.
    2. Pad (with zeros) or trim to TARGET_LEN (66,150 samples = 3.0s).
    3. Compute Mel Spectrogram (n_mels=64, hop_length=512).
    4. Convert to Log-Mel via power_to_db (ref=np.max).
    5. Per-sample zero-mean unit-variance standardization.
    6. Shape alignment to (TIME_STEPS, N_MELS, 1) = (130, 64, 1).
    """
    # 1. Resample if necessary
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # 2. Pad or Trim to TARGET_LEN
    if len(y) < TARGET_LEN:
        padding = TARGET_LEN - len(y)
        y = np.pad(y, (0, padding), mode="constant")
    elif len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]

    # 3. Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=1024
    ) # Shape: (N_MELS, time)

    # 4. Power to dB
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # 5. Fix length to exact TIME_STEPS (130) if rounding edge case
    if log_mel.shape[1] < TIME_STEPS:
        pad_width = TIME_STEPS - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="edge")
    elif log_mel.shape[1] > TIME_STEPS:
        log_mel = log_mel[:, :TIME_STEPS]

    # 6. Per-sample standardization
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-6
    norm_log_mel = (log_mel - mean) / std

    # 7. Transpose to (Time, Frequency, Channel) -> (130, 64, 1)
    features = np.expand_dims(norm_log_mel.T, axis=-1).astype(np.float32)

    assert features.shape == (TIME_STEPS, N_MELS, 1), (
        f"Feature shape mismatch: expected ({TIME_STEPS}, {N_MELS}, 1), got {features.shape}"
    )

    return features
