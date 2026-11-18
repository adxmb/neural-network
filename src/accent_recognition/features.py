""" Audio feature extraction for accent recognition

Each audio clip is characterised as a fixed-length vector consisting 
of the mean and stddev over the full clip of 20 MFCCs and their deltas
"""
import numpy as np
import librosa

SAMPLE_RATE = 16000
N_MFCC = 20
MAX_SECONDS = 15.0
MIN_SECONDS = 0.5
FEATURE_SIZE = N_MFCC * 4

def features_from_signal(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Compute the feature vector from waveform"""
    if y.size < int(MIN_SECONDS * sr):
        raise ValueError(f"Audio too short: requires minimum {MIN_SECONDS}s of audio")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    stacked = np.vstack([mfcc, delta])
    return np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)]).astype(np.float32)

def extract_features(file_path, sr=SAMPLE_RATE):
    """Load audio file and compute feature vector"""
    y, _ = librosa.load(file_path, sr=sr, mono=True, duration=MAX_SECONDS)
    return features_from_signal(y, sr)