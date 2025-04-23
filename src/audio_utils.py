import librosa
import numpy as np
import soundfile as sf


def read_audio(audio_path, fs=16000, trim=False, int_type=False):

    X, fs_orig = sf.read(audio_path)
    if fs_orig != fs:
        X = librosa.resample(X, orig_sr=fs_orig, target_sr=fs)

    if trim:
        X = librosa.effects.trim(X, top_db=20)[0]
    # from float to int
    if int_type:
        X = (X * 32768).astype(np.int32)

    return X, fs
