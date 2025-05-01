import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

SAMPLE_RATE = 16000
DURATION = 5 
N_MELS = 128
FRAME_LENGTH = 1024
HOP_LENGTH = 512
DATA_PATH = os.path.join(".", "data", "ASVspoof2019_LA")

def read_protocol_file(protocol_path):
    """Read ASVspoof protocol file and return speaker IDs, file paths, and labels."""
    data = pd.read_csv(
        protocol_path,
        sep=" ",
        names=["speaker_id", "filename", "system_id", "key"],
        usecols=["speaker_id", "filename", "system_id", "key"],
    )
    split_name = os.path.basename(protocol_path).split(".")[-3]
    data["filepath"] = data["filename"].apply(
        lambda x: os.path.join(DATA_PATH, f"ASVspoof2019_LA_{split_name}", "flac", f"{x}.flac")
    )
    return data[["speaker_id", "filepath", "key"]]

def load_audio(filepath):
    """Load audio file and normalize to fixed length."""
    try:
        audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        target_length = SAMPLE_RATE * DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
        else:
            audio = audio[:target_length]
        return audio
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_mel_spectrogram(audio):
    """Extract mel spectrogram from audio."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def preprocess_data(split="train", max_files=None):
    """Preprocess audio data for a given split (train, dev, eval)."""
    protocol_path = os.path.join(
        DATA_PATH, "ASVspoof2019_LA_cm_protocols", f"ASVspoof2019.LA.cm.{split}.trn.txt"
    )
    data_df = read_protocol_file(protocol_path)
    
    if max_files:
        data_df = data_df.sample(n=min(max_files, len(data_df)), random_state=42)
    
    audio_data = []
    labels = []
    
    for _, row in data_df.iterrows():
        audio = load_audio(row["filepath"])
        if audio is not None:
            audio_data.append(audio)
            labels.append(row["key"])

    X = np.array([extract_mel_spectrogram(audio) for audio in audio_data])
    X = X[..., np.newaxis] 

    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    if split == "train":
        with open(os.path.join(".", "models", "label_encoder.pkl"), "wb") as f:
            pickle.dump(le, f)
    
    return X, y, le

if __name__ == "__main__":
    
    X, y, le = preprocess_data(split="train", max_files=100)
    print(f"Processed {len(X)} samples with shape {X.shape}")
    print(f"Labels: {le.classes_}")