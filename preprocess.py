import pandas as pd
import numpy as np
import librosa
import os

def read_protocol_file(protocol_path):
    data = pd.read_csv(
        protocol_path,
        sep=" ",
        names=["speaker_id", "file_name", "-", "attack_id", "label"],
    )
    print(f"Labels in {protocol_path}: {data['label'].unique()}")
    print(f"Label counts: {data['label'].value_counts()}")
    return data

def preprocess_data(split="train", max_files=None):
    base_dir = "C:/Users/hp/OneDrive/Desktop/audio_classifier/data/ASVspoof2019_LA"
    audio_dir = f"{base_dir}/ASVspoof2019_LA_{split}/flac"
    if split == "train":
        protocol_path = f"{base_dir}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    else:
        protocol_path = f"{base_dir}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{split}.trl.txt"
    
    print(f"Protocol path: {protocol_path}")
    print(f"Audio directory: {audio_dir}")
    
    if not os.path.exists(protocol_path):
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    
    data_df = read_protocol_file(protocol_path)
    X, y = [], []
    file_labels = []
    
    for idx, row in data_df.iterrows():
        if max_files and idx >= max_files:
            break
        file_path = os.path.join(audio_dir, f"{row['file_name']}.flac")
        label = 1 if row["label"] == "bonafide" else 0
        if os.path.exists(file_path):
            try:
                y_audio, sr = librosa.load(file_path, sr=16000)
                mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                if log_mel_spec.shape[1] > 128:
                    log_mel_spec = log_mel_spec[:, :128]
                else:
                    log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 128 - log_mel_spec.shape[1])), mode="constant")
                log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
                X.append(log_mel_spec[..., np.newaxis])
                y.append(label)
                file_labels.append((row['file_name'], row['label']))
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1} files")
            except Exception as e:
                print(f"Error processing {file_path} (label={row['label']}): {e}")
        else:
            print(f"File not found: {file_path} (label={row['label']})")
    
    X = np.array(X)
    y = np.array(y)
    print(f"Feature shape: {X.shape}")
    print(f"Label distribution: Bonafide={sum(y)}, Spoof={len(y) - sum(y)}")
    print(f"Unique labels: {np.unique(y)}")
    print(f"Processed files (first 10): {file_labels[:10]}")
    
    return X, y, None