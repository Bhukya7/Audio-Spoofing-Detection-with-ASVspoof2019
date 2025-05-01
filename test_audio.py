import librosa
import os

file_path = "C:/Users/hp/OneDrive/Desktop/audio_classifier/data/ASVspoof2019_LA/ASVspoof2019_LA_train/flac/LA_T_1271820.flac"
try:
    y_audio, sr = librosa.load(file_path, sr=16000)
    print("Successfully loaded spoof file")
except Exception as e:
    print(f"Error loading {file_path}: {e}")