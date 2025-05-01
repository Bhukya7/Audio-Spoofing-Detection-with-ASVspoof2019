import pandas as pd
import numpy as np
import os
import librosa
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def process_audio(file_path, sr=16000, n_mels=128, duration=4.0):
    
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    target_length = int(sr * duration)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)
    
    spectrogram = tf.image.resize(spectrogram[..., np.newaxis], [128, 128]).numpy()
    return spectrogram

def plot_spectrogram(spectrogram, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram.squeeze(), aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar()
    plt.show()

def load_data(protocol_file, audio_dir, max_files_per_class=1000):
    protocol_data = pd.read_csv(protocol_file, sep=" ", names=["speaker_id", "file_id", "col3", "col4", "label"])
    bonafide_data = protocol_data[protocol_data["label"] == "bonafide"]
    spoof_data = protocol_data[protocol_data["label"] == "spoof"]
    bonafide_sample = bonafide_data.sample(n=max_files_per_class, random_state=42)
    spoof_sample = spoof_data.sample(n=max_files_per_class, random_state=42)
    combined_data = pd.concat([bonafide_sample, spoof_sample])
    combined_data = shuffle(combined_data, random_state=42)
    
    features = []
    labels = []
    processed_files = []
    
    for i, row in enumerate(combined_data.iterrows()):
        file_id = row[1]["file_id"]
        label = row[1]["label"]
        audio_path = os.path.join(audio_dir, f"{file_id}.flac")
        if os.path.exists(audio_path):
            feature = process_audio(audio_path)
            features.append(feature)
            labels.append(1 if label == "bonafide" else 0)
            processed_files.append((file_id, label))
            if i % 100 == 0:
                print(f"Processed {i} files")
            if i < 2:  
                plot_spectrogram(feature, f"File: {file_id}, Label: {label}")
        else:
            print(f"File not found: {audio_path}")
    
    return np.array(features), np.array(labels), processed_files

def main():
   
    train_protocol = "C:/Users/hp/OneDrive/Desktop/audio_classifier/data/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    train_audio_dir = "C:/Users/hp/OneDrive/Desktop/audio_classifier/data/ASVspoof2019_LA/ASVspoof2019_LA_train/flac"
    dev_protocol = "C:/Users/hp/OneDrive/Desktop/audio_classifier/data/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    dev_audio_dir = "C:/Users/hp/OneDrive/Desktop/audio_classifier/data/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac"
    
    X_train, y_train, train_files = load_data(train_protocol, train_audio_dir, max_files_per_class=1000)
    X_dev, y_dev, dev_files = load_data(dev_protocol, dev_audio_dir, max_files_per_class=500)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_dev shape: {X_dev.shape}")
    print(f"Label distribution (train): Bonafide={sum(y_train)}, Spoof={len(y_train) - sum(y_train)}")
    print(f"Label distribution (dev): Bonafide={sum(y_dev)}, Spoof={len(y_dev) - sum(y_dev)}")
    print("First 10 training labels:", y_train[:10])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_dev, y_dev),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, lr_scheduler]
    )
    
    y_pred = (model.predict(X_dev) > 0.5).astype(int)
    print("Classification Report:")
    print(classification_report(y_dev, y_pred, target_names=['spoof', 'bonafide']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_dev, y_pred))

if __name__ == "__main__":
    main()