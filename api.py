from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import tensorflow as tf
import pickle
from pydantic import BaseModel
import uvicorn
import io
import os

app = FastAPI(title="Audio Authenticity Classifier")

model = tf.keras.models.load_model(os.path.join("..", "models", "audio_classifier.h5"))
with open(os.path.join("..", "models", "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
FRAME_LENGTH = 1024
HOP_LENGTH = 512

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

def preprocess_audio(file_content):
    """Preprocess uploaded audio file to extract mel spectrogram."""
    try:
        audio, _ = librosa.load(io.BytesIO(file_content), sr=SAMPLE_RATE, duration=DURATION, mono=True)
        target_length = SAMPLE_RATE * DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
        else:
            audio = audio[:target_length]
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = mel_spec_db[np.newaxis, ..., np.newaxis]
        return mel_spec_db
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict if audio is real or fake."""
    if not file.filename.endswith(".flac"):
        raise HTTPException(status_code=400, detail="Only FLAC files are supported")
    
    content = await file.read()
    processed_audio = preprocess_audio(content)
    
    prediction = model.predict(processed_audio, verbose=0)
    confidence = float(prediction[0][0])
    label = label_encoder.inverse_transform([int(confidence > 0.5)])[0]
    
    return PredictionResponse(
        prediction=label,
        confidence=confidence if label == "bonafide" else 1 - confidence
    )

@app.get("/")
async def root():
    return {"message": "Audio Authenticity Classifier API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)