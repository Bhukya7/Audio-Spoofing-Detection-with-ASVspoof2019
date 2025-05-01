import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
from preprocess import preprocess_data
import os

def evaluate_model(model_path, X, y, split_name):
    """Evaluate the model on the specified split."""
    model = tf.keras.models.load_model(model_path)
    y_pred = (model.predict(X) > 0.5).astype(int)
    print(f"\nEvaluation on {split_name} set:")
    print(classification_report(y, y_pred, target_names=["Spoof", "Bonafide"], zero_division=0))
    print("Confusion Matrix (Spoof=0, Bonafide=1):")
    print(confusion_matrix(y, y_pred))
    accuracy = np.mean(y_pred.flatten() == y)
    print(f"Accuracy: {accuracy:.4f}")
    
    y_scores = model.predict(X).ravel()
    fpr, tpr, _ = roc_curve(y, y_scores)
    eer_threshold = np.argmin(np.abs(tpr - (1 - fpr)))
    eer = 1 - tpr[eer_threshold]
    print(f"Equal Error Rate (EER): {eer:.4f}")

def main():
    model_path = os.path.join(".", "models", "audio_classifier.keras")
    X_dev, y_dev, _ = preprocess_data(split="dev", max_files=500)
    X_eval, y_eval, _ = preprocess_data(split="eval", max_files=500)
    
    evaluate_model(model_path, X_dev, y_dev, "dev")
    evaluate_model(model_path, X_eval, y_eval, "eval")

if __name__ == "__main__":
    main()