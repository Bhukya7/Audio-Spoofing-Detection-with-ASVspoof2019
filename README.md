# Audio Spoofing Detection with ASVspoof2019

This project implements an audio spoofing detection system using the ASVspoof2019 Logical Access (LA) dataset. A Convolutional Neural Network (CNN) is trained to classify audio samples as `bonafide` (genuine) or `spoof` (fake) based on mel-spectrogram features. The model achieves 99% accuracy on a balanced development set, with excellent precision, recall, and F1-scores.

## Project Overview

The goal is to detect spoofed audio using the ASVspoof2019 dataset, which contains genuine and spoofed audio samples. The system preprocesses audio files to extract mel-spectrograms, trains a CNN with regularization and data augmentation, and evaluates performance using classification metrics and a confusion matrix. A testing script is included to classify individual audio files.

### Workflow Diagram

```mermaid
graph TD
    A[ASVspoof2019 Data] --> B[Load & Preprocess<br>• Mel-spectrogram<br>• Resize 128x128]
    B --> C[Train CNN<br>• 3 Conv layers<br>• Dropout 0.6]
    C --> D[Evaluate<br>• Precision/Recall<br>• Confusion matrix]
    C --> E[Inference]
    D --> F[Outputs]
    E --> F

    style A fill:#e1f5fe,stroke:#039be5,color:black
    style F fill:#e1f5fe,stroke:#039be5,color:black
    style B fill:#FFF9C4,stroke:#FFD600,color:black
    style C fill:#C8E6C9,stroke:#43A047,color:black
    style D fill:#FFCDD2,stroke:#E53935,color:black
    style E fill:#BBDEFB,stroke:#1E88E5,color:black
```

#### Features
- **Data Preprocessing**: Extracts 128x128 mel-spectrograms with normalization and augmentation (time stretching, pitch shifting, noise addition).
- **Model Architecture**: CNN with convolutional layers, batch normalization, dropout, and L2 regularization.
- **Training**: Uses early stopping and learning rate scheduling for optimal convergence.
- **Evaluation**: Reports precision, recall, F1-score, and confusion matrix for development and test sets.
- **Testing**: Supports classification of individual audio files using the trained model.

## Results (Development Set)
- **Development Set (1000 samples, 500 spoof, 500 bonafide)**:

| Metric          | Spoof | Bonafide | Overall |
|-----------------|-------|----------|---------|
| Accuracy        | -     | -        | 99%     |
| Precision       | 99%   | 100%     | -       |
| Recall          | 100%  | 99%      | -       |
| F1-Score        | 99%   | 99%      | -       |

### Confusion Matrix (500 spoof, 500 bonafide):

|                | Predicted Spoof | Predicted Bonafide |
|----------------|-----------------|--------------------|
| **Actual Spoof** | 498             | 2                  |
| **Actual Bonafide** | 5               | 495                |



