# Speech Emotion Recognition

Developed in Google Colab.

This notebook trains a four-class speech-emotion-recognition (SER) model using
the RAVDESS and CREMA-D speech datasets. It produces predictions for **angry**,
**happy**, **neutral**, and **sad** speech.

## Requirements

Run the following in Colab before executing the notebook:

```python
!pip install resampy
!pip install -U librosa
```

The notebook also uses `pandas`, `numpy`, `matplotlib`, `scikit-learn`,
`tensorflow`, `seaborn`, and `Pillow`. These are normally available in Google
Colab; install any missing packages with `pip`.

**THE data/RAVDESS/ AND data/CREMA/ FOLDERS CONTAIN ONLY 10 SAMPLE .wav FILES EACH DUE TO GitHub FILE LIMIT CONSTRAINTS.**

**Download the entire RAVDESS dataset (1440 audio files) from https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio**

**Download the entire CREMA-D dataset (5000+ audio files) from https://www.kaggle.com/datasets/ejlok1/cremad**

## Mount Google Drive

The notebook reads data from and saves outputs to Google Drive. Mount Drive and
keep the expected folder structure:

```python
from google.colab import drive
drive.mount('drive')
```

```text
My Drive/
└── Speech Emotion Recognition/
    ├── data/
    │   ├── RAVDESS/
    │   ├── CREMA/
    │   └── RavdessCrema4Labels.csv       # created in Step 1
    ├── Feature Extraction/
    │   └── Features86Y.csv               # created in Step 5
    └── models/
        ├── ser_model.keras               # created in Step 7
        ├── ser_scaler.pkl
        └── ser_labelencoder.pkl
```

Update `path_Ravdess` and `path_Crema` if your Drive folders are located
elsewhere. The input directories should contain the original audio files; the
notebook derives labels from their RAVDESS (`-` delimited) and CREMA-D (`_`
delimited) filenames.

## 1. Combine RAVDESS & Crema data

The notebook loads both datasets into dataframes, maps their filename emotion
codes to labels, and combines the records.

To make a common balanced four-class dataset, it:

- removes `disgust`, `fearful`, and `surprised` samples;
- merges RAVDESS `calm` into `neutral`;
- downsamples every remaining class to the size of the smallest class;
- shuffles the resulting dataframe with `random_state=42`.

The balanced paths and labels are saved as:

```text
drive/My Drive/Speech Emotion Recognition/data/RavdessCrema4Labels.csv
```

---

## 2. Data loading

If Step 1 is complete, you can continue from Step 2 using the saved
`RavdessCrema4Labels.csv` file.

This section loads the CSV, drops its saved index column (`Unnamed: 0`), checks
the per-class distribution, and plots/listens to example waveforms and
spectrograms.

---

## 3. Data augmentation

The notebook includes optional augmentation helpers for adding noise, stretching
time, shifting a signal, and changing pitch. They are demonstrated on one audio
sample but are not added to the training dataframe in the current workflow.

---

## 4. Data pre-processing

Each audio file is loaded at 16 kHz, silence is trimmed using `top_db=20`, and
the signal is peak-normalised to the range `[-1, 1]`. All signals are then
right-padded with zeros to match the longest processed sample.

---

## 5. Feature extraction — 86 audio features

For every padded signal, the notebook extracts 86 mean/statistical features:

- 3 temporal features: zero-crossing rate, RMS energy, and entropy of energy;
- 12 chroma-STFT features;
- 39 MFCC features: 13 MFCCs, 13 delta MFCCs, and 13 delta-delta MFCCs;
- 20 Mel-spectrogram features;
- 12 spectral features: centroid, rolloff, bandwidth, 7 contrast values,
  flatness, and flux.

The features and class label are saved to:

```text
drive/My Drive/Speech Emotion Recognition/Feature Extraction/Features86Y.csv
```

---

## 6. Feature standardisation and splits

The 86 features are standardised with `StandardScaler`. The data is split
stratified by label into 60% training, 20% validation, and 20% test sets.
Labels are transformed with `LabelEncoder` and one-hot encoded for TensorFlow.

---

## 7. Neural network model

The model uses three input branches: a dense temporal branch, a
CNN–bidirectional-LSTM branch for the middle feature group, and a dense spectral
branch. Their outputs are concatenated and passed through a dense softmax layer
with four outputs.

Training uses Adam, categorical cross-entropy, early stopping (patience 10),
and learning-rate reduction on validation loss. The saved notebook run reports:

```text
Train accuracy:      0.750
Validation accuracy: 0.601
Test accuracy:       0.630
```

After training, run the prediction cell, to see the model's performance on test data. The notebook saves the trained model, scaler, and label encoder
to the `models/` directory.

---

## 8. Predict an emotion for one audio file

The final section selects an audio path, applies the same preprocessing and
feature extraction, loads the saved model, scaler and label encoder, and prints the predicted emotion with probabilities for angry, happy, neutral, and sad.

For a new audio file, replace `audio_path1` with its path. Keep the preprocessing,
feature extraction, feature order, scaler, and label encoder identical to those
used during training.

## Note

- Rerun Steps 1–7 when changing the datasets, feature extraction, class set, or
  preprocessing settings.
