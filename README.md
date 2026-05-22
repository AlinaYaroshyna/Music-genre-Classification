# Music Genre Classification

This repository contains a simple music genre classification pipeline based on audio feature extraction and machine learning classifiers.

## Repository Structure

- `exstraction.py`  
  Extracts audio features from `.wav` music files stored in the `data/` directory.

- `classification.py`  
  Trains and evaluates three classifiers using Repeated Stratified K-Fold cross-validation.  
  Model performance is measured using balanced accuracy score.

- `visualization.py`  
  Loads classification results and generates charts and visualizations.

---

## Applied Algorithms

### Feature Extractors

- MFCC (Mel-frequency cepstral coefficients)
- Chroma features
- Spectral centroid
- Zero-crossing rate

### Classifiers

- SVM
- MLPClassifier
- K-Nearest Neighbors (KNN)

---

## How to Use

The repository already contains extracted features in `features_labels.npz`, so you can run `classification.py` directly.

Additional saved outputs include:

- `classification_results.npz`
- `confusion_matrices.npz`
- Classification reports stored in the `reports/` directory

If you want to generate features from raw audio files again, extract the `data.zip` and run:

```bash
python exstraction.py
```

Then run classification:

```bash
python classification.py
```

To generate visualizations:

```bash
python visualization.py
```
```bash
python visualization.py
```
