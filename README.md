# [ICML 2026] NeurOCNN: A Neural-Operator-Based Model for Physiological Time Series

NeurOCNN is a neural-operator-inspired architecture for physiological time-series classification under sampling-rate shifts. The model combines spline-parameterized continuous-time convolutions for local morphology extraction with a Fourier projection/pooling interface for fixed-dimensional physical-time representation learning. This design allows the model to be trained at one sampling rate and evaluated zero-shot at unseen sampling rates without redefining the architecture.

## Project Structure

- `data_loaders/`: Data loading utilities.
- `models/`: Model definitions (e.g., NeurOCNN, FNO etc.).
- `preprocessed_data/`: Directory for preprocessed data.
- `training/`: Training utilities and loops.
- `utils/`: General utility functions.
- `train.py`: Main training script.
- `nfold_cv.py`: Script for N-fold cross-validation.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Preprocess the data using the `preprocessing.ipynb` notebook.

## Usage

To train the model, run:

```bash
python train.py
```

To run N-fold cross-validation:

```bash
python nfold_cv.py
```
