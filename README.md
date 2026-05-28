# Code for NeurOCNN Paper

This repository contains the code for the NeurOCNN paper (Submitted to ICML 2026).

## Project Structure

- `data_loaders/`: Data loading utilities.
- `models/`: Model definitions (e.g., FNO).
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
