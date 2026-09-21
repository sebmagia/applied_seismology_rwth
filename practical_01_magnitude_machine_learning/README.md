# Practical 1: Magnitude prediction with machine learning

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebmagia/applied_seismology_rwth/blob/main/practical_01_magnitude_machine_learning/Mag_pred_waveform_cnn.ipynb)

A worked practical comparing linear regression, a dense ANN and a 1D CNN, followed by K-means clustering. The notebook generates its own synthetic waveforms; no extra data download is needed beyond the supplied helper code.

## Run

In Colab, open the badge and run cells from top to bottom. The first cell downloads the course repository and installs dependencies. CPU execution is supported; an available GPU speeds up the CNN. `N_SAMPLES`, `MAX_EPOCHS`, and `BATCH_SIZE` control the demonstration size. Start a fresh runtime to download later GitHub updates. Save a copy in Drive to preserve edits and download generated figures to retain them.

Locally, install dependencies with `python -m pip install -r requirements.txt` in a compatible Python environment, then open the notebook in Jupyter from this folder. Jupyter itself is a separate local installation.

## Files

- `Mag_pred_waveform_cnn.ipynb`: complete worked lesson.
- `Utility.py`: synthetic waveform generation, model definitions and plots.
- `requirements.txt`: Python dependencies.

## Changes from the supplied version

- Added Colab setup, section explanations and discussion questions; cleared old outputs.
- Split events before scaling, with separate training/validation/test subsets shared by all models. Scalers and feature-correlation analysis use training events only.
- Used distinct feature, waveform and target scalers, explicit validation data, and CNN inputs shaped `(events, samples, 1)`.
- Corrected the regression equation label to indicate standardized variables.
- Used TensorFlow's Keras namespace consistently, seeded model initialization and removed an unused Seaborn import.
- Retained the supplied model architectures, waveform physics and magnitude formulas. Results will differ from saved original results because preprocessing and splits were corrected.

Synthetic-data performance does not demonstrate real-world magnitude prediction skill. No new license is assigned to the supplied teaching material. The Colab link assumes the `main` branch.

Implementation references: [scikit-learn: avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) and [TensorFlow Conv1D input shape](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D).

## Validation

All local notebook code cells completed using 800 waveforms and a temporary two-epoch limit for each neural network, including plots, predictions and both clustering sections. Checked that the three subsets do not overlap, scalers use training events only, CNN inputs have shape `(events, 6000, 1)`, and predictions are finite. Tested with TensorFlow CPU 2.21.0 and scikit-learn 1.8.0. The delivered notebook retains 100 maximum epochs with early stopping; full convergence and the Colab-only download/install branch were not tested. Dependencies allow version ranges rather than an exact locked environment.
