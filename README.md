# Real-Time ASL Alphabet Recognition

Webcam-based **American Sign Language (ASL) alphabet classification** using hand detection and a TensorFlow/Keras image classifier.

> This is an alphabet gesture classification demo, **not** full sign-language translation. It does not interpret grammar, sentences, facial expressions, or the temporal movements required for natural sign languages.

## What the code actually uses

| Stage | Technology |
| --- | --- |
| Image ingestion | OpenCV webcam or image directories |
| Hand localization | cvzone HandDetector (MediaPipe-based) |
| Image preprocessing | Crop, center/pad, resize and normalize |
| Model training | TensorFlow / Keras with MobileNetV2 transfer learning |
| Live inference | `scripts/test.py` |
| Evaluation | `scripts/eval_test.py`, confusion matrices and per-class summaries |

The original README mentioned PyTorch and no MediaPipe; the current scripts use **TensorFlow/Keras and cvzone/MediaPipe**.

## Installation

Recommended local environment: **Python 3.10** (the TensorFlow and MediaPipe versions below are designed for that era of Python).

```bash
python -m venv .venv
# Activate .venv for your operating system
python -m pip install -r requirements.txt
```

The first installation can be large because it includes TensorFlow and computer-vision dependencies.

## Run inference

From the repository root, with a working webcam and accessible `Model/keras_model.h5` and `Model/labels.txt`:

```bash
python scripts/test.py
```

Press `Q` or Escape to exit. The webcam example requires a graphical desktop and permission to access the camera.

## Evaluate or retrain

With labeled image folders under `ASL_kaggle/train/` and `ASL_kaggle/test/`:

```bash
python scripts/eval_test.py
python scripts/train.py
```

Training uses a MobileNetV2 backbone and writes model weights and labels to `Model/`. Evaluating produces metrics and plots in `Output/`. GPU acceleration may improve training time, but is not required by the scripts.

## Project layout

```text
ASL_kaggle/        Image dataset (train/test)
Model/             Saved Keras model and labels
scripts/train.py   Model training
scripts/test.py    Webcam inference
scripts/eval_test.py  Test-set metrics and plots
scripts/split_dataset.py  Dataset split utility
Output/            Saved example metrics/plots
```

## Limitations and responsible use

- Classification quality depends on lighting, hand pose, background, and similarity to training images.
- Dataset split, sampling, class balance and image licensing should be independently reviewed before quoting accuracy or redistributing trained weights.
- Real-time webcam classification is **not** evidence of full ASL comprehension or accessibility readiness.
- This project contains a large image dataset; future work should document its provenance/license and consider external dataset storage if allowed.

## Validation

The repository CI only checks that Python scripts compile and required files are present; it does not download dependencies, access a webcam or retrain the model.
