# eval_test.py
import os, csv, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ---------- Robust Paths ----------
ROOT = Path(__file__).resolve().parent.parent   # <-- project root

DATA_ROOT = ROOT / "ASL_kaggle"
TEST_DIR  = DATA_ROOT / "test"

MODEL_DIR   = ROOT / "Model"
MODEL_PATH  = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"

OUTPUT_DIR = ROOT / "Output"
OUTPUT_DIR.mkdir(exist_ok=True)


CSV_OUT     = OUTPUT_DIR / "eval_results.csv"
CSV_PERCLS  = OUTPUT_DIR / "per_class_metrics.csv"
RESULTS_TXT = OUTPUT_DIR / "RESULTS.txt"

# Plots (PNG only)
CM_PNG         = OUTPUT_DIR / "confusion_matrix.png"
TP_TN_PNG      = OUTPUT_DIR / "tp_fp_fn_tn.png"
CM_NORM_PNG    = OUTPUT_DIR / "confusion_matrix_normalized.png"
ACC_BAR_PNG    = OUTPUT_DIR / "per_class_accuracy.png"
SUPPORT_PNG    = OUTPUT_DIR / "support_per_class.png"
CONF_HIST_PNG  = OUTPUT_DIR / "confidence_histogram.png"

BATCH = 64
# -----------------------------------------------

def load_labels(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def normalize_label(name: str) -> str:
    if name.endswith("_test"):
        return name[:-5]
    if name.endswith("_train"):
        return name[:-6]
    return name

def is_folder_per_class(root: Path) -> bool:
    return root.exists() and any(p.is_dir() for p in root.iterdir())

def list_images_recursive(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def make_test_ds_folder_per_class(root: Path, img_size: int, class_names: list):
    ds = tf.keras.utils.image_dataset_from_directory(
        root,
        image_size=(img_size, img_size),
        batch_size=BATCH,
        label_mode="int",
        shuffle=False,
        class_names=class_names,
    )
    return ds.prefetch(tf.data.AUTOTUNE)

def make_test_ds_flat(root: Path, labels: list, img_size: int):
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    files = list_images_recursive(root)

    filepaths, y_true = [], []
    dropped = 0

    for p in files:
        lab = normalize_label(p.stem)
        if lab in label_to_idx:
            filepaths.append(str(p))
            y_true.append(label_to_idx[lab])
        else:
            dropped += 1

    if len(filepaths) == 0:
        raise ValueError("No usable test images found (after label filtering).")

    if dropped > 0:
        print(f"⚠️ Dropped {dropped} test images (labels not found in labels.txt).")

    ds = tf.data.Dataset.from_tensor_slices((filepaths, y_true))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32)
        return img, tf.cast(label, tf.int32)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds, filepaths

# ------------------ Plots ------------------

def plot_confusion_matrix(cm, labels, out_path: Path):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

def plot_confusion_matrix_normalized(cm, labels, out_path: Path):
    cm = cm.astype(np.float32)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0) * 100.0

    plt.figure(figsize=(10, 10))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title("Confusion Matrix (Normalized, %)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

def plot_tp_fp_fn_tn(cm, labels, out_path: Path):
    total = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - (tp + fp + fn)

    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(14, 6))
    plt.bar(x - 1.5 * width, tp, width, label="TP")
    plt.bar(x - 0.5 * width, fp, width, label="FP")
    plt.bar(x + 0.5 * width, fn, width, label="FN")
    plt.bar(x + 1.5 * width, tn, width, label="TN")
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Count")
    plt.title("Per-Class TP / FP / FN / TN (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

def plot_per_class_accuracy(per_class_acc, labels, out_path: Path):
    acc_pct = per_class_acc * 100.0
    x = np.arange(len(labels))

    plt.figure(figsize=(14, 6))
    plt.bar(x, acc_pct)
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

def plot_support(support, labels, out_path: Path):
    x = np.arange(len(labels))

    plt.figure(figsize=(14, 6))
    plt.bar(x, support)
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Samples (n)")
    plt.title("Test Samples per Class (Support)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

def plot_confidence_histogram(y_conf, out_path: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(y_conf, bins=30)
    plt.xlabel("Prediction Confidence (max probability)")
    plt.ylabel("Number of samples")
    plt.title("Confidence Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

# ------------------ Main ------------------

def main():
    t_total0 = time.perf_counter()

    print("RUNNING FILE :", Path(__file__).resolve())
    print("WORKDIR      :", Path.cwd().resolve())
    print("TEST_DIR     :", TEST_DIR.resolve())

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if not LABELS_PATH.exists():
        raise FileNotFoundError(LABELS_PATH)
    if not TEST_DIR.exists():
        raise FileNotFoundError(TEST_DIR)

    model  = keras.models.load_model(str(MODEL_PATH), compile=False)
    labels = load_labels(LABELS_PATH)
    num_classes = len(labels)

    in_shape = model.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    IMG = int(in_shape[1])

    HAS_INTERNAL_PREPROC = any(getattr(l, "name", "") == "preprocess" for l in model.layers)

    print("Detected test format:", "FOLDER-PER-CLASS" if is_folder_per_class(TEST_DIR) else "FLAT (a_test.jpg)")

    # Dataset
    if is_folder_per_class(TEST_DIR):
        test_ds = make_test_ds_folder_per_class(TEST_DIR, IMG, labels)
        filepaths = []
        for cls in labels:
            cls_dir = TEST_DIR / cls
            if cls_dir.exists():
                imgs = sorted([p for p in cls_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
                filepaths += [str(p) for p in imgs]
    else:
        test_ds, filepaths = make_test_ds_flat(TEST_DIR, labels, IMG)

    # Warm-up (not timed)
    for batch_imgs, _ in test_ds.take(1):
        if HAS_INTERNAL_PREPROC:
            _ = model.predict(batch_imgs, verbose=0)
        else:
            arr = (batch_imgs.numpy().astype(np.float32) / 127.5) - 1.0
            _ = model.predict(arr, verbose=0)

    # Predict timing
    t_pred0 = time.perf_counter()

    y_true_all, y_pred_all, y_conf_all = [], [], []
    total_images = 0

    for batch_imgs, batch_labels in test_ds:
        bs = int(batch_imgs.shape[0])
        total_images += bs

        if HAS_INTERNAL_PREPROC:
            probs = model.predict(batch_imgs, verbose=0)
        else:
            arr = (batch_imgs.numpy().astype(np.float32) / 127.5) - 1.0
            probs = model.predict(arr, verbose=0)

        preds = np.argmax(probs, axis=1)
        confs = probs[np.arange(len(probs)), preds]

        y_true_all.append(batch_labels.numpy())
        y_pred_all.append(preds)
        y_conf_all.append(confs)

    t_pred1 = time.perf_counter()

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_conf = np.concatenate(y_conf_all)

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.where(support > 0, np.diag(cm) / support, 0.0)

    overall_acc = float((y_true == y_pred).mean())

    valid = support > 0
    acc_valid = per_class_acc.copy()
    acc_valid[~valid] = np.nan
    best_val = np.nanmax(acc_valid)
    worst_val = np.nanmin(acc_valid)
    best_idxs = np.where(np.isclose(acc_valid, best_val))[0]
    worst_idxs = np.where(np.isclose(acc_valid, worst_val))[0]

    # Plots
    plot_confusion_matrix(cm, labels, CM_PNG)
    plot_tp_fp_fn_tn(cm, labels, TP_TN_PNG)
    plot_confusion_matrix_normalized(cm, labels, CM_NORM_PNG)
    plot_per_class_accuracy(per_class_acc, labels, ACC_BAR_PNG)
    plot_support(support, labels, SUPPORT_PNG)
    plot_confidence_histogram(y_conf, CONF_HIST_PNG)

    # CSVs
    n = min(len(filepaths), len(y_true))
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "true_label", "pred_label", "confidence"])
        for i in range(n):
            w.writerow([filepaths[i], labels[int(y_true[i])], labels[int(y_pred[i])], f"{float(y_conf[i]):.6f}"])

    with open(CSV_PERCLS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "support", "accuracy_percent"])
        for i in range(num_classes):
            w.writerow([labels[i], int(support[i]), f"{per_class_acc[i]*100:.2f}"])

    # Timing
    pred_seconds = t_pred1 - t_pred0
    total_seconds = time.perf_counter() - t_total0
    sec_per_img = pred_seconds / max(1, total_images)
    fps = 1.0 / sec_per_img if sec_per_img > 0 else 0.0

    # RESULTS.txt
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write("REAL-TIME SIGN LANGUAGE RECOGNITION – OVERALL RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model file          : {MODEL_PATH.name}\n")
        f.write(f"Input image size    : {IMG} x {IMG}\n")
        f.write(f"Number of classes   : {num_classes}\n")
        f.write(f"Total test samples  : {len(y_true)}\n\n")
        f.write(f"Overall Accuracy    : {overall_acc*100:.2f} %\n\n")

        f.write("Timing (Evaluation)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total runtime            : {total_seconds:.3f} seconds\n")
        f.write(f"Inference time (predict) : {pred_seconds:.3f} seconds\n")
        f.write(f"Avg time per image       : {sec_per_img*1000:.3f} ms\n")
        f.write(f"Approx throughput        : {fps:.2f} FPS\n\n")

        f.write("Per-Class Accuracy\n")
        f.write("-" * 30 + "\n")
        for i in range(num_classes):
            f.write(f"{labels[i]:<10s} : {per_class_acc[i]*100:6.2f}% (n={support[i]})\n")

        f.write("\nBest Performing Class(es)\n")
        f.write(", ".join([f"{labels[i]} ({per_class_acc[i]*100:.2f}%)" for i in best_idxs]) + "\n")

        f.write("\nWorst Performing Class(es)\n")
        f.write(", ".join([f"{labels[i]} ({per_class_acc[i]*100:.2f}%)" for i in worst_idxs]) + "\n")

        f.write("\nSaved Plots (PNG)\n")
        f.write(f"- {CM_PNG.name}\n")
        f.write(f"- {TP_TN_PNG.name}\n")
        f.write(f"- {CM_NORM_PNG.name}\n")
        f.write(f"- {ACC_BAR_PNG.name}\n")
        f.write(f"- {SUPPORT_PNG.name}\n")
        f.write(f"- {CONF_HIST_PNG.name}\n")

    print("\n✅ Evaluation completed")
    print(f"📁 Output saved in: {OUTPUT_DIR.resolve()}")
    print(f"⏱️ Total runtime: {total_seconds:.3f}s | Predict: {pred_seconds:.3f}s | {sec_per_img*1000:.3f} ms/img (~{fps:.2f} FPS)")


if __name__ == "__main__":
    main()
