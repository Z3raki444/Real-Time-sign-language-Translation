# train.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------- Config --------------------
DATA_ROOT = Path("ASL_Alphabet")

TRAIN_DIR = DATA_ROOT / "asl_alphabet_train"
TEST_DIR  = DATA_ROOT / "asl_alphabet_test"

MODEL_DIR  = Path("Model")
MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2

EPOCHS_HEAD = 12
EPOCHS_FT = 10
LR_HEAD = 1e-3
LR_FT = 1e-5

SEED = 42
# ------------------------------------------------


def write_labels(class_names, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")


def extract_label_from_filename(path: Path) -> str:
    # Example: A_train.jpg -> A, A_test.jpg -> A
    name = path.stem  # A_train
    if "_train" in name:
        return name.split("_train")[0]
    if "_test" in name:
        return name.split("_test")[0]
    # fallback: take text before first underscore
    return name.split("_")[0]


def list_images(folder: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for e in exts:
        files.extend(folder.glob(e))
    return sorted(files)


def decode_resize(path, label, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.08)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

    # MobileNetV2 preprocessing: scales to [-1, 1]
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label


def make_dataset(file_paths, labels, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda p, y: decode_resize(p, y, augment=True),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda p, y: decode_resize(p, y, augment=False),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError("Expected folders: asl_alphabet_train/ and asl_alphabet_test/")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # ---- Collect files ----
    train_files = list_images(TRAIN_DIR)
    test_files  = list_images(TEST_DIR)

    if len(train_files) == 0 or len(test_files) == 0:
        raise ValueError("No images found. Check folder names and image extensions.")

    # ---- Build labels from filenames ----
    train_labels_str = [extract_label_from_filename(p) for p in train_files]
    test_labels_str  = [extract_label_from_filename(p) for p in test_files]

    # Use labels from TRAIN to define class order (sorted)
    class_names = sorted(list(set(train_labels_str)))
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Num classes:", num_classes)
    write_labels(class_names, LABELS_PATH)

    # Map label string -> int
    label_to_index = {name: i for i, name in enumerate(class_names)}
    train_labels = np.array([label_to_index[x] for x in train_labels_str], dtype=np.int32)

    # If test contains unexpected labels, this will fail loudly (good)
    test_labels = np.array([label_to_index[x] for x in test_labels_str], dtype=np.int32)

    # Convert file paths to strings for tf.data
    train_files = np.array([str(p) for p in train_files])
    test_files  = np.array([str(p) for p in test_files])

    # ---- Split train into train/val ----
    idx = np.arange(len(train_files))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)

    val_size = int(len(idx) * VAL_SPLIT)
    val_idx = idx[:val_size]
    tr_idx  = idx[val_size:]

    tr_files, tr_labels = train_files[tr_idx], train_labels[tr_idx]
    val_files, val_labels = train_files[val_idx], train_labels[val_idx]

    train_ds = make_dataset(tr_files, tr_labels, training=True)
    val_ds   = make_dataset(val_files, val_labels, training=False)
    test_ds  = make_dataset(test_files, test_labels, training=False)

    # ---- Model ----
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_HEAD),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\n=== Stage 1: Train classifier head ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    print("\n=== Stage 2: Fine-tune MobileNetV2 ===")
    base.trainable = True
    fine_tune_at = 100
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= fine_tune_at

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_FT),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD + EPOCHS_FT,
        initial_epoch=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    best_model = keras.models.load_model(MODEL_PATH, compile=False)
    best_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("\n=== Test Evaluation (Best Saved Model) ===")
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
    print(f"Test Accuracy: {test_acc*100:.2f}%  |  Test Loss: {test_loss:.4f}")

    print("\nSaved:")
    print(" -", MODEL_PATH)
    print(" -", LABELS_PATH)


if __name__ == "__main__":
    main()
