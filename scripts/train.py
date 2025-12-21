# train.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================================================
# Paths (ROBUST: works from any working directory)
# =========================================================
ROOT = Path(__file__).resolve().parent.parent   # <-- project root

DATA_DIR  = ROOT / "ASL_kaggle"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"

MODEL_DIR   = ROOT / "Model"
MODEL_PATH  = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"

# =========================================================
# Config
# =========================================================
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT  = 0.2
SEED       = 42

EPOCHS_HEAD = 12
EPOCHS_FT   = 10
LR_HEAD     = 1e-3
LR_FT       = 1e-5
# =========================================================


def write_labels(class_names, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(f"{c}\n")


def make_train_val_ds(folder: Path):
    """
    Training split decides class order.
    Validation is forced to follow the same order.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        folder,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
        validation_split=VAL_SPLIT,
        subset="training",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        folder,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        seed=SEED,
        validation_split=VAL_SPLIT,
        subset="validation",
        class_names=train_ds.class_names,  # IMPORTANT
    )
    return train_ds, val_ds


def make_test_ds(folder: Path, class_names):
    return tf.keras.utils.image_dataset_from_directory(
        folder,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        class_names=class_names,  # ensure same mapping
    )


def main():
    # -----------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------
    print("RUNNING FILE :", Path(__file__).resolve())
    print("WORKDIR      :", Path.cwd().resolve())
    print("TRAIN_DIR    :", TRAIN_DIR.resolve())
    print("TEST_DIR     :", TEST_DIR.resolve())

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {TEST_DIR}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------
    train_ds, val_ds = make_train_val_ds(TRAIN_DIR)
    class_names = train_ds.class_names
    num_classes = len(class_names)

    print("Classes:", class_names)

    test_ds = make_test_ds(TEST_DIR, class_names)

    write_labels(class_names, LABELS_PATH)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

    # -----------------------------------------------------
    # Data augmentation
    # -----------------------------------------------------
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="augmentation",
    )

    preprocess = layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="preprocess",
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
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


    # -----------------------------------------------------
    # Stage 1: Train classifier head
    # -----------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(LR_HEAD),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("\n=== Stage 1: Train classifier head ===")
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    # -----------------------------------------------------
    # Stage 2: Fine-tuning
    # -----------------------------------------------------
    print("\n=== Stage 2: Fine-tune MobileNetV2 ===")
    base.trainable = True

    fine_tune_at = 100
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= fine_tune_at

    model.compile(
        optimizer=keras.optimizers.Adam(LR_FT),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD + EPOCHS_FT,
        initial_epoch=len(hist1.history["loss"]),
        callbacks=callbacks,
    )

    # -----------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------
    best_model = keras.models.load_model(MODEL_PATH, compile=False)
    best_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("\n=== Test Evaluation ===")
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
    print(f"Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")

    print("\nSaved:")
    print(" -", MODEL_PATH)
    print(" -", LABELS_PATH)


if __name__ == "__main__":
    main()
