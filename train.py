# train.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# -------------------- Config --------------------
DATA_DIR   = Path("ASL_kaggle")
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"

MODEL_DIR    = Path("Model")
MODEL_PATH   = MODEL_DIR / "keras_model.h5"
LABELS_PATH  = MODEL_DIR / "labels.txt"

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT  = 0.2
SEED       = 42

EPOCHS_HEAD = 15
EPOCHS_FT1  = 10
EPOCHS_FT2  = 10

LR_HEAD = 3e-4
LR_FT1  = 1e-5
LR_FT2  = 5e-6

LABEL_SMOOTHING = 0.1
L2_WEIGHT = 1e-4
# ------------------------------------------------

def write_labels(class_names, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")

def make_train_val_ds(folder: Path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=VAL_SPLIT,
        subset="training",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        validation_split=VAL_SPLIT,
        subset="validation",
        class_names=train_ds.class_names,
    )
    return train_ds, val_ds

def make_test_ds(folder: Path, class_names):
    return tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_names=class_names,
    )

def main():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError("Expected folders: ASL_kaggle/train and ASL_kaggle/test")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    train_ds, val_ds = make_train_val_ds(TRAIN_DIR)
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    test_ds = make_test_ds(TEST_DIR, class_names)

    write_labels(class_names, LABELS_PATH)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

    # Better augmentation (still realistic for hand signs)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.08),
            layers.RandomTranslation(0.04, 0.04),
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.10),
        ],
        name="augmentation",
    )

    preprocess = layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="preprocess",
    )

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base.trainable = False

    # Head with BatchNorm + L2 + Dropout (reduces confusion)
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_WEIGHT),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    loss_fn = keras.losses.SparseCategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
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

    # -------- Stage 1: Head --------
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_HEAD),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    print("\n=== Stage 1: Train classifier head ===")
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    # -------- Stage 2: Fine-tune (last blocks only) --------
    print("\n=== Stage 2: Fine-tune (last layers) ===")
    base.trainable = True

    # Unfreeze only last ~60 layers first (safer)
    fine_tune_at = len(base.layers) - 60
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= fine_tune_at

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_FT1),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    initial_epoch = len(hist1.history["loss"])
    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epoch + EPOCHS_FT1,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    # -------- Stage 3: Fine-tune deeper (optional) --------
    print("\n=== Stage 3: Fine-tune deeper (optional) ===")

    # Unfreeze last ~100 layers with even smaller LR
    fine_tune_at = len(base.layers) - 100
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= fine_tune_at

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_FT2),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    initial_epoch2 = initial_epoch + len(hist2.history["loss"])
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epoch2 + EPOCHS_FT2,
        initial_epoch=initial_epoch2,
        callbacks=callbacks,
    )

    # Evaluate best checkpoint
    best_model = keras.models.load_model(MODEL_PATH, compile=False)
    best_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=loss_fn,
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
