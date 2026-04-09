"""
Plant Disease Detection - Training Script
Dataset: PlantVillage (https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

Usage:
    python train.py --data_dir <path_to_PlantVillage> [options]

Example:
    python train.py --data_dir ./PlantVillage --epochs 20 --batch_size 32
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 224          # EfficientNetB0 default input size
SEED = 42


# ── Argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train Plant Disease Detection Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="PlantVillage",
        help="Path to the PlantVillage dataset directory (contains class sub-folders)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Initial learning rate"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of data used for validation (0–1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for saved model, plots, and logs",
    )
    parser.add_argument(
        "--fine_tune_at",
        type=int,
        default=100,
        help="Unfreeze base-model layers from this index onward for fine-tuning",
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=10,
        help="Extra epochs for the fine-tuning phase",
    )
    return parser.parse_args()


# ── Data preparation ──────────────────────────────────────────────────────────
def build_data_generators(data_dir: str, val_split: float, batch_size: int):
    """
    Creates train / validation ImageDataGenerators with augmentation on the
    training set and simple rescaling on the validation set.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
    )

    common_kwargs = dict(
        directory=data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        seed=SEED,
    )

    train_gen = train_datagen.flow_from_directory(
        subset="training", shuffle=True, **common_kwargs
    )
    val_gen = val_datagen.flow_from_directory(
        subset="validation", shuffle=False, **common_kwargs
    )

    return train_gen, val_gen


# ── Model definition ──────────────────────────────────────────────────────────
def build_model(num_classes: int, learning_rate: float) -> tuple:
    """
    Builds a transfer-learning model using EfficientNetB0 as the feature
    extractor followed by a custom classification head.
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False   # freeze during initial training

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="PlantDiseaseNet")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks(output_dir: str, phase: str = "phase1"):
    checkpoint_path = os.path.join(output_dir, f"best_model_{phase}.keras")
    return [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(os.path.join(output_dir, f"training_log_{phase}.csv")),
    ]


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_history(history, output_dir: str, suffix: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(output_dir, f"training_curves{suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves → {path}")


def plot_confusion_matrix(cm, class_names, output_dir: str):
    fig, ax = plt.subplots(figsize=(max(12, len(class_names) // 2),
                                    max(10, len(class_names) // 2)))
    sns.heatmap(
        cm,
        annot=len(class_names) <= 40,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {path}")


# ── Main training loop ────────────────────────────────────────────────────────
def main():
    args = parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Plant Disease Detection — Training")
    print("=" * 60)
    print(f"  Data directory : {args.data_dir}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Epochs (phase1): {args.epochs}")
    print(f"  Fine-tune epochs: {args.fine_tune_epochs}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Learning rate  : {args.learning_rate}")
    print("=" * 60 + "\n")

    # ── 1. Build data generators ───────────────────────────────────────────
    print("Loading dataset …")
    train_gen, val_gen = build_data_generators(
        args.data_dir, args.val_split, args.batch_size
    )

    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())

    print(f"  Classes found  : {num_classes}")
    print(f"  Training images: {train_gen.samples}")
    print(f"  Val images     : {val_gen.samples}\n")

    # Save class index mapping
    class_map_path = os.path.join(args.output_dir, "class_indices.json")
    with open(class_map_path, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"Class indices saved → {class_map_path}\n")

    # ── 2. Build model ─────────────────────────────────────────────────────
    model, base_model = build_model(num_classes, args.learning_rate)
    model.summary()

    # ── 3. Phase 1 — train classification head (base frozen) ──────────────
    print("\n[Phase 1] Training classification head …\n")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=get_callbacks(args.output_dir, phase="phase1"),
    )
    plot_history(history1, args.output_dir, suffix="_phase1")

    # ── 4. Phase 2 — fine-tune top layers of base model ───────────────────
    print(f"\n[Phase 2] Fine-tuning from layer {args.fine_tune_at} onward …\n")
    base_model.trainable = True
    for layer in base_model.layers[: args.fine_tune_at]:
        layer.trainable = False

    # Lower LR for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.fine_tune_epochs,
        callbacks=get_callbacks(args.output_dir, phase="phase2"),
    )
    plot_history(history2, args.output_dir, suffix="_phase2")

    # ── 5. Evaluate on validation set ─────────────────────────────────────
    print("\n[Evaluation] Computing metrics on validation set …\n")
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved → {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, args.output_dir)

    # ── 6. Save final model ────────────────────────────────────────────────
    final_model_path = os.path.join(args.output_dir, "plant_disease_model.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved → {final_model_path}")

    # Also export as SavedModel for TF Serving compatibility
    saved_model_path = os.path.join(args.output_dir, "plant_disease_savedmodel")
    tf.saved_model.save(model, saved_model_path)
    print(f"SavedModel exported → {saved_model_path}")

    print("\n✅  Training complete!\n")


if __name__ == "__main__":
    main()
