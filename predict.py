"""
Plant Disease Detection — Inference Script
Predicts the disease class of a single plant-leaf image.

Usage:
    python predict.py --image <path_to_image> --model <path_to_model> --class_map <path_to_class_indices.json>

Example:
    python predict.py \
        --image leaf.jpg \
        --model output/plant_disease_model.keras \
        --class_map output/class_indices.json \
        --top_k 5
"""

import argparse
import json
import numpy as np
from pathlib import Path

import tensorflow as tf
from PIL import Image


IMG_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser(description="Plant Disease Detection — Predict")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="output/plant_disease_model.keras",
        help="Path to the saved Keras model",
    )
    parser.add_argument(
        "--class_map",
        type=str,
        default="output/class_indices.json",
        help="Path to class_indices.json produced during training",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top predictions to display",
    )
    return parser.parse_args()


def load_and_preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)


def main():
    args = parse_args()

    # Load class index → label mapping
    with open(args.class_map, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    # Load model
    print(f"Loading model from {args.model} …")
    model = tf.keras.models.load_model(args.model)

    # Preprocess image
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_array = load_and_preprocess(str(image_path))

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]  # shape: (num_classes,)

    top_k_indices = predictions.argsort()[-args.top_k:][::-1]

    print(f"\nResults for: {image_path.name}")
    print("-" * 40)
    for rank, idx in enumerate(top_k_indices, start=1):
        label = idx_to_class.get(idx, f"class_{idx}")
        confidence = predictions[idx] * 100
        print(f"  #{rank}  {label:<45}  {confidence:.2f}%")
    print("-" * 40)

    best_label = idx_to_class.get(int(top_k_indices[0]), "unknown")
    best_conf = predictions[top_k_indices[0]] * 100
    print(f"\n→ Prediction: {best_label}  ({best_conf:.2f}% confidence)\n")


if __name__ == "__main__":
    main()
