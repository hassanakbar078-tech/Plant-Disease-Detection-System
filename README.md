# Plant Disease Detection System

A deep-learning pipeline that classifies plant leaf images into **38 disease / healthy categories** using transfer learning on the [PlantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).

---

## Repository Structure

```
.
├── train.py               # Main training script
├── predict.py             # Single-image inference script
├── download_dataset.py    # Helper to download the dataset via Kaggle API
├── requirements.txt       # Python dependencies
└── output/                # Created automatically during training
    ├── plant_disease_model.keras
    ├── plant_disease_savedmodel/
    ├── class_indices.json
    ├── training_curves_phase1.png
    ├── training_curves_phase2.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/hassanakbar078-tech/Plant-Disease-Detection-System.git
cd Plant-Disease-Detection-System
pip install -r requirements.txt
```

### 2. Download the PlantVillage dataset

**Option A — Automated download via Kaggle API**

1. Create a free account at [kaggle.com](https://www.kaggle.com).
2. Go to **Settings → API → Create New Token** — this downloads `kaggle.json`.
3. Place `kaggle.json` at `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
4. Run:

```bash
python download_dataset.py
```

**Option B — Manual download**

1. Visit: <https://www.kaggle.com/datasets/arjuntejaswi/plant-village>
2. Click **Download** and extract the zip.
3. Make sure the extracted folder is named `PlantVillage` and placed in the project root.  
   It should contain one sub-folder per class, e.g. `Tomato___Bacterial_spot/`, `Potato___healthy/`, etc.

---

## Training

```bash
python train.py --data_dir PlantVillage
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `PlantVillage` | Path to the dataset root |
| `--epochs` | `20` | Epochs for phase-1 (frozen base) |
| `--fine_tune_epochs` | `10` | Extra epochs for phase-2 (fine-tuning) |
| `--batch_size` | `32` | Batch size |
| `--learning_rate` | `0.001` | Initial learning rate |
| `--val_split` | `0.15` | Fraction of data for validation |
| `--fine_tune_at` | `100` | Unfreeze EfficientNetB0 from this layer index |
| `--output_dir` | `output` | Where to save model and artifacts |

### Training approach

The model uses **EfficientNetB0** (pre-trained on ImageNet) as the backbone:

1. **Phase 1** — The base model is frozen. Only the custom classification head is trained.
2. **Phase 2** — The top layers of EfficientNetB0 are unfrozen and the entire network is fine-tuned at a lower learning rate.

Data augmentation (rotation, flips, zoom, shifts) is applied to the training set to improve generalisation.

---

## Inference

```bash
python predict.py \
    --image path/to/leaf.jpg \
    --model output/plant_disease_model.keras \
    --class_map output/class_indices.json \
    --top_k 5
```

Example output:
```
Results for: leaf.jpg
----------------------------------------
  #1  Tomato___Early_blight                          92.34%
  #2  Tomato___Septoria_leaf_spot                     4.21%
  #3  Tomato___healthy                                1.88%
----------------------------------------

→ Prediction: Tomato___Early_blight  (92.34% confidence)
```

---

## Dataset

- **Source**: [PlantVillage — Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- **Classes**: 38 (plant × disease combinations, including healthy)
- **Images**: ~87,000 RGB images of plant leaves
- **Plants covered**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## License

This project is provided for educational and research purposes.  
The PlantVillage dataset is subject to its own [license terms on Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).
