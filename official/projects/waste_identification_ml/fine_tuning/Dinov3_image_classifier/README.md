<!-- disableFinding(LINE_OVER_80) -->
# DINOv3 Image Classifier Fine-Tuning

This folder contains utilities and training scripts for fine-tuning a Facebook
DINOv3 Vision Transformer (**ViT-L/16**) backbone with a custom linear
classification head on image classification datasets.

---

## Dataset Structure

The training pipeline expects dataset images to be structured in the standard
**PyTorch `ImageFolder` format**.

The root dataset directory must contain two top-level folders:

- `train/`: Training images used to optimize the model weights.
- `val/`: Validation images used to monitor generalization loss and trigger early
  stopping / checkpointing.

Each of `train/` and `val/` must contain subfolders named strictly after the
category/class labels. Image files (`.jpg`, `.jpeg`, `.png`, etc.) are placed
inside their respective category folders.

### Directory Tree (Text)

```text
dataset_root/
├── train/
│   ├── category_1/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── category_2/
│   │   ├── image_003.jpg
│   │   └── ...
│   └── category_N/
│       └── ...
└── val/
    ├── category_1/
    │   ├── image_101.jpg
    │   ├── image_102.jpg
    │   └── ...
    ├── category_2/
    │   ├── image_103.jpg
    │   └── ...
    └── category_N/
        └── ...
```

---

## Training Techniques: v1 vs v2

This directory provides two fine-tuning techniques tailored for different
training strategies:

- **`train_classifier_v1.py` (Standard Unified Training)**:
  - **Unified Learning Rate**: Applies a single learning rate and uniform weight
    decay across all trainable parameters (backbone and classification head).
  - **Direct Cosine Decay**: Decays the learning rate smoothly from its initial
    value to the minimum learning rate floor starting from epoch 0 (no warmup).

- **`train_classifier_v2.py` (Advanced Split-LR & Warmup Training)**:
  - **Split Learning Rates**: Uses a lower learning rate for the pretrained
    backbone (for gentle nudging) and a higher learning rate for the randomly
    initialized head (for faster convergence).
  - **Selective Weight Decay**: Weight decay is applied only to 2D weight
    matrices; biases and 1D normalization parameters (e.g. LayerNorm) are
    excluded.
  - **Warmup + Cosine Schedule**: Includes a linear warmup phase for the first
    ~10% of epochs to stabilize optimizer momentum before transitioning into
    cosine decay.

---

## Contents & Module Overview

| File / Directory | Description |
| :--- | :--- |
| `train_classifier_v1.py` | Training script using a uniform learning rate and direct cosine decay schedule. |
| `train_classifier_v2.py` | Advanced training script with split backbone/head learning rates, parameter-group weight decay, and linear warmup. |
| `models.py` | DINOv3 model wrapper, linear classification head, and feature pooling strategies (`POOLING_CLS`, `POOLING_CLS_MEAN_PATCH`). |
| `datasets.py` | PyTorch `ImageFolder` data loading, preprocessing transforms, and normalization. |
| `training_callbacks.py` | Callbacks for early stopping, best checkpoint saving, and loss/accuracy curve plotting. |

---

## Model Architecture & Training Details

- **Backbone**: DINOv3 Vision Transformer Large with 16x16 patch size
  (`dinov3_vitl16`).
- **Feature Dimension**: 1024-dimensional embedding (or 2048 when using
  concatenated patch pooling).
- **Pooling Strategy**: `POOLING_CLS` (final CLS token) or
  `POOLING_CLS_MEAN_PATCH` (CLS concatenated with mean patch tokens).
- **Mixed Precision**: CUDA `bfloat16` autocast for accelerated training
  throughput and reduced GPU memory footprint.
- **Gradient Clipping**: Maximum L2 norm gradient clipping (1.0).
- **Callbacks**: `SaveBestModel` checkpoint saving and `EarlyStopping` based on
  validation loss.

---

## Configuration Parameters

All configurations are defined as module-level constants at the top of
`train_classifier_v1.py` and `train_classifier_v2.py`. Adjust the following
parameters before launching training:

### Dataset & Model Paths

- **`TRAIN_DIRECTORY`**: Path to the root training dataset directory containing
  class subdirectories of images in PyTorch `ImageFolder` format.
- **`VALIDATION_DIRECTORY`**: Path to the root validation dataset directory
  used for evaluating model generalization and early stopping.
- **`DINOV3_REPO_DIRECTORY`**: Path to the local clone of the Facebook DINOv3
  repository, used to load the model architecture via `torch.hub`.
- **`DINOV3_WEIGHTS_PATH`**: Path to the pretrained DINOv3 backbone checkpoint
  weights file (`.pth`).
- **`OUTPUT_DIRECTORY`**: Destination folder where model checkpoints, training
  logs, and loss/accuracy plots are saved.

### Training Hyperparameters

- **`BATCH_SIZE`**: Number of image samples processed per batch during training
  and validation iterations.
- **`EPOCHS`**: Total number of complete training passes over the dataset.
- **`IMAGE_SIZE`**: Target square resolution (height and width) to which input
  images are resized.
- **`NUMBER_OF_WORKERS`**: Number of parallel CPU worker processes used by the
  data loaders.
- **`USE_CLASS_WEIGHTS`**: Boolean flag to compute and apply inverse frequency
  class weights to counteract class imbalance.
- **`EARLY_STOPPING_PATIENCE`**: Number of epochs to wait without validation
  loss improvement before halting training early.

### Optimizer & Schedule Parameters

**For `train_classifier_v1.py`:**

- **`LEARNING_RATE`**: Uniform learning rate applied across both backbone and
  classification head.
- **`COSINE_MINIMUM_LEARNING_RATE`**: Minimum learning rate floor reached at the
  end of the cosine schedule.
- **`WEIGHT_DECAY`**: Weight decay penalty applied uniformly across all
  trainable parameters.

**For `train_classifier_v2.py`:**

- **`BACKBONE_LEARNING_RATE`**: Lower learning rate for fine-tuning the
  pretrained backbone layers.
- **`HEAD_LEARNING_RATE`**: Higher learning rate for optimizing the randomly
  initialized classification head.
- **`WARMUP_EPOCHS_FRACTION`**: Fraction of total epochs dedicated to linear
  learning rate warmup.
- **`WARMUP_START_FACTOR`**: Initial learning rate multiplier at the start of
  the warmup phase.
- **`COSINE_MINIMUM_LEARNING_RATE`**: Minimum learning rate floor reached at
  the conclusion of cosine decay.
- **`WEIGHT_DECAY`**: Weight decay applied selectively to 2D weight matrices
  (excluding biases and 1D LayerNorms).

---

## Author

- **Umair Sabir** - Lead Machine Learning Engineer
