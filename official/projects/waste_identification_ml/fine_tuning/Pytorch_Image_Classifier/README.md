<!-- disableFinding(LINE_OVER_80) -->
# Vision Transformer (ViT) Image Classifier Fine-Tuning

This folder contains utilities, scripts, and notebooks for training,
fine-tuning, and evaluating a PyTorch-based Vision Transformer (**ViT-B/16**)
image classifier using transfer learning. The pipeline is designed for
fine-grained classification tasks such as waste and recyclables identification.

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

## Contents & Module Overview

| File / Directory | Description |
| :--- | :--- |
| `vit_training.py` | Main script for running training and fine-tuning the ViT image classifier. |
| `training_with_callbacks.py` | Core training loop (`train`, `train_step`, `test_step`) supporting learning rate schedulers and `EarlyStopping` with checkpoint saving. |
| `inference_utils.py` | Helper functions for loading fine-tuned ViT models, running single-image inference, running batch dataset evaluation, and plotting confusion matrices. |
| `Inference_ImageClassifier_TransferLearning.ipynb` | Inference and evaluation notebook: checkpoint loading, performance metrics, and confusion matrix visualizations. |

---

## Model Architecture & Training Details

- **Backbone**: Vision Transformer Base with 16x16 patch size
  (`torchvision.models.vit_b_16`).
- **Feature Dimension**: 768-dimensional embedding.
- **Transfer Learning**: Backbone weights are pre-trained on ImageNet and
  frozen during transfer learning; the linear classifier head (`model.heads`)
  is replaced with a custom `nn.Linear(in_features=768, out_features=num_classes)`.
- **Input Resolution**: 224 x 224 pixels.
- **Transforms & Normalization**:
  - Resize / Crop to `(224, 224)`
  - Normalization using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`,
    `std=[0.229, 0.224, 0.225]`)
- **Callbacks**:
  - `EarlyStopping`: Monitors validation loss, halts training after a
    configurable patience threshold, and saves the best model checkpoint
    (`best_model_epoch_<epoch>.pt`).

---

## Configuration Parameters

Before running the training script, configure the following parameters:

### Dataset & Checkpoint Paths

- **`TRAIN_DATA_DIRECTORY`**: Path to the root training dataset directory
  containing class subdirectories of images in PyTorch `ImageFolder` format.
- **`VALIDATION_DATA_DIRECTORY`**: Path to the root validation dataset
  directory used for evaluating model performance and monitoring early
  stopping.
- **`MODEL_OUTPUT_PATH`**: Destination directory and filename prefix where the
  best model checkpoints written by the early stopping callback are saved.

### Training Hyperparameters

- **`BATCH_SIZE`**: Number of image samples processed per batch during training
  and validation iterations.
- **`NUMBER_OF_EPOCHS`**: Total number of complete training passes over the
  dataset.
- **`LEARNING_RATE`**: Initial learning rate for the optimizer updating the
  classification head.
- **`SCHEDULER_MINIMUM_LEARNING_RATE`**: The minimum learning rate floor
  reached by the learning rate scheduler during decay.

---

## Author

- **Umair Sabir** - Lead Machine Learning Engineer
