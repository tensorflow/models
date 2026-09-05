<!-- disableFinding(LINE_OVER_80) -->
# RF-DETR Auto-Labeler Pipeline

An automated data generation and extraction pipeline that builds a labeled
image classification dataset from raw, unannotated full-scene images using a
pretrained RF-DETR instance segmentation model on waste or trash images.

This pipeline is intended for scenarios where **not enough labeled data is
available to train a full segmentation or object detection model**. Instead of
requiring expensive manual annotation, the pipeline automatically extracts and
crops individual objects from full-frame images to produce a balanced,
ready-to-train dataset for downstream image classifiers.

---

## Dataset Structure

### 1. Input Layout (Initial State)

Raw images must be organized under a common root directory, with one
subdirectory per category containing an `images/` folder:

```text
<root_dir>/
├── category_1/
│   └── images/
│       ├── raw_img_001.jpg
│       ├── raw_img_002.jpg
│       └── ...
├── category_2/
│   └── images/
│       ├── raw_img_003.jpg
│       └── ...
└── category_N/
    └── images/
        └── ...
```

### 2. Intermediate Directory Layout (After Filtering & Splitting)

During pipeline execution:

- **`<root_dir>_empty/`**: Created after `filter_sparse_images.py`. Any image
  with fewer detected objects than `min_detections` is moved here to isolate
  empty/uninformative frames while preserving relative paths.
- **`train_val_images/`**: Created after `split_train_val.py` beside `images/`
  within each category subfolder, containing the subsampled images partitioned
  into `train/` and `val/` splits.

```text
<root_dir>/
├── category_1/
│   ├── images/                # Filtered images remaining after Stage 1
│   │   ├── raw_img_001.jpg
│   │   └── ...
│   └── train_val_images/      # Created by Stage 2
│       ├── train/
│       │   ├── img_001.jpg
│       │   └── ...
│       └── val/
│           ├── img_002.jpg
│           └── ...
└── category_2/
    ├── images/
    │   └── ...
    └── train_val_images/
        ├── train/
        │   └── ...
        └── val/
            └── ...

<root_dir>_empty/              # Created by Stage 1 (sparse / empty images)
├── category_1/
│   └── images/
│       ├── empty_img_001.jpg
│       └── ...
└── category_2/
    └── images/
        └── ...
```

### 3. Output Layout (Classifier Dataset)

The pipeline outputs a classifier-ready dataset structured in standard
**PyTorch `ImageFolder` format** under `<root_dir>_classifier/`, containing
cropped and augmented object images:

```text
<root_dir>_classifier/
├── train/
│   ├── category_1/
│   │   ├── crop_001.jpg
│   │   ├── crop_001_vflip.jpg
│   │   └── ...
│   ├── category_2/
│   │   └── ...
│   └── category_N/
│       └── ...
└── val/
    ├── category_1/
    │   ├── crop_101.jpg
    │   └── ...
    ├── category_2/
    │   └── ...
    └── category_N/
        └── ...
```

---

## Pipeline Workflow

The pipeline runs sequentially through 4 modular stages:

1. **Stage 1: Filter Sparse Images (`filter_sparse_images.py`)**
   Runs RF-DETR detection on raw images in `images/` and identifies near-empty
   frames containing fewer objects than `min_detections`. These frames are
   moved out to a sibling `<root_dir>_empty/` directory so subsequent stages
   avoid processing uninformative scenes.

2. **Stage 2: Split into Train / Val (`split_train_val.py`)**
   Subsamples the remaining images (e.g. keeping every *N*-th frame to remove
   temporal duplicates) and creates a sibling `train_val_images/` folder under
   each category, splitting the images into `train/` and `val/` subdirectories
   according to `train_ratio`.

3. **Stage 3: RF-DETR Detection & Object Cropping (`segmentation.py`)**
   Runs RF-DETR object detection over each split in `train_val_images/`, filters
   overlapping and duplicate detections, crops each individual detected
   object, and saves letterboxed crops into
   `<root_dir>_classifier/{train,val}/<category>/`.

4. **Stage 4: Training Data Augmentation (`augment_train_split.py`)**
   Applies spatial and photometric augmentations (such as flips, rotations,
   blur, noise, and color jitter) **strictly to the `train/` split** under
   `<root_dir>_classifier/train/`. The `val/` split is left unaugmented for
   unbiased evaluation.

---

## Contents & Module Overview

| File / Directory | Description |
| :--- | :--- |
| `main.sh` | Orchestration shell script that runs all 4 pipeline stages sequentially. |
| `config.yaml` | Central configuration file containing all tunable paths, thresholds, and pipeline knobs. |
| `config_loader.py` | Configuration parser, schema validator, and type checker. |
| `filter_sparse_images.py` | Stage 1: Detects and isolates sparse/empty images. |
| `split_train_val.py` | Stage 2: Subsamples images and partitions them into train and validation sets. |
| `segmentation.py` | Stage 3: Runs RF-DETR detection, filtering, and object cropping. |
| `augment_train_split.py` | Stage 4: Generates augmented image copies for the training split. |
| `detection_utils.py` | Utility functions for RF-DETR model initialization, inference, bounding box merging, and filtering. |

---

## Configuration Parameters

All settings are configured in `config.yaml`. Before launching the pipeline,
configure the following parameters:

### Paths & Hardware

- **`root_dir`**: Path to the parent directory containing the raw class subfolders.
- **`rfdetr_checkpoint_path`**: Path to the pretrained RF-DETR model checkpoint file.

### Dataset Sizing & Splitting

- **`keep_every_nth`**: Subsampling stride to drop consecutive redundant frames.
- **`train_ratio`**: Proportion of images assigned to the training split versus validation split.
- **`min_detections`**: Minimum number of detected objects required to keep an image from being moved to empty storage.

### Object Detection & Cropping

- **`crop_size`**: Target dimensions `[height, width]` for letterboxed object crop outputs.
- **`crop_variants`**: Background rendering style for cropped objects (e.g., raw, black_background, imagenet_mean_background).

### Data Augmentation

- **`augmentations`**: List of augmentation operations applied to the training crops (e.g., horizontal/vertical flips, rotations, blur, noise, color jitter).
- **`rotation_fill_color`**: RGB color values used to pad borders when rotating cropped images.

---

## Running the Pipeline

Execute the pipeline via `main.sh`:

```bash
bash main.sh
```

---

## Author

- **Umair Sabir** - Lead Machine Learning Engineer
