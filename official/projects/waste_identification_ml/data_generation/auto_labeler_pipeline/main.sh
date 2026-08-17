#!/bin/bash
# Auto-labeler pipeline: builds a labelled dataset for training an image
# classifier from raw, unlabelled images.
#
# Each dataset subfolder under `root_dir` is treated as one class. The
# pipeline uses SAM3 to detect and segment objects matching the active
# prompt, crops them, splits them into train/val, and augments the train
# split -- producing a classifier-ready dataset at `<root_dir>_classifier`
# with no manual labelling required.
#
# All stages read their settings from config.yaml (via config_loader.py) in
# the working directory. Change knobs there, not here. Stops on first error.

set -e

echo "===================================="
echo "Stage 1/4: Filter sparse images"
echo "===================================="
# Moves images with fewer than `min_detections` detected objects out to a
# sibling `<root_dir>_empty` directory, so the later stages don't waste GPU
# time on near-empty frames.
python3 filter_sparse_images.py

echo "===================================="
echo "Stage 2/4: Split into train/val"
echo "===================================="
# Subsamples each dataset (keeps every Nth image) and splits the kept
# images into `train/` and `val/` folders under each dataset's
# `train_val_images/`.
python3 split_train_val.py

echo "===================================="
echo "Stage 3/4: SAM3 segmentation"
echo "===================================="
# Runs SAM3 on every image in each split, crops out each detected object,
# and writes the crops into a classifier-ready
# `<root_dir>_classifier/{train,val}/<class>/` layout. One class per
# dataset subfolder.
python3 segmentation.py

echo "===================================="
echo "Stage 4/4: Train augmentation"
echo "===================================="
# Applies the configured augmentations (flips, rotations, blur, noise,
# jitter) to the train split only, saving augmented copies alongside the
# originals. The val split is intentionally left untouched.
python3 augment_train_split.py

echo "===================================="
echo "Pipeline complete."
echo "===================================="