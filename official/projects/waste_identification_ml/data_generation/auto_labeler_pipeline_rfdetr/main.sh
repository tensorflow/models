#!/bin/bash
# Auto-labeler pipeline: builds a labelled dataset for training an image
# classifier from raw, unlabelled images.
#
# Each dataset subfolder under `root_dir` is treated as one class. The
# pipeline uses RF-DETR to detect objects, crops them, splits them into
# train/val, and augments the train split -- producing a classifier-ready
# dataset at `<root_dir>_classifier` with no manual labelling required.
#
# All stages read their settings from config.yaml (via config_loader.py) in
# the working directory. Change knobs there, not here. Stops on first error.

set -e

echo "===================================="
echo "Download RF-DETR model checkpoint"
echo "===================================="
# The CircularNet team publishes an RF-DETR-Seg-Medium checkpoint fine-tuned
# on waste imagery. This gives you waste-specific classes (bottles, pouches,
# wrappers, sachets, etc.) instead of the 80 generic COCO classes that the
# default RFDETRSegMedium() weights would provide.
CHECKPOINT_URL="https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/CN-ModelCheckpoints/July2026_checkpoint/checkpoint_best_total.pth"
CHECKPOINT_PATH="./checkpoint_best_total.pth"

wget -nc -O "${CHECKPOINT_PATH}" "${CHECKPOINT_URL}"

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
echo "Stage 3/4: RF-DETR segmentation"
echo "===================================="
# Runs RF-DETR on every image in each split, crops out each detected object,
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