#!/bin/bash

sudo apt update
sudo apt install unzip aria2 -y

DATA_DIR=$1
aria2c -j 8 -Z \
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
  http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip \
  http://images.cocodataset.org/zips/train2017.zip \
  http://images.cocodataset.org/zips/val2017.zip \
  --dir=$DATA_DIR;

unzip $DATA_DIR/"*".zip -d $DATA_DIR;
mkdir $DATA_DIR/zips && mv $DATA_DIR/*.zip $DATA_DIR/zips;
unzip $DATA_DIR/annotations/panoptic_train2017.zip -d $DATA_DIR
unzip $DATA_DIR/annotations/panoptic_val2017.zip -d $DATA_DIR

python3 official/vision/data/create_coco_tf_record.py \
  --logtostderr  \
  --image_dir="$DATA_DIR/val2017" \
  --object_annotations_file="$DATA_DIR/annotations/instances_val2017.json"  \
  --output_file_prefix="$DATA_DIR/tfrecords/val"  \
  --panoptic_annotations_file="$DATA_DIR/annotations/panoptic_val2017.json" \
  --panoptic_masks_dir="$DATA_DIR/panoptic_val2017" \
  --num_shards=8 \
  --include_masks \
  --include_panoptic_masks


python3 official/vision/data/create_coco_tf_record.py \
  --logtostderr  \
  --image_dir="$DATA_DIR/train2017" \
  --object_annotations_file="$DATA_DIR/annotations/instances_train2017.json"  \
  --output_file_prefix="$DATA_DIR/tfrecords/train"  \
  --panoptic_annotations_file="$DATA_DIR/annotations/panoptic_train2017.json" \
  --panoptic_masks_dir="$DATA_DIR/panoptic_train2017" \
  --num_shards=32 \
  --include_masks \
  --include_panoptic_masks
