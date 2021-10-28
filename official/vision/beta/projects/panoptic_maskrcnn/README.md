# Panoptic Segmentation

## Description

Panoptic Segmentation combines the two distinct vision tasks - semantic
segmentation and instance segmentation. These tasks are unified such that, each
pixel in the image is assigned the label of the class it belongs to, and also
the instance identifier of the object it a part of.

## Environment setup
The code can be run on multiple GPUs or TPUs with different distribution
strategies. See the TensorFlow distributed training
[guide](https://www.tensorflow.org/guide/distributed_training) for an overview
of `tf.distribute`.

The code is compatible with TensorFlow 2.6+. See requirements.txt for all
prerequisites.

```bash
$ git clone https://github.com/tensorflow/models.git
$ cd models
$ pip3 install -r official/requirements.txt
```

## Preparing Dataset
### Download and extract COCO dataset
```bash
$ sudo apt update
$ sudo apt install unzip aria2 -y

$ export DATA_DIR=<path-to-store-tfrecords>
$ aria2c -j 8 -Z \
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
  http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip \
  http://images.cocodataset.org/zips/train2017.zip \
  http://images.cocodataset.org/zips/val2017.zip \
  --dir=$DATA_DIR;

$ unzip $DATA_DIR/"*".zip -d $DATA_DIR;
$ mkdir $DATA_DIR/zips && mv $DATA_DIR/*.zip $DATA_DIR/zips;
$ unzip $DATA_DIR/annotations/panoptic_train2017.zip -d $DATA_DIR
$ unzip $DATA_DIR/annotations/panoptic_val2017.zip -d $DATA_DIR
```

### Create TFrecords
```bash
cd official/vision/beta/data

$ python3 create_coco_tf_record.py \
  --logtostderr  \
  --image_dir="$DATA_DIR/val2017" \
  --object_annotations_file="$DATA_DIR/annotations/instances_val2017.json"  \
  --output_file_prefix="$DATA_DIR/tfrecords/val"  \
  --panoptic_annotations_file="$DATA_DIR/annotations/panoptic_val2017.json" \
  --panoptic_masks_dir="$DATA_DIR/panoptic_val2017" \
  --num_shards=8 \
  --include_masks \
  --include_panoptic_masks


$ python3 create_coco_tf_record.py \
  --logtostderr  \
  --image_dir="$DATA_DIR/train2017" \
  --object_annotations_file="$DATA_DIR/annotations/instances_train2017.json"  \
  --output_file_prefix="$DATA_DIR/tfrecords/train"  \
  --panoptic_annotations_file="$DATA_DIR/annotations/panoptic_train2017.json" \
  --panoptic_masks_dir="$DATA_DIR/panoptic_train2017" \
  --num_shards=32 \
  --include_masks \
  --include_panoptic_masks
```
### Upload tfrecords to a Google Cloud Storage Bucket
```bash
$ gsutil -m cp -r "$DATA_DIR/tfrecords" gs://<bucket-details>
```

## Launch Training
```bash
$ export MODEL_DIR="gs://<path-to-model-directory>"
$ export TPU_NAME="<tpu-name>"
$ export ANNOTATION_FILE="gs://<path-to-coco-annotation-json>"
$ export TRAIN_DATA="gs://<path-to-train-data>"
$ export EVAL_DATA="gs://<path-to-eval-data>"
$ export OVERRIDES="task.validation_data.input_path=${EVAL_DATA},\
  task.train_data.input_path=${TRAIN_DATA},\
  task.annotation_file=${ANNOTATION_FILE},\
  runtime.distribution_strategy=tpu"


$ python3 train.py \
  --experiment panoptic_fpn_coco \
  --mode train \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --params_override=$OVERRIDES
```

## Launch Evaluation
```bash
$ export MODEL_DIR="gs://<path-to-model-directory>"
$ export NUM_GPUS="<number-of-gpus>"
$ export PRECISION="<floating-point-precision>"
$ export ANNOTATION_FILE="gs://<path-to-coco-annotation-json>"
$ export TRAIN_DATA="gs://<path-to-train-data>"
$ export EVAL_DATA="gs://<path-to-eval-data>"
$ export OVERRIDES="task.validation_data.input_path=${EVAL_DATA}, \
  task.train_data.input_path=${TRAIN_DATA}, \
  task.annotation_file=${ANNOTATION_FILE}, \
  runtime.distribution_strategy=mirrored, \
  runtime.mixed_precision_dtype=$PRECISION, \
  runtime.num_gpus=$NUM_GPUS"


$ python3 train.py \
  --experiment panoptic_fpn_coco \
  --mode eval \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --params_override=$OVERRIDES
```
**Note**: The [PanopticSegmentationGenerator](https://github.com/tensorflow/models/blob/ac7f9e7f2d0508913947242bad3e23ef7cae5a43/official/vision/beta/projects/panoptic_maskrcnn/modeling/layers/panoptic_segmentation_generator.py#L22) layer uses dynamic shapes and hence generating panoptic masks is not supported on Cloud TPUs. Running evaluation on Cloud TPUs is not supported for the same reson. 
## Pretrained Models
Backbone     | Schedule     | Experiment name             | Box mAP |  Mask mAP  | Overall PQ | Things PQ | Stuff PQ
:------------| :----------- | :---------------------------| ------- | ---------- | ---------- | --------- | -------:
ResNet-50    | 1x           | `panoptic_fpn_coco`         | 38.19   |   34.25    |   39.14    |  45.42    |  29.65
ResNet-50    | 3x           | `panoptic_fpn_coco`         | 40.64   |   36.29    |   40.91    |  47.68    |  30.69

**Note**: Here 1x schedule refers to ~12 epochs

___
## Citation

```
@misc{kirillov2019panoptic,
      title={Panoptic Feature Pyramid Networks}, 
      author={Alexander Kirillov and Ross Girshick and Kaiming He and Piotr Dollár},
      year={2019},
      eprint={1901.02446},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```