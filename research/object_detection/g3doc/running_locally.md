# Running Locally

This page walks through the steps required to train an object detection model
on a local machine. It assumes the reader has completed the
following prerequisites:

1. The Tensorflow Object Detection API has been installed as documented in the
[installation instructions](installation.md). This includes installing library
dependencies, compiling the configuration protobufs and setting up the Python
environment.
2. A valid data set has been created. See [this page](preparing_inputs.md) for
instructions on how to generate a dataset for the PASCAL VOC challenge or the
Oxford-IIIT Pet dataset.
3. A Object Detection pipeline configuration has been written. See
[this page](configuring_jobs.md) for details on how to write a pipeline configuration.

## Recommended Directory Structure for Training and Evaluation

```
+data
  -label_map file
  -train TFRecord file
  -eval TFRecord file
+models
  + model
    -pipeline config file
    +train
    +eval
```

## Running the Training Job

A local training job can be run with the following command:

```bash
# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH={path to pipeline config file}
MODEL_DIR={path to model directory}
NUM_TRAIN_STEPS=50000
NUM_EVAL_STEPS=2000
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr
```

where `${PIPELINE_CONFIG_PATH}` points to the pipeline config and
`${MODEL_DIR}` points to the directory in which training checkpoints
and events will be written to. Note that this binary will interleave both
training and evaluation.

## Running Tensorboard

Progress for training and eval jobs can be inspected using Tensorboard. If
using the recommended directory structure, Tensorboard can be run using the
following command:

```bash
tensorboard --logdir=${MODEL_DIR}
```

where `${MODEL_DIR}` points to the directory that contains the
train and eval directories. Please note it may take Tensorboard a couple minutes
to populate with data.
