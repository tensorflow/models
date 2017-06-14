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
Oxford-IIT Pet dataset.
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
# From the tensorflow/models/ directory
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
```

where `${PATH_TO_YOUR_PIPELINE_CONFIG}` points to the pipeline config and
`${PATH_TO_TRAIN_DIR}` points to the directory in which training checkpoints
and events will be written to. By default, the training job will
run indefinitely until the user kills it.

## Running the Evaluation Job

Evaluation is run as a separate job. The eval job will periodically poll the
train directory for new checkpoints and evaluate them on a test dataset. The
job can be run using the following command:

```bash
# From the tensorflow/models/ directory
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}
```

where `${PATH_TO_YOUR_PIPELINE_CONFIG}` points to the pipeline config,
`${PATH_TO_TRAIN_DIR}` points to the directory in which training checkpoints
were saved (same as the training job) and `${PATH_TO_EVAL_DIR}` points to the
directory in which evaluation events will be saved. As with the training job,
the eval job run until terminated by default.

## Running Tensorboard

Progress for training and eval jobs can be inspected using Tensorboard. If
using the recommended directory structure, Tensorboard can be run using the
following command:

```bash
tensorboard --logdir=${PATH_TO_MODEL_DIRECTORY}
```

where `${PATH_TO_MODEL_DIRECTORY}` points to the directory that contains the
train and eval directories. Please note it make take Tensorboard a couple
minutes to populate with data.
