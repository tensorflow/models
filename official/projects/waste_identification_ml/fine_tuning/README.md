# CircularNet Fune-tuning Guide

## Below are the steps to fune-tune CircularNet on a custom training dataset

1.  Create a VM instance in Compute Engine of Google Cloud Platform with desired
    number of GPUs.
2.  Install compatible Cuda version, and validate GPU devices using
    `nvidia-smi` command.
3.  SSH to the VM instance in Compute Engine and create a conda environment
    `conda create -n circularnet-train python=3.11`
4. Activate the conda environment
    `conda activate circularnet-train`
5. Install the following libraries
    `pip install tensorflow[and-cuda] tf-models-official`
6. Move training data in TFRecord format to a GCP bucket, or into the VM
     instance.
7. Move the configuration file for model training into the VM. The configuration
     file contains all the parameters and path to datasets. A sample
     configuration file `config.yaml` has been provided for GPU training, and
     description of few entries is provided below.
8. Create a directory to save the output checkpoints.
9. Run the following command to initiate the training -
    `python -m official.vision.train --experiment="circularnet_finetuning"
    --mode="train_and_eval" --model_dir="output_directory"
    --config_file="config.yaml"`
10. Training can also be run in the background by starting a screen session.

## Config file parameters

-   `annotation_file` - path to the validation file in COCO JSON format.
-   `init_checkpoint` - path to the checkpoints for transfer learning, these
     be the CircularNet checkpoints.
-   `init_checkpoint_modules` - to load both the backbone or decoder or any one
     of them.
-   `freeze_backbone` - to freeze backbone while training.
-   `input_size` - image size according to which the model is trained.
-   `num_classes` - total number of classes + 1 ( background )
-   `per_category_metrics` - to derive metric for each class
-   `global_batch_size` - batch size.
-   `input_path` - path to the input dataset set.
-   `parser` - contains the data augmentation operations.
-   `steps_per_loop` - number of steps to complete one epoch. It's usually
     `training data size / batch size`.
-   `summary_interval` - interval to plot the metrics
-   `train_steps` - total steps for training. Its equal to
     `steps_per_loop x epochs`
-   `validation_interval` - interval to evaluate the validation data.
-   `validation_steps` - steps to cover validation data. Its equal to
     `validation data size / batch size`
-   `warmup_learning_rate` - the warm-up phase is an initial stage in the
     training process where the learning rate is gradually increased from a very
     low value to the base learning rate. The warmup_learning_rate is typically
     set to a small fraction of the base learning rate
-   `warmup_steps` - steps for the warmup learning rate
-   `initial_learning_rate` - The initial learning rate is the value of the
     learning rate at the very start of the training process.
-   `checkpoint_interval` - number of steps to export the model.

## A common practice to calculate the parameters are below:

```python
total_training_samples = 4389
total_validation_samples = 485

train_batch_size = 512
val_batch_size = 128
num_epochs = 700
warmup_learning_rate = 0.0001
initial_learning_rate = 0.001

steps_per_loop = total_training_samples // train_batch_size
summary_interval = steps_per_loop
train_steps = num_epochs * steps_per_loop
validation_interval = steps_per_loop
validation_steps = total_validation_samples // val_batch_size
warmup_steps = steps_per_loop * 10
checkpoint_interval = steps_per_loop * 5
decay_steps = int(train_steps)

print(f'steps_per_loop: {steps_per_loop}')
print(f'summary_interval: {summary_interval}')
print(f'train_steps: {train_steps}')
print(f'validation_interval: {validation_interval}')
print(f'validation_steps: {validation_steps}')
print(f'warmup_steps: {warmup_steps}')
print(f'warmup_learning_rate: {warmup_learning_rate}')
print(f'initial_learning_rate: {initial_learning_rate}')
print(f'decay_steps: {decay_steps}')
print(f'checkpoint_interval: {checkpoint_interval}')
```