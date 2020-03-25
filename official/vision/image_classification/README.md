# Image Classification

This folder contains TF 2.0 model examples for image classification:

* [MNIST](#mnist)
* [Classifier Trainer](#classifier-trainer), a framework that uses the Keras
compile/fit methods for image classification models, including:
  * ResNet
  * EfficientNet[^1]

[^1]: Currently a work in progress. We cannot match "AutoAugment (AA)" in [the original version](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).
For more information about other types of models, please refer to this
[README file](../../README.md).

## Before you begin
Please make sure that you have the latest version of TensorFlow
installed and
[add the models folder to your Python path](/official/#running-the-models).

### ImageNet preparation

Download the ImageNet dataset and convert it to TFRecord format.
The following [script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py)
and [README](https://github.com/tensorflow/tpu/tree/master/tools/datasets#imagenet_to_gcspy)
provide a few options.

### Running on Cloud TPUs

Note: These models will **not** work with TPUs on Colab.

You can train image classification models on Cloud TPUs using
`tf.distribute.TPUStrategy`. If you are not familiar with Cloud TPUs, it is
strongly recommended that you go through the
[quickstart](https://cloud.google.com/tpu/docs/quickstart) to learn how to
create a TPU and GCE VM.

## MNIST

To download the data and run the MNIST sample model locally for the first time,
run one of the following command:

```bash
python3 mnist_main.py \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --train_epochs=10 \
  --distribution_strategy=one_device \
  --num_gpus=$NUM_GPUS \
  --download
```

To train the model on a Cloud TPU, run the following command:

```bash
python3 mnist_main.py \
  --tpu=$TPU_NAME \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --train_epochs=10 \
  --distribution_strategy=tpu \
  --download
```

Note: the `--download` flag is only required the first time you run the model.


## Classifier Trainer
The classifier trainer is a unified framework for running image classification
models using Keras's compile/fit methods. Experiments should be provided in the
form of YAML files, some examples are included within the configs/examples
folder. Please see [configs/examples](./configs/examples) for more example
configurations.

The provided configuration files use a per replica batch size and is scaled
by the number of devices. For instance, if `batch size` = 64, then for 1 GPU
the global batch size would be 64 * 1 = 64. For 8 GPUs, the global batch size
would be 64 * 8 = 512. Similarly, for a v3-8 TPU, the global batch size would
be 64 * 8 = 512, and for a v3-32, the global batch size is 64 * 32 = 2048.

### ResNet50

#### On GPU:
```bash
python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=resnet \
  --dataset=imagenet \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --config_file=configs/examples/resnet/imagenet/gpu.yaml \
  --params_override='runtime.num_gpus=$NUM_GPUS'
```

#### On TPU:
```bash
python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=resnet \
  --dataset=imagenet \
  --tpu=$TPU_NAME \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --config_file=config/examples/resnet/imagenet/tpu.yaml
```

### EfficientNet
**Note: EfficientNet development is a work in progress.**
#### On GPU:
```bash
python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=efficientnet \
  --dataset=imagenet \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --config_file=configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml \
  --params_override='runtime.num_gpus=$NUM_GPUS'
```


#### On TPU:
```bash
python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=efficientnet \
  --dataset=imagenet \
  --tpu=$TPU_NAME \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --config_file=config/examples/efficientnet/imagenet/efficientnet-b0-tpu.yaml
```

Note that the number of GPU devices can be overridden in the command line using
`--params_overrides`. The TPU does not need this override as the device is fixed
by providing the TPU address or name with the `--tpu` flag.

