This folder contains the Keras implementation of the ResNet models. For more
information about the models, please refer to this [README file](../../README.md).

Similar to the [estimator implementation](../../r1/resnet), the Keras
implementation has code for both CIFAR-10 data and ImageNet data. The CIFAR-10
version uses a ResNet56 model implemented in
[`resnet_cifar_model.py`](./resnet_cifar_model.py), and the ImageNet version
uses a ResNet50 model implemented in [`resnet_model.py`](./resnet_model.py).

To use
either dataset, make sure that you have the latest version of TensorFlow
installed and
[add the models folder to your Python path](/official/#running-the-models),
otherwise you may encounter an error like `ImportError: No module named
official.resnet`.

## CIFAR-10

Download and extract the CIFAR-10 data. You can use the following script:
```bash
python ../../r1/resnet/cifar10_download_and_extract.py
```

After you download the data, you can run the program by:

```bash
python resnet_cifar_main.py
```

If you did not use the default directory to download the data, specify the
location with the `--data_dir` flag, like:

```bash
python resnet_cifar_main.py --data_dir=/path/to/cifar
```

## ImageNet

Download the ImageNet dataset and convert it to TFRecord format.
The following [script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py)
and [README](https://github.com/tensorflow/tpu/tree/master/tools/datasets#imagenet_to_gcspy)
provide a few options.

Once your dataset is ready, you can begin training the model as follows:

```bash
python resnet_imagenet_main.py
```

Again, if you did not download the data to the default directory, specify the
location with the `--data_dir` flag:

```bash
python resnet_imagenet_main.py --data_dir=/path/to/imagenet
```

There are more flag options you can specify. Here are some examples:

- `--use_synthetic_data`: when set to true, synthetic data, rather than real
data, are used;
- `--batch_size`: the batch size used for the model;
- `--model_dir`: the directory to save the model checkpoint;
- `--train_epochs`: number of epoches to run for training the model;
- `--train_steps`: number of steps to run for training the model. We now only
support a number that is smaller than the number of batches in an epoch.
- `--skip_eval`: when set to true, evaluation as well as validation during
training is skipped

For example, this is a typical command line to run with ImageNet data with
batch size 128 per GPU:

```bash
python -m resnet_imagenet_main \
    --model_dir=/tmp/model_dir/something \
    --num_gpus=2 \
    --batch_size=128 \
    --train_epochs=90 \
    --train_steps=10 \
    --use_synthetic_data=false
```

See [`common.py`](common.py) for full list of options.

## Using multiple GPUs
You can train these models on multiple GPUs using `tf.distribute.Strategy` API.
You can read more about them in this
[guide](https://www.tensorflow.org/guide/distribute_strategy).

In this example, we have made it easier to use is with just a command line flag
`--num_gpus`. By default this flag is 1 if TensorFlow is compiled with CUDA,
and 0 otherwise.

- --num_gpus=0: Uses tf.distribute.OneDeviceStrategy with CPU as the device.
- --num_gpus=1: Uses tf.distribute.OneDeviceStrategy with GPU as the device.
- --num_gpus=2+: Uses tf.distribute.MirroredStrategy to run synchronous
distributed training across the GPUs.

If you wish to run without `tf.distribute.Strategy`, you can do so by setting
`--distribution_strategy=off`.

