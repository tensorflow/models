This forlder contains the Keras implementation of the ResNet models. For more 
information about the models, please refer to this [README file](../README.md).

Similar to the [estimator implementation](/official/resnet), the Keras 
implementation has code for both CIFAR-10 data and ImageNet data.  To use 
either dataset, Make sure that you have the latest version of TensorFlow 
installed and 
[add the models folder to your Python path](/official/#running-the-models),
otherwise you may encounter an error like `ImportError: No module named 
official.resnet`.

## CIFAR-10

Download and extract the CIFAR-10 data. You can use the following script:
```bash
python cifar10_download_and_extract.py
# Then to train the model, run the following:
python cifar10_main.py

```

After you download the data, specify the location with the `--data_dir` flag, 
like:

```python
python keras_cifar_main.py --data_dir=/path/to/cifar
```

## ImageNet

Download the ImageNet dataset and convert it to TFRecord format. 
The following [script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py)
and [README](https://github.com/tensorflow/tpu/tree/master/tools/datasets#imagenet_to_gcspy)
provide a few options.

Once your dataset is ready, you can begin training the model as follows:

```bash
python keras_imagenet_main.py --data_dir=/path/to/imagenet
```

There are more flag options you can specify. See 
[`keras_common_test.py`](keras_common_test.py) for full list of options.

## Compute Devices
Training is accomplished using the DistributionStrategies API. (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md)

The appropriate distribution strategy is chosen based on the `--num_gpus` flag.
By default this flag is one if TensorFlow is compiled with CUDA, and zero
otherwise.

num_gpus:
+ 0:  Use OneDeviceStrategy and train on CPU.
+ 1:  Use OneDeviceStrategy and train on GPU.
+ 2+: Use MirroredStrategy (data parallelism) to distribute a batch between devices.


