# ResNet in TensorFlow

Deep residual networks, or ResNets for short, provided the breakthrough idea of identity mappings in order to enable training of very deep convolutional neural networks. This folder contains an implementation of ResNet for the ImageNet dataset written in TensorFlow.

See the following papers for more background:

[1] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

[2] [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

In code, v1 refers to the ResNet defined in [1] but where a stride 2 is used on
the 3x3 conv rather than the first 1x1 in the bottleneck. This change results
in higher and more stable accuracy with less epochs than the original v1 and has
shown to scale to higher batch sizes with minimal degradation in accuracy.
There is no originating paper and the first mention we are aware of was in the
[torch version of ResNetv1](https://github.com/facebook/fb.resnet.torch). Most
popular v1 implementations are this implementation which we call ResNetv1.5. In
testing we found v1.5 requires ~12% more compute to train and has 6% reduced
throughput for inference compared to ResNetv1. Comparing the v1 model to the
v1.5 model, which has happened in blog posts, is an apples-to-oranges
comparison especially in regards to hardware or platform performance. CIFAR-10
ResNet does not use the bottleneck and is not impacted by these nuances.

v2 refers to [2]. The principle difference between the two versions is that v1
applies batch normalization and activation after convolution, while v2 applies
batch normalization, then activation, and finally convolution. A schematic
comparison is presented in Figure 1 (left) of [2].

Please proceed according to which dataset you would like to train/evaluate on:


## CIFAR-10

### Setup

You simply need to have the latest version of TensorFlow installed.
First make sure you've [added the models folder to your Python path](/official/#running-the-models); otherwise you may encounter an error like `ImportError: No module named official.resnet`.

Then download and extract the CIFAR-10 data from Alex's website, specifying the location with the `--data_dir` flag. Run the following:

```
python cifar10_download_and_extract.py
```

Then to train the model, run the following:

```
python cifar10_main.py
```

Use `--data_dir` to specify the location of the CIFAR-10 data used in the previous step. There are more flag options as described in `cifar10_main.py`.


## ImageNet

### Setup
To begin, you will need to download the ImageNet dataset and convert it to TFRecord format. Follow along with the [Inception guide](https://github.com/tensorflow/models/tree/master/research/inception#getting-started) in order to prepare the dataset.

Once your dataset is ready, you can begin training the model as follows:

```
python imagenet_main.py --data_dir=/path/to/imagenet
```

The model will begin training and will automatically evaluate itself on the validation data roughly once per epoch.

Note that there are a number of other options you can specify, including `--model_dir` to choose where to store the model and `--resnet_size` to choose the model size (options include ResNet-18 through ResNet-200). See [`resnet.py`](resnet.py) for the full list of options.


## Compute Devices
Training is accomplished using the DistributionStrategies API. (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md)

The appropriate distribution strategy is chosen based on the `--num_gpus` flag. By default this flag is one if TensorFlow is compiled with CUDA, and zero otherwise.

num_gpus:
+ 0:  Use OneDeviceStrategy and train on CPU.
+ 1:  Use OneDeviceStrategy and train on GPU.
+ 2+: Use MirroredStrategy (data parallelism) to distribute a batch between devices.

### Pre-trained model
You can download 190 MB pre-trained versions of ResNet-50. Reported accuracies are top-1 single-crop accuracy for the ImageNet validation set. Simply download and uncompress the file, and point the model to the extracted directory using the `--model_dir` flag.

ResNet-50 v2 (Accuracy 76.05%):
* [Checkpoint](http://download.tensorflow.org/models/official/20180601_resnet_v2_imagenet_checkpoint.tar.gz)
* [SavedModel](http://download.tensorflow.org/models/official/20180601_resnet_v2_imagenet_savedmodel.tar.gz)

ResNet-50 v2 (fp16, Accuracy 75.56%):
* [Checkpoint](http://download.tensorflow.org/models/official/20180601_resnet_v2_fp16_imagenet_checkpoint.tar.gz)
* [SavedModel](http://download.tensorflow.org/models/official/20180601_resnet_v2_fp16_imagenet_savedmodel.tar.gz)

ResNet-50 v1 (Accuracy 75.91%):
* [Checkpoint](http://download.tensorflow.org/models/official/20180601_resnet_v1_imagenet_checkpoint.tar.gz)
* [SavedModel](http://download.tensorflow.org/models/official/20180601_resnet_v1_imagenet_savedmodel.tar.gz)

### Transfer Learning
You can use a pretrained model to initialize a training process. In addition you are able to freeze all but the final fully connected layers to fine tune your model. Transfer Learning is useful when training on your own small datasets. For a brief look at transfer learning in the context of convolutional neural networks, we recommend reading these [short notes](http://cs231n.github.io/transfer-learning/).


To fine tune a pretrained resnet you must make three changes to your training procedure:

1) Build the exact same model as previously except we change the number of labels in the final classification layer.

2) Restore all weights from the pre-trained resnet except for the final classification layer; this will get randomly initialized instead.

3) Freeze earlier layers of the network

We can perform these three operations by specifying two flags: ```--pretrained_model_checkpoint_path``` and ```--fine_tune```. The first flag is a string that points to the path of a pre-trained resnet model. If this flag is specified, it will load all but the final classification layer. A key thing to note: if both ```--pretrained_model_checkpoint_path``` and a non empty ```model_dir``` directory are passed, the tensorflow estimator will load only the ```model_dir```. For more on this please see [WarmStartSettings](https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/WarmStartSettings) and [Estimators](https://www.tensorflow.org/guide/estimators).

The second flag ```--fine_tune``` is a boolean that indicates whether earlier layers of the network should be frozen. You may set this flag to false if you wish to continue training a pre-trained model from a checkpoint. If you set this flag to true, you can train a new classification layer from scratch.
