# MobileNet_v1

[MobileNets](https://arxiv.org/abs/1704.04861) are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models, such as Inception, are used. MobileNets can be run efficiently on mobile devices with [TensorFlow Mobile](https://www.tensorflow.org/mobile/).

MobileNets trade off between latency, size and accuracy while comparing favorably with popular models from the literature.

![alt text](mobilenet_v1.png "MobileNet Graph")

# Pre-trained Models

Choose the right MobileNet model to fit your latency and size budget. The size of the network in memory and on disk is proportional to the number of parameters. The latency and power usage of the network scales with the number of Multiply-Accumulates (MACs) which measures the number of fused Multiplication and Addition operations. These MobileNet models have been trained on the
[ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/)
image classification dataset. Accuracies were computed by evaluating using a single image crop.

Model Checkpoint | Million MACs | Million Parameters | Top-1 Accuracy| Top-5 Accuracy |
:----:|:------------:|:----------:|:-------:|:-------:|
[MobileNet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz)|569|4.24|70.7|89.5|
[MobileNet_v1_1.0_192](http://download.tensorflow.org/models/mobilenet_v1_1.0_192_2017_06_14.tar.gz)|418|4.24|69.3|88.9|
[MobileNet_v1_1.0_160](http://download.tensorflow.org/models/mobilenet_v1_1.0_160_2017_06_14.tar.gz)|291|4.24|67.2|87.5|
[MobileNet_v1_1.0_128](http://download.tensorflow.org/models/mobilenet_v1_1.0_128_2017_06_14.tar.gz)|186|4.24|64.1|85.3|
[MobileNet_v1_0.75_224](http://download.tensorflow.org/models/mobilenet_v1_0.75_224_2017_06_14.tar.gz)|317|2.59|68.4|88.2|
[MobileNet_v1_0.75_192](http://download.tensorflow.org/models/mobilenet_v1_0.75_192_2017_06_14.tar.gz)|233|2.59|67.4|87.3|
[MobileNet_v1_0.75_160](http://download.tensorflow.org/models/mobilenet_v1_0.75_160_2017_06_14.tar.gz)|162|2.59|65.2|86.1|
[MobileNet_v1_0.75_128](http://download.tensorflow.org/models/mobilenet_v1_0.75_128_2017_06_14.tar.gz)|104|2.59|61.8|83.6|
[MobileNet_v1_0.50_224](http://download.tensorflow.org/models/mobilenet_v1_0.50_224_2017_06_14.tar.gz)|150|1.34|64.0|85.4|
[MobileNet_v1_0.50_192](http://download.tensorflow.org/models/mobilenet_v1_0.50_192_2017_06_14.tar.gz)|110|1.34|62.1|84.0|
[MobileNet_v1_0.50_160](http://download.tensorflow.org/models/mobilenet_v1_0.50_160_2017_06_14.tar.gz)|77|1.34|59.9|82.5|
[MobileNet_v1_0.50_128](http://download.tensorflow.org/models/mobilenet_v1_0.50_128_2017_06_14.tar.gz)|49|1.34|56.2|79.6|
[MobileNet_v1_0.25_224](http://download.tensorflow.org/models/mobilenet_v1_0.25_224_2017_06_14.tar.gz)|41|0.47|50.6|75.0|
[MobileNet_v1_0.25_192](http://download.tensorflow.org/models/mobilenet_v1_0.25_192_2017_06_14.tar.gz)|34|0.47|49.0|73.6|
[MobileNet_v1_0.25_160](http://download.tensorflow.org/models/mobilenet_v1_0.25_160_2017_06_14.tar.gz)|21|0.47|46.0|70.7|
[MobileNet_v1_0.25_128](http://download.tensorflow.org/models/mobilenet_v1_0.25_128_2017_06_14.tar.gz)|14|0.47|41.3|66.2|


Here is an example of how to download the MobileNet_v1_1.0_224 checkpoint:

```shell
$ CHECKPOINT_DIR=/tmp/checkpoints
$ mkdir ${CHECKPOINT_DIR}
$ wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
$ tar -xvf mobilenet_v1_1.0_224_2017_06_14.tar.gz
$ mv mobilenet_v1_1.0_224.ckpt.* ${CHECKPOINT_DIR}
$ rm mobilenet_v1_1.0_224_2017_06_14.tar.gz
```
More information on integrating MobileNets into your project can be found at the [TF-Slim Image Classification Library](https://github.com/tensorflow/models/blob/master/slim/README.md).

To get started running models on-device go to [TensorFlow Mobile](https://www.tensorflow.org/mobile/).
