# ResNet in TensorFlow

Deep residual networks, or ResNets for short, provided the breakthrough idea of identity mappings in order to enable training of very deep convolutional neural networks. This folder contains an implementation of ResNet for the ImageNet dataset written in TensorFlow.

See the following papers for more background:

[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

[Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

Please proceed according to which dataset you would like to train/evaluate on:


## CIFAR-10

### Setup

You simply need to have the latest version of TensorFlow installed.

First download and extract the CIFAR-10 data from Alex's website, specifying the location with the `--data_dir` flag. Run the following:

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

Note that there are a number of other options you can specify, including `--model_dir` to choose where to store the model and `--resnet_size` to choose the model size (options include ResNet-18 through ResNet-200). See [`imagenet_main.py`](imagenet_main.py) for the full list of options.

### Pre-trained model
You can download a 190 MB pre-trained version of ResNet-50 achieving 75.3% top-1 single-crop accuracy here: [resnet50_2017_11_30.tar.gz](http://download.tensorflow.org/models/official/resnet50_2017_11_30.tar.gz). Simply download and uncompress the file, and point the model to the extracted directory using the `--model_dir` flag.
