# Running the TensorFlow Official ResNet with TensorRT

[TensorRT](https://developer.nvidia.com/tensorrt) is NVIDIA's inference
optimizer for deep learning. Briefly, TensorRT rewrites parts of the
execution graph to allow for faster prediction times.

Here we provide a sample script that can:

1. Convert a TensorFlow SavedModel to a Frozen Graph.
2. Load a Frozen Graph for inference.
3. Time inference loops using the native TensorFlow graph.
4. Time inference loops using FP32, FP16, or INT8<sup>1</sup> precision modes from TensorRT.

We provide some results below, as well as instructions for running this script.

<sup>1</sup> INT8 mode is a work in progress; please see [INT8 Mode is the Bleeding Edge](#int8-mode-is-the-bleeding-edge) below.

## How to Run This Script

### Step 1: Install Prerequisites

1. [Install TensorFlow.](https://www.tensorflow.org/install/)
2. [Install TensorRT.](http://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)
3. We use the image processing functions from the [Official version of ResNet](/official/resnet/imagenet_preprocessing.py). Please checkout the Models repository if you haven't
already, and add the Official Models to your Python path:

```
git clone https://github.com/tensorflow/models.git
export PYTHONPATH="$PYTHONPATH:/path/to/models"
```

### Step 2: Get a model to test

The provided script runs with the [Official version of ResNet trained with
ImageNet data](/official/resnet), but can be used for other models as well,
as long as you have a SavedModel or a Frozen Graph.

You can download the ResNetv2-ImageNet [SavedModel](http://download.tensorflow.org/models/official/resnetv2_imagenet_savedmodel.tar.gz)
or [Frozen Graph](http://download.tensorflow.org/models/official/resnetv2_imagenet_frozen_graph.pb),
or, if you want to train the model yourself,
pass `--export_dir` to the Official ResNet [imagenet_main.py](/official/resnet/imagenet_main.py).

When running this script, you can pass in a SavedModel directory containing the
Protobuf MetaGraphDef and variables directory to `savedmodel_dir`, or pass in
a Protobuf frozen graph file directly to `frozen_graph`. If you downloaded the
SavedModel linked above, note that you should untar it before passing in to the
script.

### Step 3: Get an image to test

The script can accept a JPEG image file to use for predictions. If none is
provided, random data will be generated. We provide a sample `image.jpg` here
which can be passed in with the `--image_file` flag.

### Step 4: Run the model

You have TensorFlow, TensorRT, a graph def, and a picture.
Now it's time to time.

For the full set of possible parameters, you can run
`python tensorrt.py --help`. Assuming you used the files provided above,
you would run:

```
python tensorrt.py --frozen_graph=resnetv2_imagenet_frozen_graph.pb \
  --image_file=image.jpg --native --fp32 --fp16 --output_dir=/my/output
```

This will print the predictions for each of the precision modes that were run
(native, which is the native precision of the model passed in, as well
as the TensorRT version of the graph at precisions of fp32 and fp16):

```
INFO:tensorflow:Starting timing.
INFO:tensorflow:Timing loop done!
Predictions:
Precision:  native [u'seashore, coast, seacoast, sea-coast', u'promontory, headland, head, foreland', u'breakwater, groin, groyne, mole, bulwark, seawall, jetty', u'lakeside, lakeshore', u'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus']
Precision:  FP32 [u'seashore, coast, seacoast, sea-coast', u'promontory, headland, head, foreland', u'breakwater, groin, groyne, mole, bulwark, seawall, jetty', u'lakeside, lakeshore', u'sandbar, sand bar']
Precision:  FP16 [u'seashore, coast, seacoast, sea-coast', u'promontory, headland, head, foreland', u'lakeside, lakeshore', u'sandbar, sand bar', u'breakwater, groin, groyne, mole, bulwark, seawall, jetty']
```

The script will generate or append to a file in the output_dir, `log.txt`,
which includes the timing information for each of the models:

```
==========================
network: native_resnetv2_imagenet_frozen_graph.pb,   batchsize 128, steps 100
  fps   median: 1041.4,   mean: 1056.6,   uncertainty: 2.8,   jitter: 6.1
  latency   median: 0.12292,  mean: 0.12123,  99th_p: 0.13151,  99th_uncertainty: 0.00024

==========================
network: tftrt_fp32_resnetv2_imagenet_frozen_graph.pb,   batchsize 128, steps 100
  fps   median: 1253.0,   mean: 1250.8,   uncertainty: 3.4,   jitter: 17.3
  latency   median: 0.10215,  mean: 0.10241,  99th_p: 0.11482,  99th_uncertainty: 0.01109

==========================
network: tftrt_fp16_resnetv2_imagenet_frozen_graph.pb,   batchsize 128, steps 100
  fps   median: 2280.2,   mean: 2312.8,   uncertainty: 10.3,  jitter: 100.1
  latency   median: 0.05614,  mean: 0.05546,  99th_p: 0.06103,  99th_uncertainty: 0.00781

```

The script will also output the GraphDefs used for each of the modes run,
for future use and inspection:

```
ls /my/output
log.txt
tftrt_fp16_imagenet_frozen_graph.pb
tftrt_fp32_imagenet_frozen_graph.pb
```

## Troubleshooting and Notes

### INT8 Mode is the Bleeding Edge

Note that currently, INT8 mode results in a segfault using the models provided.
We are working on it.

```
E tensorflow/contrib/tensorrt/log/trt_logger.cc:38] DefaultLogger Parameter check failed at: Network.cpp::addScale::118, condition: shift.count == 0 || shift.count == weightCount
Segmentation fault (core dumped)
```

### GPU/Precision Compatibility

Not all GPUs support the ops required for all precisions. For example, P100s
cannot currently run INT8 precision.

### Label Offsets

Some ResNet models represent 1000 categories, and some represent all 1001, with
the 0th category being "background". The models provided are of the latter type.
If you are using a different model and find that your predictions seem slightly
off, try passing in the `--ids_are_one_indexed` arg, which adjusts the label
alignment for models with only 1000 categories.


## Model Links
[ResNet-v2-ImageNet Frozen Graph](http://download.tensorflow.org/models/official/resnetv2_imagenet_frozen_graph.pb)

[ResNet-v2-ImageNet SavedModel](http://download.tensorflow.org/models/official/resnetv2_imagenet_savedmodel.tar.gz)

[ResNet-v1-ImageNet Frozen Graph](http://download.tensorflow.org/models/official/resnetv1_imagenet_frozen_graph.pb)

[ResNet-v1-ImageNet SavedModel](http://download.tensorflow.org/models/official/resnetv1_imagenet_savedmodel.tar.gz)
