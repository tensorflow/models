# Installation

## Dependencies

Tensorflow Object Detection API depends on the following libraries:

*   Protobuf 3.0.0
*   Python-tk
*   Pillow 1.0
*   lxml
*   tf-slim (https://github.com/google-research/tf-slim)
*   slim (which is included in the "tensorflow/models/research/" checkout)
*   Jupyter notebook
*   Matplotlib
*   Tensorflow (1.15.0)
*   Cython
*   contextlib2
*   cocoapi

For detailed steps to install Tensorflow, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/). Note that as of now, tf2.x
versions are not supported. A typical user can install Tensorflow using one
of the following commands:

``` bash
# For CPU
pip install tensorflow==1.15
# For GPU
pip install tensorflow-gpu==1.15
```

The remaining libraries can be installed on Ubuntu 16.04 using via apt-get:

```bash
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib
pip install --user tf_slim
```

Alternatively, users can install dependencies using pip:

```bash
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib
pip install --user tf_slim
```

<!-- common_typos_disable -->
**Note**: sometimes "sudo apt-get install protobuf-compiler" will install
Protobuf 3+ versions for you and some users have issues when using 3.5.
If that is your case, try the [manual](#Manual-protobuf-compiler-installation-and-usage) installation.

## Download the tensorflow/models repository

```bash
git clone https://github.com/tensorflow/models.git
```

To use this library, you need to download this repository, whenever it says
`<path-to-tensorflow>` it will be referring to the folder that you downloaded
this repository into.

## COCO API installation

Download the
[cocoapi](https://github.com/cocodataset/cocoapi) and
copy the pycocotools subfolder to the tensorflow/models/research directory if
you are interested in using COCO evaluation metrics. The default metrics are
based on those used in Pascal VOC evaluation. To use the COCO object detection
metrics add `metrics_set: "coco_detection_metrics"` to the `eval_config` message
in the config file. To use the COCO instance segmentation metrics add
`metrics_set: "coco_mask_metrics"` to the `eval_config` message in the config
file.

```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/
```

Alternatively, users can install `pycocotools` using pip:

```bash
pip install --user pycocotools
```

## Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following command from
the [tensorflow/models/research/
](https://github.com/tensorflow/models/tree/master/research/)
directory:


``` bash
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

**Note**: If you're getting errors while compiling, you might be using an incompatible protobuf compiler. If that's the case, use the following manual installation

## Manual protobuf-compiler installation and usage

**If you are on linux:**

Download and install the 3.0 release of protoc, then unzip the file.

```bash
# From tensorflow/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
```

Run the compilation process again, but use the downloaded version of protoc

```bash
# From tensorflow/models/research/
./bin/protoc object_detection/protos/*.proto --python_out=.
```

**If you are on MacOS:**

If you have homebrew, download and install the protobuf with
```brew install protobuf```

Alternately, run:
```PROTOC_ZIP=protoc-3.3.0-osx-x86_64.zip
curl -OL https://github.com/google/protobuf/releases/download/v3.3.0/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
rm -f $PROTOC_ZIP
```

Run the compilation process again:

``` bash
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

## Add Libraries to PYTHONPATH

When running locally, the tensorflow/models/research/ and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
tensorflow/models/research/:


``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file, replacing \`pwd\` with the absolute path of
tensorflow/models/research on your system. After updating ~/.bashrc file you
can run the following command:

Note: Some of the functions defined in tensorflow/models/research/slim has been
moved to [tf-slim](https://github.com/google-research/tf-slim), so installing
tf_slim is required now.

``` bash
source ~/.bashrc
```

# Testing the Installation

You can test that you have correctly installed the Tensorflow Object Detection\
API by running the following command:


```bash
# If using Tensorflow 1.X:
python object_detection/builders/model_builder_tf1_test.py
```
