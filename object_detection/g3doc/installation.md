# Installation

## Dependencies

Tensorflow Object Detection API depends on the following libraries:

* Protobuf 2.6
* Pillow 1.0
* lxml
* tf Slim (which is included in the "tensorflow/models" checkout)
* Jupyter notebook
* Matplotlib
* Tensorflow

For detailed steps to install Tensorflow, follow the
[Tensorflow installation instructions](https://www.tensorflow.org/install/).
A typically user can install Tensorflow using one of the following commands:

``` bash
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```

The remaining libraries can be installed on Ubuntu 16.04 using via apt-get:

``` bash
sudo apt-get install protobuf-compiler python-pil python-lxml
sudo pip install jupyter
sudo pip install matplotlib
```

Alternatively, users can install dependencies using pip:

``` bash
sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib
```

## Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following command from
the tensorflow/models directory:


``` bash
# From tensorflow/models/
protoc object_detection/protos/*.proto --python_out=.
```

## Add Libraries to PYTHONPATH

When running locally, the tensorflow/models/ and slim directories should be
appended to PYTHONPATH. This can be done by running the following from
tensorflow/models/:


``` bash
# From tensorflow/models/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file.

# Testing the Installation

You can test that you have correctly installed the Tensorflow Object Detection\
API by running the following command:

``` bash
python object_detection/builders/model_builder_test.py
```
