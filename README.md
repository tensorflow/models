# Casting Defect Detection
Creating accurate machine learning models capable of localizing and identifying
multiple objects in a single image remains a core challenge in computer vision.



# Installation

## Dependencies

* Protobuf 2.6
* Pillow 1.0
* lxml
* Jupyter notebook
* Matplotlib
* Tensorflow


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

Compile the Protobuf binaries and add the object detection API to the path


``` bash
# Compile the Protobuf binaries
protoc object_detection/protos/*.proto --python_out=.

# Add API to path
export PYTHONPATH=$PYTHONPATH:`pwd`

# Test the installation
python object_detection/builders/model_builder_test.py
```
