#!/bin/bash
sudo apt install protobuf-compiler python3-tk -y
pip3 install virtualenv
rm -rf tf_detection && virtualenv tf_detection --python=python3
echo "export PYTHONPATH=$PYTHONPATH:$(pwd)/research:$(pwd)/research/slim" >> start_tf_detection
source start_tf_detection && pip install tensorflow==1.12.0 jupyter notebook Cython contextlib2 matplotlib pillow lxml 
git clone https://github.com/JinFree/cocoapi.git
cp -r cocoapi/PythonAPI/pycocotools research/
cd research/
protoc object_detection/protos/*.proto --python_out=.
python3 object_detection/builders/model_builder_test.py
