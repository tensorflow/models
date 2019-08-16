#!/bin/bash
sudo apt install protobuf-compiler python3-tk -y
pip3 install virtualenv
source start_tf_detection && pip install tensorflow==1.12.0 jupyter notebook Cython contextlib2 matplotlib pillow lxml 
git clone https://github.com/JinFree/cocoapi.git
cd cocoapi/PythonAPI
cp -r pycocotools ../../research/
cd ../../models/research/
protoc object_detection/protos/*.proto --python_out=.
python object_detection/builders/model_builder_test.py
