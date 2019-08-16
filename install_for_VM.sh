#!/bin/bash
sudo apt install protobuf-compiler python3-tk -y
pip3 install virtualenv
source start_tf_detection && pip install tensorflow==1.12.0 jupyter notebook Cython contextlib2 matplotlib pillow lxml 

