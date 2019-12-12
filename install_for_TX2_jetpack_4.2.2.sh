#!/bin/bash
sudo apt install python3-pip python3-numpy python3-matplotlib python3-pil python3-lxml python3-tk-y
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev 
wget https://developer.download.nvidia.com/compute/redist/jp/v42/tensorflow-gpu/tensorflow_gpu-1.14.0+nv19.7-cp36-cp36m-linux_aarch64.whl
pip3 install tensorflow_gpu-1.14.0+nv19.7-cp36-cp36m-linux_aarch64.whl

sudo apt install protobuf-compiler python3-tk -y
rm -rf tf_detection
echo "export PYTHONPATH=$(pwd)/research:$(pwd)/research/slim" >> tx2_tf_detection
source tx2_tf_detection && pip3 install jupyter notebook Cython contextlib2
git clone https://github.com/JinFree/cocoapi.git
cd cocoapi/PythonAPI
python3 setup.py build_ext --inplace
cd ../../
cp -r cocoapi/PythonAPI/pycocotools research/
cd research/
protoc object_detection/protos/*.proto --python_out=.

python3 object_detection/builders/model_builder_test.py
