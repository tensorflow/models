# BASNet: Boundary-Aware Salient Object Detection

This repository is the unofficial implementation of the following paper. Please
see the paper
[BASNet: Boundary-Aware Salient Object Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html)
for more details.

## Requirements
[![TensorFlow 2.4](https://img.shields.io/badge/TensorFlow-2.4-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.4.0)
[![Python 3.7](https://img.shields.io/badge/Python-3.7-3776AB)](https://www.python.org/downloads/release/python-379/)

## Train
```shell
$ python3 train.py \
  --experiment=basnet_duts \
  --mode=train \
  --model_dir=$MODEL_DIR \
  --config_file=./configs/experiments/basnet_dut_gpu.yaml
```

## Test
```shell
$ python3 train.py \
  --experiment=basnet_duts \
  --mode=eval \
  --model_dir=$MODEL_DIR \
  --config_file=./configs/experiments/basnet_dut_gpu.yaml
  --params_override='runtime.num_gpus=1, runtime.distribution_strategy=one_device, task.model.input_size=[256, 256, 3]'
```

## Results
Dataset    | maxF<sub>β</sub> | relaxF<sub>β</sub>   | MAE
:--------- | :--------------- | :------------------- | -------:
DUTS-TE    | 0.865            | 0.793                | 0.046

