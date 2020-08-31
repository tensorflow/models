# Only BatchNorm
  
[![Paper](http://img.shields.io/badge/paper-arXiv.2003.00152-B3181B.svg)](https://arxiv.org/pdf/2003.00152.pdf) 
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vishal-V/tf-models/blob/master/...)   -->

This repository is the unofficial implementation of the following [[Paper]](https://arxiv.org/pdf/2003.00152.pdf).

* Training BatchNorm and Only BatchNorm: On the Expressivity of Random Features in CNNs

## Description/Abstract

Batch normalization (BatchNorm) has become an indispensable tool for training
deep neural networks, yet it is still poorly understood. Although previous work
has typically focused on studying its normalization component, BatchNorm also
adds two per-feature trainable parameters—a coefficient and a bias—whose role
and expressive power remain unclear. To study this question, we investigate the
performance achieved when training only these parameters and freezing all others
at their random initializations. We find that doing so leads to surprisingly high
performance. For example, sufficiently deep ResNets reach 82% (CIFAR-10) and
32% (ImageNet, top-5) accuracy in this configuration, far higher than when training
an equivalent number of randomly chosen parameters elsewhere in the network.
BatchNorm achieves this performance in part by naturally learning to disable
around a third of the random features. Not only do these results highlight the
under-appreciated role of the affine parameters in BatchNorm, but—in a broader
sense—they characterize the expressive power of neural networks constructed
simply by shifting and rescaling random features.

  
<img src="../assets/onlybn.png" width="860px" height="257px"/>  
  
<!-- ## History

> :memo: Provide a changelog. -->
  
## Key Features

- [x] TensorFlow 2.3.0
- [x] Inference example (Colab Demo)
- [x] Graph mode training with `model.fit`
- [x] Functional model with `tf.keras.layers`
- [x] Input pipeline using `tf.data` and `tfds`
- [x] GPU accelerated
- [ ] Fully integrated with `absl-py` from [abseil.io](https://abseil.io)
- [x] Clean implementation
- [x] Following the best practices
- [x] Apache 2.0 License

## Requirements

[![TensorFlow 2.3](https://img.shields.io/badge/tensorflow-2.3-brightgreen)](https://github.com/tensorflow/tensorflow/releases/tag/v2.3.0)
[![Python 3.7](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-382/)


To install requirements:

```setup
pip install -r requirements.txt
```

## Results
#
### Image Classification (Only BatchNorm weights) 
 
| Model name | Download | Top 1 Accuracy |
|------------|----------|----------------|
| ResNet-14 (N=2)| [Checkpoint](https://drive.google.com/...) | 46.67% |
| ResNet-32 (N=5)| [Checkpoint](https://drive.google.com/...) | 51.29% |
| ResNet-56 (N=9)| [Checkpoint](https://drive.google.com/...) | 55.21% |
| ResNet-110 (N=18)| [Checkpoint](https://drive.google.com/...) | 65.19% |
| ResNet-218 (N=36)| [Checkpoint](https://drive.google.com/...) | 70.09% |
| ResNet-434 (N=72)| [Checkpoint](https://drive.google.com/...) | 73.67% |
| ResNet-866 (N=144)| [Checkpoint](https://drive.google.com/...) | 77.83% |
#  
## Dataset

`CIFAR10` dataset - 10 classes with 50,000 images in the train set and 10,000 images in the test set.
  

## Training

> :memo: Provide training information.  
>  
> * Provide details for preprocessing, hyperparameters, random seeds, and environment.  
> * Provide a command line example for training.  

Please run this command line for training.

```shell
python3 resnet_cifar.py
```
This trains the OnlyBN model for the ResNet-14 architecture. Replace `num_blocks` with the appropriate value for 'N' from the results table above to train the respective ResNet architecture.  
  
## Evaluation
<!-- 
> :memo: Provide an evaluation script with details of how to reproduce results.  
>  
> * Describe data preprocessing / postprocessing steps.  
> * Provide a command line example for evaluation.   -->

Please run this command line for evaluation.

```shell
python3 ...
```

## References

> :memo: Provide links to references.  

## Citation

> :memo: Make your repository citable.  
>  
> * Reference: [Making Your Code Citable](https://guides.github.com/activities/citable-code/)  

If you want to cite this repository in your research paper, please use the following information.

## Authors or Maintainers

* Vishal Vinod ([@Vishal-V](https://github.com/Vishal-V))
  
This project is licensed under the terms of the **Apache License 2.0**.
