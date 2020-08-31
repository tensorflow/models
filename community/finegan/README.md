# finegan  
  
[![Paper](http://img.shields.io/badge/paper-arXiv.1811.11155-B3181B.svg)](https://arxiv.org/abs/1811.11155) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vishal-V/tf-models/blob/master/...)  

This repository is the unofficial implementation of the following [[Paper]](https://arxiv.org/abs/1811.11155).

* FineGAN: Unsupervised Hierarchical Disentanglement for Fine-Grained Object Generation and Discovery 

## Description

FineGAN, a novel unsupervised GAN framework, which disentangles the background, object shape, and object appearance to hierarchically generate images of fine-grained object categories. To disentangle the factors without supervision, the key idea is to use information theory to associate each factor to a latent code, and to condition the relationships between the codes in a specific way to induce the desired hierarchy. Through extensive experiments, FineGAN achieves the desired disentanglement to generate realistic and diverse images belonging to fine-grained classes of birds, dogs, and cars. FineGAN's automatically learned features can also cluster real images as a first attempt at solving the novel problem of unsupervised fine-grained object category discovery.
  
<img src="../assets/finegan.png" width="960px" height="377px"/>  
  
  
## Key Features

- [x] TensorFlow 2.2.0
- [x] Inference example
- [x] Eager mode training with `tf.GradientTape`
- [x] Graph mode training with `model.train_on_batch`
- [x] Functional model with `tf.keras.layers`
- [x] Input pipeline using `tf.data` and `tfds`
- [ ] Tensorflow Serving
- [x] Vectorized transformations
- [x] GPU accelerated
- [x] Clean implementation
- [x] Following the best practices
- [x] Apache 2.0 License

## Requirements

[![TensorFlow 2.3](https://img.shields.io/badge/tensorflow-2.3-brightgreen)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Python 3.7](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-382/)

To install requirements:

```setup
pip install -r requirements.txt
```

## Results
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/finegan-unsupervised-hierarchical/image-generation-on-cub-128-x-128)](https://paperswithcode.com/sota/image-generation-on-cub-128-x-128?p=finegan-unsupervised-hierarchical)


### Image Generation
 
| Model name | Download | FID | Inception Score |  
|------------|----------|----------------|----------------|  
| Model name | [Checkpoint (300 epochs)](https://drive.google.com/drive/folders/1sE52YsxEftFgLrzlpgucQKMw2LCd4IZd?usp=sharing), [Notebook](https://github.com/Vishal-V/tf-models/blob/master/finegan/notebooks/efficient%20trials.ipynb)| 35.76 [11.25] | 14.7 [52.53]|  
  

## Dataset
  
CUB 200 - Fine grained image dataset with 200 classes of birds.
  
## Training

First download the dataset by running the following command:
```shell
python3 data_download.py
```
Please run this command line for training the model from scratch for 100 epochs.

```shell
python3 train.py
```

## Evaluation

>  
> * [TODO]   Benchmark the model and used saved_model to run inferences

<!-- Please run this command line for evaluation.

```shell
python3 ...
``` -->

## References

```
@inproceedings{singh-cvpr2019,
  title = {FineGAN: Unsupervised Hierarchical Disentanglement for Fine-Grained Object Generation and Discovery},
  author = {Krishna Kumar Singh and Utkarsh Ojha and Yong Jae Lee},
  booktitle = {CVPR},
  year = {2019}
}
```

## Citation


## Authors or Maintainers

* Vishal Vinod ([@Vishal-V](https://github.com/Vishal-V))
  
This project is licensed under the terms of the **Apache License 2.0**.
