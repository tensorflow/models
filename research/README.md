![Logo](https://storage.googleapis.com/model_garden_artifacts/TF_Model_Garden.png)

# TensorFlow Research Models

This directory contains code implementations and pre-trained models of published research papers.

The research models are maintained by their respective authors.

## Table of Contents
- [TensorFlow Research Models](#tensorflow-research-models)
  - [Table of Contents](#table-of-contents)
  - [Modeling Libraries and Models](#modeling-libraries-and-models)
  - [Models and Implementations](#models-and-implementations)
    - [Computer Vision](#computer-vision)
    - [Natural Language Processing](#natural-language-processing)
    - [Audio and Speech](#audio-and-speech)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Others](#others)
    - [Old Models and Implementations in TensorFlow 1](#old-models-and-implementations-in-tensorflow-1)
  - [Contributions](#contributions)

## Modeling Libraries and Models

| Directory | Name | Description | Maintainer(s) |
|-----------|------|-------------|---------------|
| [object_detection](object_detection) | TensorFlow Object Detection API | A framework that makes it easy to construct, train and deploy object detection models<br /><br />A collection of object detection models pre-trained on the COCO dataset, the Kitti dataset, the Open Images dataset, the AVA v2.1 dataset, and the iNaturalist Species Detection Dataset| jch1, tombstone, pkulzc |
| [slim](slim) | TensorFlow-Slim Image Classification Model Library | A lightweight high-level API of TensorFlow for defining, training and evaluating image classification models <br />• Inception V1/V2/V3/V4<br />• Inception-ResNet-v2<br />• ResNet V1/V2<br />• VGG 16/19<br />• MobileNet V1/V2/V3<br />• NASNet-A_Mobile/Large<br />• PNASNet-5_Large/Mobile | sguada, marksandler2 |

## Models and Implementations

### Computer Vision

| Directory | Paper(s) | Conference | Maintainer(s) |
|-----------|----------|------------|---------------|
| [attention_ocr](attention_ocr) | [Attention-based Extraction of Structured Information from Street View Imagery](https://arxiv.org/abs/1704.03549) | ICDAR 2017 | xavigibert |
| [autoaugment](autoaugment) | [1] [AutoAugment](https://arxiv.org/abs/1805.09501)<br />[2] [Wide Residual Networks](https://arxiv.org/abs/1605.07146)<br />[3] [Shake-Shake regularization](https://arxiv.org/abs/1705.07485)<br />[4] [ShakeDrop Regularization for Deep Residual Learning](https://arxiv.org/abs/1802.02375) | [1] CVPR 2019<br />[2] BMVC 2016<br /> [3] ICLR 2017<br /> [4] ICLR 2018 | barretzoph |
| [deeplab](deeplab) | [1] [DeepLabv1: Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/abs/1412.7062)<br />[2] [DeepLabv2: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)<br />[3] [DeepLabv3: Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)<br />[4] [DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)<br />| [1] ICLR 2015 <br />[2] TPAMI 2017 <br />[4] ECCV 2018 | aquariusjay, yknzhu |
| [delf](delf)  | [1] DELF (DEep Local Features): [Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/abs/1612.06321)<br />[2] [Detect-to-Retrieve: Efficient Regional Aggregation for Image Search](https://arxiv.org/abs/1812.01584)<br />[3] DELG (DEep Local and Global features): [Unifying Deep Local and Global Features for Image Search](https://arxiv.org/abs/2001.05027)<br />[4] GLDv2: [Google Landmarks Dataset v2 -- A Large-Scale Benchmark for Instance-Level Recognition and Retrieval](https://arxiv.org/abs/2004.01804) | [1] ICCV 2017<br />[2] CVPR 2019<br />[4] CVPR 2020 | andrefaraujo |
| [lstm_object_detection](lstm_object_detection) | [Mobile Video Object Detection with Temporally-Aware Feature Maps](https://arxiv.org/abs/1711.06368) | CVPR 2018 | yinxiaoli, yongzhe2160, lzyuan |
| [marco](marco) | MARCO: [Classification of crystallization outcomes using deep convolutional neural networks](https://arxiv.org/abs/1803.10342) | | vincentvanhoucke |
| [vid2depth](vid2depth) | [Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints](https://arxiv.org/abs/1802.05522) | CVPR 2018 | rezama |

### Natural Language Processing

| Directory | Paper(s) | Conference | Maintainer(s) |
|-----------|----------|------------|---------------|
| [adversarial_text](adversarial_text) | [1] [Adversarial Training Methods for Semi-Supervised Text](https://arxiv.org/abs/1605.07725) Classification<br />[2] [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432) | [1] ICLR 2017<br />[2] NIPS 2015 | rsepassi, a-dai |
| [cvt_text](cvt_text) | [Semi-Supervised Sequence Modeling with Cross-View Training](https://arxiv.org/abs/1809.08370) | EMNLP 2018 | clarkkev, lmthang |

### Audio and Speech

| Directory | Paper(s) | Conference | Maintainer(s) |
|-----------|----------|------------|---------------|
| [audioset](audioset) | [1] [Audio Set: An ontology and human-labeled dataset for audio events](https://research.google/pubs/pub45857/)<br />[2] [CNN Architectures for Large-Scale Audio Classification](https://research.google/pubs/pub45611/) | ICASSP 2017 | plakal, dpwe |
| [deep_speech](deep_speech) | [Deep Speech 2](https://arxiv.org/abs/1512.02595) | ICLR 2016 | yhliang2018 |

### Reinforcement Learning

| Directory | Paper(s) | Conference | Maintainer(s) |
|-----------|----------|------------|---------------|
| [efficient-hrl](efficient-hrl) | [1] [Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296)<br />[2] [Near-Optimal Representation Learning for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1810.01257) | [1] NIPS 2018<br /> [2] ICLR 2019 | ofirnachum |
| [pcl_rl](pcl_rl) | [1] [Improving Policy Gradient by Exploring Under-appreciated Rewards](https://arxiv.org/abs/1611.09321)<br />[2] [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892)<br />[3] [Trust-PCL: An Off-Policy Trust Region Method for Continuous Control](https://arxiv.org/abs/1707.01891) | [1] ICLR 2017<br />[2] NIPS 2017<br />[3] ICLR 2018 | ofirnachum |

### Others

| Directory | Paper(s) | Conference | Maintainer(s) |
|-----------|----------|------------|---------------|
| [lfads](lfads) | [LFADS - Latent Factor Analysis via Dynamical Systems](https://arxiv.org/abs/1608.06315) | | jazcollins, sussillo |
| [rebar](rebar) | [REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models](https://arxiv.org/abs/1703.07370) | NIPS 2017 | gjtucker |

### Old Models and Implementations in TensorFlow 1

:warning: If you are looking for old models, please visit the [Archive branch](https://github.com/tensorflow/models/tree/archive/research).

---

## Contributions

If you want to contribute, please review the [contribution guidelines](https://github.com/tensorflow/models/wiki/How-to-contribute).
