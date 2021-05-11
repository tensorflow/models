# Mobile Video Networks (MoViNets)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorflow/models/blob/master/official/vision/beta/projects/movinet/movinet_tutorial.ipynb)
[![TensorFlow Hub](https://img.shields.io/badge/TF%20Hub-Models-FF6F00?logo=tensorflow)](https://tfhub.dev/google/collections/movinet)
[![Paper](http://img.shields.io/badge/Paper-arXiv.2103.11511-B3181B?logo=arXiv)](https://arxiv.org/abs/2103.11511)

This repository is the official implementation of
[MoViNets: Mobile Video Networks for Efficient Video
Recognition](https://arxiv.org/abs/2103.11511).

## Description

Mobile Video Networks (MoViNets) are efficient video classification models
runnable on mobile devices. MoViNets demonstrate state-of-the-art accuracy and
efficiency on several large-scale video action recognition datasets.

There is a large gap between video model performance of accurate models and
efficient models for video action recognition. On the one hand, 2D MobileNet
CNNs are fast and can operate on streaming video in real time, but are prone to
be noisy and are inaccurate. On the other hand, 3D CNNs are accurate, but are
memory and computation intensive and cannot operate on streaming video.

MoViNets bridge this gap, producing:

- State-of-the art efficiency and accuracy across the model family (MoViNet-A0
to A6).
- Streaming models with 3D causal convolutions substantially reducing memory
usage.
- Temporal ensembles of models to boost efficiency even higher.

Small MoViNets demonstrate higher efficiency and accuracy than MobileNetV3 for
video action recognition (Kinetics 600).

MoViNets also improve efficiency by outputting high-quality predictions with a
single frame, as opposed to the traditional multi-clip evaluation approach.

[![Multi-Clip Eval](https://storage.googleapis.com/tf_model_garden/vision/movinet/artifacts/movinet_multi_clip_eval.png)](https://arxiv.org/pdf/2103.11511.pdf)

[![Streaming Eval](https://storage.googleapis.com/tf_model_garden/vision/movinet/artifacts/movinet_stream_eval.png)](https://arxiv.org/pdf/2103.11511.pdf)

## History

- Initial Commit.

## Authors and Maintainers

* Dan Kondratyuk ([@hyperparticle](https://github.com/hyperparticle))
* Liangzhe Yuan ([@yuanliangzhe](https://github.com/yuanliangzhe))
* Yeqing Li ([@yeqingli](https://github.com/yeqingli))

## Table of Contents

- [Requirements](#requirements)
- [Results and Pretrained Weights](#results-and-pretrained-weights)
  - [Kinetics 600](#kinetics-600)
- [Training and Evaluation](#training-and-evaluation)
- [References](#references)
- [License](#license)
- [Citation](#citation)

## Requirements

[![TensorFlow 2.4](https://img.shields.io/badge/TensorFlow-2.1-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.1.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB?logo=python)](https://www.python.org/downloads/release/python-360/)

To install requirements:

```shell
pip install -r requirements.txt
```

## Results and Pretrained Weights

[![TensorFlow Hub](https://img.shields.io/badge/TF%20Hub-Models-FF6F00?logo=tensorflow)](https://tfhub.dev/google/collections/movinet)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-dev-FF6F00?logo=tensorflow)](https://tensorboard.dev/experiment/Q07RQUlVRWOY4yDw3SnSkA/)

### Kinetics 600

[![MoViNet Comparison](https://storage.googleapis.com/tf_model_garden/vision/movinet/artifacts/movinet_comparison.png)](https://arxiv.org/pdf/2103.11511.pdf)

[tensorboard.dev summary](https://tensorboard.dev/experiment/Q07RQUlVRWOY4yDw3SnSkA/)
of training runs across all models.

The table below summarizes the performance of each model and provides links to
download pretrained models. All models are evaluated on single clips with the
same resolution as training.

Streaming MoViNets will be added in the future.

| Model Name | Top-1 Accuracy | Top-5 Accuracy | GFLOPs\* | Checkpoint | TF Hub SavedModel |
|------------|----------------|----------------|----------|------------|-------------------|
| MoViNet-A0-Base | 71.41 | 90.91 | 2.7 | [checkpoint (12 MiB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_base.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a0/base/kinetics-600/classification/) |
| MoViNet-A1-Base | 76.01 | 93.28 | 6.0 | [checkpoint (18 MiB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a1_base.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a1/base/kinetics-600/classification/) |
| MoViNet-A2-Base | 78.03 | 93.99 | 10 | [checkpoint (20 MiB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a2_base.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/) |
| MoViNet-A3-Base | 81.22 | 95.35 | 57 | [checkpoint (29 MiB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a3_base.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a3/base/kinetics-600/classification/) |
| MoViNet-A4-Base | 82.96 | 95.98 | 110 | [checkpoint (44 MiB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a4_base.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a4/base/kinetics-600/classification/) |
| MoViNet-A5-Base | 84.22 | 96.36 | 280 | [checkpoint (72 MiB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a5_base.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a5/base/kinetics-600/classification/) |

\*GFLOPs per video on Kinetics 600.

## Training and Evaluation

Please check out our [Colab Notebook](https://colab.research.google.com/github/tensorflow/models/tree/master/official/vision/beta/projects/movinet/movinet_tutorial.ipynb)
to get started with MoViNets.

Run this command line for continuous training and evaluation.

```shell
MODE=train_and_eval  # Can also be 'train'
CONFIG_FILE=official/vision/beta/projects/movinet/configs/yaml/movinet_a0_k600_8x8.yaml
python3 official/vision/beta/projects/movinet/train.py \
    --experiment=movinet_kinetics600 \
    --mode=${MODE} \
    --model_dir=/tmp/movinet/ \
    --config_file=${CONFIG_FILE} \
    --params_override="" \
    --gin_file="" \
    --gin_params="" \
    --tpu="" \
    --tf_data_service=""
```

Run this command line for evaluation.

```shell
MODE=eval  # Can also be 'eval_continuous' for use during training
CONFIG_FILE=official/vision/beta/projects/movinet/configs/yaml/movinet_a0_k600_8x8.yaml
python3 official/vision/beta/projects/movinet/train.py \
    --experiment=movinet_kinetics600 \
    --mode=${MODE} \
    --model_dir=/tmp/movinet/ \
    --config_file=${CONFIG_FILE} \
    --params_override="" \
    --gin_file="" \
    --gin_params="" \
    --tpu="" \
    --tf_data_service=""
```

## References

- [Kinetics Datasets](https://deepmind.com/research/open-source/kinetics)
- [MoViNets (Mobile Video Networks)](https://arxiv.org/abs/2103.11511)

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the terms of the **Apache License 2.0**.

## Citation

If you want to cite this code in your research paper, please use the following
information.

```
@article{kondratyuk2021movinets,
  title={MoViNets: Mobile Video Networks for Efficient Video Recognition},
  author={Dan Kondratyuk, Liangzhe Yuan, Yandong Li, Li Zhang, Matthew Brown, and Boqing Gong},
  journal={arXiv preprint arXiv:2103.11511},
  year={2021}
}
```
