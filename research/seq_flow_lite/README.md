# Sequence Projection Models

This repository contains implementation of the following papers.

* [*PRADO: Projection Attention Networks for Document Classification On-Device*](https://www.aclweb.org/anthology/D19-1506/)
* [*Self-Governing Neural Networks for On-Device Short Text Classification*](https://www.aclweb.org/anthology/D18-1105/)

## Description

We provide a family of models that projects sequence to fixed sized features.
The idea behind is to build embedding-free models that minimize the model size.
Instead of using embedding table to lookup embeddings, sequence projection
models computes them on the fly.


## History

### August 24, 2020
* Add PRADO and SGNN implementation.

## Authors or Maintainers

* Prabhu Kaliamoorthi
* Yicheng Fan ([@thunderfyc](https://github.com/thunderfyc))


## Requirements

[![TensorFlow 2.3](https://img.shields.io/badge/TensorFlow-2.3-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.3.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)


## Training

Train a PRADO model on civil comments dataset

```shell
bazel run -c opt :trainer -- \
--config_path=$(pwd)/configs/civil_comments_prado.txt \
--runner_mode=train --logtostderr --output_dir=/tmp/prado
```

Train a SGNN model to detect languages:

```shell
bazel run -c opt sgnn:train -- --logtostderr --output_dir=/tmp/sgnn
```

## Evaluation

Evaluate PRADO model:

```shell
bazel run -c opt :trainer -- \
--config_path=$(pwd)/configs/civil_comments_prado.txt \
--runner_mode=eval --logtostderr --output_dir=/tmp/prado
```

Evaluate SGNN model:
```shell
bazel run -c opt sgnn:run_tflite -- --model=/tmp/sgnn/model.tflite "Hello world"
```


## References

1.  **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**<br />
    Sergey Ioffe, Christian Szegedy <br />
    [[link]](https://arxiv.org/abs/1502.03167). In ICML, 2015.

2.  **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference**<br />
    Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko <br />
    [[link]](https://arxiv.org/abs/1712.05877). In CVPR, 2018.

3.  **PRADO: Projection Attention Networks for Document Classification On-Device**<br/>
    Prabhu Kaliamoorthi, Sujith Ravi, Zornitsa Kozareva <br />
    [[link]](https://www.aclweb.org/anthology/D19-1506/). In EMNLP-IJCNLP, 2019

4.  **Self-Governing Neural Networks for On-Device Short Text Classification**<br />
    Sujith Ravi, Zornitsa Kozareva <br />
    [[link]](https://www.aclweb.org/anthology/D18-1105). In EMNLP, 2018

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the terms of the **Apache License 2.0**.
