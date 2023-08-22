# TF-NLP Model Garden

⚠️ Disclaimer: Datasets hyperlinked from this page are not owned or distributed
by Google. Such datasets are made available by third parties. Please review the
terms and conditions made available by the third parties before using the data.

This codebase provides a Natural Language Processing modeling toolkit written in
[TF2](https://www.tensorflow.org/guide/effective_tf2). It allows researchers and
developers to reproduce state-of-the-art model results and train custom models
to experiment new research ideas.

## Features

*   Reusable and modularized modeling building blocks
*   State-of-the-art reproducible
*   Easy to customize and extend
*   End-to-end training
*   Distributed trainable on both GPUs and TPUs

## Major components

### Libraries

We provide modeling library to allow users to train custom models for new
research ideas. Detailed instructions can be found in READMEs in each folder.

*   [modeling/](modeling): modeling library that provides building blocks
    (e.g.,Layers, Networks, and Models) that can be assembled into
    transformer-based architectures.
*   [data/](data): binaries and utils for input preprocessing, tokenization,
    etc.

### State-of-the-Art models and examples

We provide SoTA model implementations, pre-trained models, training and
evaluation examples, and command lines. Detail instructions can be found in the
READMEs for specific papers. Below are some papers implemented in the repository
and more NLP projects can be found in the
[`projects`](https://github.com/tensorflow/models/tree/master/official/projects)
folder:

1.  [BERT](MODEL_GARDEN.md#available-model-configs): [BERT: Pre-training of Deep
    Bidirectional Transformers for Language
    Understanding](https://arxiv.org/abs/1810.04805) by Devlin et al., 2018
2.  [ALBERT](MODEL_GARDEN.md#available-model-configs):
    [A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
    by Lan et al., 2019
3.  [XLNet](MODEL_GARDEN.md):
    [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
    by Yang et al., 2019
4.  [Transformer for translation](MODEL_GARDEN.md#available-model-configs):
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et
    al., 2017

### Common Training Driver

We provide a single common driver [train.py](train.py) to train above SoTA
models on popular tasks. Please see [docs/train.md](docs/train.md) for more
details.

### Pre-trained models with checkpoints and TF-Hub

We provide a large collection of baselines and checkpoints for NLP pre-trained
models. Please see [docs/pretrained_models.md](docs/pretrained_models.md) for
more details.

## More Documentations

Please read through the model training tutorials and references in the
[docs/ folder](docs/README.md).
