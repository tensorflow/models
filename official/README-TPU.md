# Offically Supported TensorFlow 2.1 Models on Cloud TPU

## Natural Language Processing

*   [bert](nlp/bert): A powerful pre-trained language representation model:
    BERT, which stands for Bidirectional Encoder Representations from
    Transformers.
    [BERT FineTuning with Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/bert-2.x) provides step by step instructions on Cloud TPU training. You can look [Bert MNLI Tensorboard.dev metrics](https://tensorboard.dev/experiment/mIah5lppTASvrHqWrdr6NA) for MNLI fine tuning task.
*   [transformer](nlp/transformer): A transformer model to translate the WMT
    English to German dataset.
        [Training transformer on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/transformer-2.x) for step by step instructions on Cloud TPU training.

## Computer Vision

*   [mnist](vision/image_classification): A basic model to classify digits
    from the MNIST dataset. See [Running MNIST on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/mnist-2.x) tutorial and [Tensorboard.dev metrics](https://tensorboard.dev/experiment/mIah5lppTASvrHqWrdr6NA).
*   [resnet](vision/image_classification): A deep residual network that can
    be used to classify ImageNet's dataset of 1000 classes.
    See [Training ResNet on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/resnet-2.x) tutorial and [Tensorboard.dev metrics](https://tensorboard.dev/experiment/CxlDK8YMRrSpYEGtBRpOhg).
*   [retinanet](vision/detection): A fast and powerful object detector. See [Tensorboard.dev training metrics](https://tensorboard.dev/experiment/b8NRnWU3TqG6Rw0UxueU6Q).
