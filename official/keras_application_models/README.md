# Keras Application Models Benchmark
## Overview
This provides a single scaffold to benchmark the Keras built-in application [models](https://keras.io/applications/). All the models are for image classification applications, and include:

 - Xception
 - VGG16
 - VGG19
 - ResNet50
 - InceptionV3
 - InceptionResNetV2
 - MobileNet
 - DenseNet
 - NASNet

## Dataset
Synthetic dataset is used for the benchmark.

## Callbacks
Two custom callbacks are provided for model benchmarking: ExamplesPerSecondCallback and LoggingMetricCallback. For each callback, `epoch_based` and `batch_based` options are available to set the benchmark level. Check [model_callbacks.py](model_callbacks.py) for more details.

## Running Code
To benchmark a model, use `--model` to specify the model name. To perform the benchmark with eager execution, issue the following command:
```
python benchmark_main.py --model resnet50 --eager
```
Note that, if eager execution is enabled, only one GPU is utilized even if multiple GPUs are provided and multi_gpu_model is used.


To use distribution strategy in the benchmark, run the following:
```
python benchmark_main.py --model resnet50 --dist_strat
```
Currently, only one of the --eager and --dist_strat arguments can be defined, as DistributionStrategy is not supported in Eager execution now.

Arguments:
  * `--model`: Which model to be benchmarked. The model name is defined as the keys of `MODELS` in [benchmark_main.py](benchmark_main.py).
  * `--callbacks`: To specify a list of callbacks.

Use the `--help` or `-h` flag to get a full list of possible arguments.
