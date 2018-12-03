# AstroWaveNet: A generative model for light curves.

Implementation based on "WaveNet: A Generative Model of Raw Audio":
https://arxiv.org/abs/1609.03499

## Code Authors

Alex Tamkin: [@atamkin](https://github.com/atamkin)

Chris Shallue: [@cshallue](https://github.com/cshallue)

## Pull Requests / Issues

Chris Shallue: [@cshallue](https://github.com/cshallue)

## Additional Dependencies

In addition to the [required packages](../README.md#required-packages) listed in
the top-level README, this package requires:

* **TensorFlow 1.12 or greater** ([instructions](https://www.tensorflow.org/install/))
* **TensorFlow Probability** ([instructions](https://www.tensorflow.org/probability/install))
* **Six** ([instructions](https://pypi.org/project/six/))

## Basic Usage

To train a model on synthetic transits:

```bash
bazel build astrowavenet/...
```

```bash
bazel-bin/astrowavenet/trainer \
--dataset=synthetic_transits \
--model_dir=/tmp/astrowavenet/ \
--config_overrides='{"hparams": {"batch_size": 16, "num_residual_blocks": 2}}' \
--schedule=train_and_eval \
--eval_steps=100 \
--save_checkpoints_steps=1000
```
