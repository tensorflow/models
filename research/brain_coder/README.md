# Brain Coder

*Authors: Daniel Abolafia, Mohammad Norouzi, Quoc Le*

Brain coder is a code synthesis experimental environment. We provide code that reproduces the results from our recent paper [Neural Program Synthesis with Priority Queue Training](https://arxiv.org/abs/1801.03526). See single_task/README.md for details on how to build and reproduce those experiments.

## Installation

First install dependencies seperately:

* [bazel](https://docs.bazel.build/versions/master/install.html)
* [TensorFlow](https://www.tensorflow.org/install/)
* [scipy](https://www.scipy.org/install.html)
* [absl-py](https://github.com/abseil/abseil-py)

Note: even if you already have these dependencies installed, make sure they are
up-to-date to avoid unnecessary debugging.


## Building

Use bazel from the top-level repo directory.

For example:

```bash
bazel build single_task:run
```

View README.md files in subdirectories for more details.
