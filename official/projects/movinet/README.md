# Mobile Video Networks (MoViNets)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb)
[![TensorFlow Hub](https://img.shields.io/badge/TF%20Hub-Models-FF6F00?logo=tensorflow)](https://tfhub.dev/google/collections/movinet)
[![Paper](http://img.shields.io/badge/Paper-arXiv.2103.11511-B3181B?logo=arXiv)](https://arxiv.org/abs/2103.11511)

This repository is the official implementation of
[MoViNets: Mobile Video Networks for Efficient Video
Recognition](https://arxiv.org/abs/2103.11511).

-   **[UPDATE 2022-03-14] Quantized TF Lite models
    [available on TF Hub](https://tfhub.dev/s?deployment-format=lite&q=movinet)
    (also [see table](https://tfhub.dev/google/collections/movinet) for
    quantized performance)**

<p align="center">
  <img src="https://storage.googleapis.com/tf_model_garden/vision/movinet/artifacts/hoverboard_stream.gif" height=500>
</p>

Create your own video plot like the one above with this [Colab notebook](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/tools/plot_movinet_video_stream_predictions.ipynb).

## Description

Mobile Video Networks (MoViNets) are efficient video classification models
runnable on mobile devices. MoViNets demonstrate state-of-the-art accuracy and
efficiency on several large-scale video action recognition datasets.

On [Kinetics 600](https://deepmind.com/research/open-source/kinetics),
MoViNet-A6 achieves 84.8% top-1 accuracy, outperforming recent
Vision Transformer models like [ViViT](https://arxiv.org/abs/2103.15691) (83.0%)
and [VATT](https://arxiv.org/abs/2104.11178) (83.6%) without any additional
training data, while using 10x fewer FLOPs. And streaming MoViNet-A0 achieves
72% accuracy while using 3x fewer FLOPs than MobileNetV3-large (68%).

There is a large gap between video model performance of accurate models and
efficient models for video action recognition. On the one hand, 2D MobileNet
CNNs are fast and can operate on streaming video in real time, but are prone to
be noisy and inaccurate. On the other hand, 3D CNNs are accurate, but are
memory and computation intensive and cannot operate on streaming video.

MoViNets bridge this gap, producing:

- State-of-the art efficiency and accuracy across the model family (MoViNet-A0
to A6).
- Streaming models with 3D causal convolutions substantially reducing memory
usage.
- Temporal ensembles of models to boost efficiency even higher.

MoViNets also improve computational efficiency by outputting high-quality
predictions frame by frame, as opposed to the traditional multi-clip evaluation
approach that performs redundant computation and limits temporal scope.

<p align="center">
  <img src="https://storage.googleapis.com/tf_model_garden/vision/movinet/artifacts/movinet_multi_clip_eval.png" height=200>
</p>

<p align="center">
  <img src="https://storage.googleapis.com/tf_model_garden/vision/movinet/artifacts/movinet_stream_eval.png" height=200>
</p>

## History

- **2022-03-14** Support quantized TF Lite models and add/update Colab
notebooks.
- **2021-07-12** Add TF Lite support and replace 3D stream models with
mobile-friendly (2+1)D stream.
- **2021-05-30** Add streaming MoViNet checkpoints and examples.
- **2021-05-11** Initial Commit.

## Authors and Maintainers

* Dan Kondratyuk ([@hyperparticle](https://github.com/hyperparticle))
* Liangzhe Yuan ([@yuanliangzhe](https://github.com/yuanliangzhe))
* Yeqing Li ([@yeqingli](https://github.com/yeqingli))

## Table of Contents

- [Requirements](#requirements)
- [Results and Pretrained Weights](#results-and-pretrained-weights)
  - [Kinetics 600](#kinetics-600)
  - [Kinetics 400](#kinetics-400)
- [Prediction Examples](#prediction-examples)
- [TF Lite Example](#tf-lite-example)
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

<p align="center">
  <img src="https://storage.googleapis.com/tf_model_garden/vision/movinet/artifacts/movinet_comparison.png" height=500>
</p>

[tensorboard.dev summary](https://tensorboard.dev/experiment/Q07RQUlVRWOY4yDw3SnSkA/)
of training runs across all models.

The table below summarizes the performance of each model on
[Kinetics 600](https://deepmind.com/research/open-source/kinetics)
and provides links to download pretrained models. All models are evaluated on
single clips with the same resolution as training.

Note: MoViNet-A6 can be constructed as an ensemble of MoViNet-A4 and
MoViNet-A5.

#### Base Models

Base models implement standard 3D convolutions without stream buffers. Base
models are not recommended for fast inference on CPU or mobile due to
limited support for
[`tf.nn.conv3d`](https://www.tensorflow.org/api_docs/python/tf/nn/conv3d).
Instead, see the [streaming models section](#streaming-models).

| Model Name | Top-1 Accuracy | Top-5 Accuracy | Input Shape | GFLOPs\* | Checkpoint | TF Hub SavedModel |
|------------|----------------|----------------|-------------|----------|------------|-------------------|
| MoViNet-A0-Base | 72.28 | 90.92 | 50 x 172 x 172 | 2.7 | [checkpoint (12 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_base.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a0/base/kinetics-600/classification/) |
| MoViNet-A1-Base | 76.69 | 93.40 | 50 x 172 x 172 | 6.0 | [checkpoint (18 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a1_base.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a1/base/kinetics-600/classification/) |
| MoViNet-A2-Base | 78.62 | 94.17 | 50 x 224 x 224 | 10 | [checkpoint (20 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a2_base.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/) |
| MoViNet-A3-Base | 81.79 | 95.67 | 120 x 256 x 256 | 57 | [checkpoint (29 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a3_base.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a3/base/kinetics-600/classification/) |
| MoViNet-A4-Base | 83.48 | 96.16 | 80 x 290 x 290 | 110 | [checkpoint (44 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a4_base.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a4/base/kinetics-600/classification/) |
| MoViNet-A5-Base | 84.27 | 96.39 | 120 x 320 x 320 | 280 | [checkpoint (72 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a5_base.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a5/base/kinetics-600/classification/) |

\*GFLOPs per video on Kinetics 600.

#### Streaming Models

Streaming models implement causal (2+1)D convolutions with stream buffers.
Streaming models use (2+1)D convolution instead of 3D to utilize optimized
[`tf.nn.conv2d`](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)
operations, which offer fast inference on CPU. Streaming models can be run on
individual frames or on larger video clips like base models.

Note: A3, A4, and A5 models use a positional encoding in the squeeze-excitation
blocks, while A0, A1, and A2 do not. For the smaller models, accuracy is
unaffected without positional encoding, while for the larger models accuracy is
significantly worse without positional encoding.

| Model Name | Top-1 Accuracy | Top-5 Accuracy | Input Shape\* | GFLOPs\*\* | Checkpoint | TF Hub SavedModel |
|------------|----------------|----------------|---------------|------------|------------|-------------------|
| MoViNet-A0-Stream | 72.05 | 90.63 | 50 x 172 x 172 | 2.7 | [checkpoint (12 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_stream.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a0/stream/kinetics-600/classification/) |
| MoViNet-A1-Stream | 76.45 | 93.25 | 50 x 172 x 172 | 6.0 | [checkpoint (18 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a1_stream.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a1/stream/kinetics-600/classification/) |
| MoViNet-A2-Stream | 78.40 | 94.05 | 50 x 224 x 224 | 10 | [checkpoint (20 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a2_stream.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a2/stream/kinetics-600/classification/) |
| MoViNet-A3-Stream | 80.09 | 94.84 | 120 x 256 x 256 | 57 | [checkpoint (29 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a3_stream.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a3/stream/kinetics-600/classification/) |
| MoViNet-A4-Stream | 81.49 | 95.66 | 80 x 290 x 290 | 110 | [checkpoint (44 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a4_stream.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a4/stream/kinetics-600/classification/) |
| MoViNet-A5-Stream | 82.37 | 95.79 | 120 x 320 x 320 | 280 | [checkpoint (72 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a5_stream.tar.gz) | [tfhub](https://tfhub.dev/tensorflow/movinet/a5/stream/kinetics-600/classification/) |

\*In streaming mode, the number of frames correspond to the total accumulated
duration of the 10-second clip.

\*\*GFLOPs per video on Kinetics 600.

Note: current streaming model checkpoints have been updated with a slightly
different architecture. To download the old checkpoints, insert `_legacy` before
`.tar.gz` in the URL. E.g., `movinet_a0_stream_legacy.tar.gz`.

##### TF Lite Streaming Models

For convenience, we provide converted TF Lite models for inference on mobile
devices. See the [TF Lite Example](#tf-lite-example) to export and run your own
models. We also provide [quantized TF Lite binaries via TF Hub](https://tfhub.dev/s?deployment-format=lite&q=movinet).

For reference, MoViNet-A0-Stream runs with a similar latency to
[MobileNetV3-Large](https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/)
with +5% accuracy on Kinetics 600.

| Model Name | Input Shape | Pixel 4 Latency\* | x86 Latency\* | TF Lite Binary |
|------------|-------------|-------------------|---------------|----------------|
| MoViNet-A0-Stream | 1 x 1 x 172 x 172 | 22 ms | 16 ms | [TF Lite (13 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_stream.tflite) |
| MoViNet-A1-Stream | 1 x 1 x 172 x 172 | 42 ms | 33 ms | [TF Lite (45 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a1_stream.tflite) |
| MoViNet-A2-Stream | 1 x 1 x 224 x 224 | 200 ms | 66 ms | [TF Lite (53 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a2_stream.tflite) |
| MoViNet-A3-Stream | 1 x 1 x 256 x 256 | - | 120 ms | [TF Lite (73 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a3_stream.tflite) |
| MoViNet-A4-Stream | 1 x 1 x 290 x 290 | - | 300 ms | [TF Lite (101 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a4_stream.tflite) |
| MoViNet-A5-Stream | 1 x 1 x 320 x 320 | - | 450 ms | [TF Lite (153 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a5_stream.tflite) |

\*Single-frame latency measured on with unaltered float32 operations on a
single CPU core. Observed latency may differ depending on hardware
configuration. Measured on a stock Pixel 4 (Android 11) and x86 Intel Xeon
W-2135 CPU.

### Kinetics 400

We also have checkpoints for Kinetics 400 models available. See the Kinetics 600
sections for more details. To load checkpoints, set `num_classes=400`.

#### Base Models

| Model Name | Top-1 Accuracy | Top-5 Accuracy | Input Shape | GFLOPs\* | Checkpoint |
|------------|----------------|----------------|-------------|----------|------------|
| MoViNet-A0-Base | 69.40 | 89.18 | 50 x 172 x 172  | 2.7 | [checkpoint (12 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_base_k400.tar.gz) |
| MoViNet-A1-Base | 74.57 | 92.03 | 50 x 172 x 172  | 6.0 | [checkpoint (18 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a1_base_k400.tar.gz) |
| MoViNet-A2-Base | 75.91 | 92.63 | 50 x 224 x 224  | 10  | [checkpoint (20 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a2_base_k400.tar.gz) |
| MoViNet-A3-Base | 79.34 | 94.52 | 120 x 256 x 256 | 57  | [checkpoint (29 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a3_base_k400.tar.gz) |
| MoViNet-A4-Base | 80.64 | 94.93 | 80 x 290 x 290  | 110 | [checkpoint (44 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a4_base_k400.tar.gz) |
| MoViNet-A5-Base | 81.39 | 95.06 | 120 x 320 x 320 | 280 | [checkpoint (72 MB)](https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a5_base_k400.tar.gz) |

*GFLOPs per video on Kinetics 400.

## Prediction Examples

Please check out our [Colab Notebook](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb)
to get started with MoViNets.

This section provides examples on how to run prediction.

For **base models**, run the following:

```python
import tensorflow as tf

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

# Create backbone and model.
backbone = movinet.Movinet(
    model_id='a0',
    causal=False,
    use_external_states=False,
)
model = movinet_model.MovinetClassifier(
    backbone, num_classes=600, output_states=False)

# Create your example input here.
# Refer to the paper for recommended input shapes.
inputs = tf.ones([1, 8, 172, 172, 3])

# [Optional] Build the model and load a pretrained checkpoint
model.build(inputs.shape)

checkpoint_dir = '/path/to/checkpoint'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

# Run the model prediction.
output = model(inputs)
prediction = tf.argmax(output, -1)
```

For **streaming models**, run the following:

```python
import tensorflow as tf

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

model_id = 'a0'
use_positional_encoding = model_id in {'a3', 'a4', 'a5'}

# Create backbone and model.
backbone = movinet.Movinet(
    model_id=model_id,
    causal=True,
    conv_type='2plus1d',
    se_type='2plus3d',
    activation='hard_swish',
    gating_activation='hard_sigmoid',
    use_positional_encoding=use_positional_encoding,
    use_external_states=True,
)

model = movinet_model.MovinetClassifier(
    backbone,
    num_classes=600,
    output_states=True)

# Create your example input here.
# Refer to the paper for recommended input shapes.
inputs = tf.ones([1, 8, 172, 172, 3])

# [Optional] Build the model and load a pretrained checkpoint.
model.build(inputs.shape)

checkpoint_dir = '/path/to/checkpoint'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

# Split the video into individual frames.
# Note: we can also split into larger clips as well (e.g., 8-frame clips).
# Running on larger clips will slightly reduce latency overhead, but
# will consume more memory.
frames = tf.split(inputs, inputs.shape[1], axis=1)

# Initialize the dict of states. All state tensors are initially zeros.
init_states = model.init_states(tf.shape(inputs))

# Run the model prediction by looping over each frame.
states = init_states
predictions = []
for frame in frames:
  output, states = model({**states, 'image': frame})
  predictions.append(output)

# The video classification will simply be the last output of the model.
final_prediction = tf.argmax(predictions[-1], -1)

# Alternatively, we can run the network on the entire input video.
# The output should be effectively the same
# (but it may differ a small amount due to floating point errors).
non_streaming_output, _ = model({**init_states, 'image': inputs})
non_streaming_prediction = tf.argmax(non_streaming_output, -1)
```

## TF Lite Example

This section outlines an example on how to export a model to run on mobile
devices with [TF Lite](https://www.tensorflow.org/lite).

[Optional] For streaming models, they are typically trained with
`conv_type = 3d_2plus1d` for better training throughpouts. In order to achieve
better inference performance on CPU, we need to convert the `3d_2plus1d`
checkpoint to make it compatible with the `2plus1d` graph.
You could achieve this by running `tools/convert_3d_2plus1d.py`.

First, convert to [TF SavedModel](https://www.tensorflow.org/guide/saved_model)
by running `export_saved_model.py`. For example, for `MoViNet-A0-Stream`, run:

```shell
python3 export_saved_model.py \
  --model_id=a0 \
  --causal=True \
  --conv_type=2plus1d \
  --se_type=2plus3d \
  --activation=hard_swish \
  --gating_activation=hard_sigmoid \
  --use_positional_encoding=False \
  --num_classes=600 \
  --batch_size=1 \
  --num_frames=1 \
  --image_size=172 \
  --bundle_input_init_states_fn=False \
  --checkpoint_path=/path/to/checkpoint \
  --export_path=/tmp/movinet_a0_stream
```

Then the SavedModel can be converted to TF Lite using the [`TFLiteConverter`](https://www.tensorflow.org/lite/convert):

```python
saved_model_dir = '/tmp/movinet_a0_stream'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('/tmp/movinet_a0_stream.tflite', 'wb') as f:
  f.write(tflite_model)
```

To run with TF Lite using [tf.lite.Interpreter](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python)
with the Python API:

```python
# Create the interpreter and signature runner
interpreter = tf.lite.Interpreter('/tmp/movinet_a0_stream.tflite')
runner = interpreter.get_signature_runner()

# Extract state names and create the initial (zero) states
def state_name(name: str) -> str:
  return name[len('serving_default_'):-len(':0')]

init_states = {
    state_name(x['name']): tf.zeros(x['shape'], dtype=x['dtype'])
    for x in interpreter.get_input_details()
}
del init_states['image']

# Insert your video clip here
video = tf.ones([1, 8, 172, 172, 3])
clips = tf.split(video, video.shape[1], axis=1)

# To run on a video, pass in one frame at a time
states = init_states
for clip in clips:
  # Input shape: [1, 1, 172, 172, 3]
  outputs = runner(**states, image=clip)
  logits = outputs.pop('logits')
  states = outputs
```

Follow the [official guide](https://www.tensorflow.org/lite/guide) to run a
model with TF Lite on your mobile device.

## Training and Evaluation

Run this command line for continuous training and evaluation.

```shell
MODE=train_and_eval  # Can also be 'train' if using a separate evaluator job
CONFIG_FILE=official/projects/movinet/configs/yaml/movinet_a0_k600_8x8.yaml
python3 official/projects/movinet/train.py \
    --experiment=movinet_kinetics600 \
    --mode=${MODE} \
    --model_dir=/tmp/movinet_a0_base/ \
    --config_file=${CONFIG_FILE}
```

Run this command line for evaluation.

```shell
MODE=eval  # Can also be 'eval_continuous' for use during training
CONFIG_FILE=official/projects/movinet/configs/yaml/movinet_a0_k600_8x8.yaml
python3 official/projects/movinet/train.py \
    --experiment=movinet_kinetics600 \
    --mode=${MODE} \
    --model_dir=/tmp/movinet_a0_base/ \
    --config_file=${CONFIG_FILE}
```

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
