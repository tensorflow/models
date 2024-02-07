# VideoGLUE: Video General Understanding Evaluation of Foundation Models
[![Paper](http://img.shields.io/badge/Paper-arXiv.2307.03166-B3181B?logo=arXiv)](https://arxiv.org/abs/2307.03166)

This repository provides the official TensorFlow 2 implementation of
[VideoGLUE: Video General Understanding Evaluation of Foundation Models](https://arxiv.org/abs/2307.03166)

<p align="center">
  <img src="https://storage.googleapis.com/tf_model_garden/vision/videoglue/artifacts/VideoGLUE-fig2.jpg" height=350>
</p>
<p align="center">
  <em>Figure 1: We study four adaptation methods to apply a foundation model to
  video understanding downstream tasks: (a) end-to-end finetuning, (b) frozen
  backbone evaluation, (c) frozen features with multi-layer attention pooler,
  and (d) a low-rank adapter.</em>
</p>


## Description

We evaluate existing foundation models video understanding capabilities using a
carefully designed experiment protocol consisting of three hallmark tasks
(action recognition, temporal localization, and spatiotemporal localization),
eight datasets well received by the community, and four adaptation methods
tailoring a foundation model (FM) for a downstream task. Moreover, we propose a
scalar VideoGLUE score (VGS) to measure an FMs efficacy and efficiency when
adapting to general video understanding tasks. Our main findings are as follows.
First, task-specialized models significantly outperform the six FMs studied in
this work, in sharp contrast to what FMs have achieved in natural language and
image understanding. Second,video-native FMs, whose pretraining data contains
the video modality, are generally better than image-native FMs in classifying
motion-rich videos, localizing actions in time, and understanding a video of
more than one action. Third, the video-native FMs can perform well on video
tasks under light adaptations to downstream tasks(e.g., freezing the FM
backbones), while image-native FMs win in full end-to-end finetuning. The first
two observations reveal the need and tremendous opportunities to conduct
research on video-focused FMs, and the last confirms that both tasks and
adaptation methods matter when it comes to the evaluation of FMs.

## Requirements
* [DMVR: DeepMind Video Readers](https://github.com/deepmind/dmvr)
* [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)

## Tasks
### Video Classification
Use the following script to run the `video classification` experiment. Update the config to fit your compute environment.

```shell
PARAMS_OVERRIDE="task.train_data.global_batch_size=2,\
task.train_data.shuffle_buffer_size=2,\
task.validation_data.global_batch_size=4,\
trainer.steps_per_loop=2,\
trainer.validation_steps=2,\
runtime.distribution_strategy=mirrored,\
runtime.mixed_precision_dtype=float32"

CONFIG="${PWD}/official/projects/videoglue/configs/yaml/vmae/ft/vc_vmae_vit3d_sthv2.yaml"
EXPERIMENT="mh_video_classification_strong_aug"
MODE="train"  # change to 'eval' for running evaluation loop.

python3 official/projects/videoglue/train.py \
--model_dir="/tmp/video_classification" \
--mode="${MODE}" \
--experiment="${EXPERIMENT}" \
--config_file="${CONFIG}" \
--params_override="${PARAMS_OVERRIDE}"
```

### Spatiotemporal Action Localization
Use the following script to run the `spatiotemporal action localization`
experiment. Update the config to fit your compute environment.

```shell
PARAMS_OVERRIDE="task.train_data.global_batch_size=2,\
task.train_data.shuffle_buffer_size=2,\
task.validation_data.global_batch_size=4,\
trainer.steps_per_loop=2,\
trainer.validation_steps=2,\
runtime.distribution_strategy=mirrored,\
runtime.mixed_precision_dtype=float32"

CONFIG="${PWD}/third_party/tensorflow_models/official/projects/videoglue/configs/yaml/vmae/ft/stal_vmae_vit3d_ava.yaml"
EXPERIMENT="spatiotemporal_action_localization_vit12"
MODE="train"  # change to 'eval' for running evaluation loop.

python3 -m official/projects/videoglue/train.py \
--model_dir="/tmp/spatiotemporal_action_localization" \
--mode="${MODE}" \
--experiment="${EXPERIMENT}" \
--config_file="${CONFIG}" \
--params_override="${PARAMS_OVERRIDE}"
```

### Temporal Action Localization

Following prior works, we employ
[G-TAD](https://arxiv.org/abs/1911.11462) as our task head for predicting
action categories and start and end timestamps. Please follow this
[implementation](https://github.com/frostinassiky/gtad) to run Temporal Action
Localization benchmarks.

To extract features on ActivityNet using FM of choice, we use clips of `16
frames` at a frame rate of 15 fps and a stride of `16 frames` (i.e.,
non-overlapping clips). This gives one feature vector per `16/15 ~= 1.067`
seconds.

## Code
- [x] Tasks
  - [x] Task: Video Classification.
  - [x] Task: Spatiotemporal Action Localization.
  - [x] Task: Temporal Action Localization.
- [x] Adaptations
  - [x] Adaptation: End-to-end fine-tuning.
  - [x] Adaptation: Frozen backbone pooler head.
  - [ ] Adaptation: Frozen backbone multi-layer attention pooling.
  - [ ] Adaptation: Low-rank adapter fine-tuning.

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the terms of the **Apache License 2.0**.

## Citation
```
@inproceedings{yuan2023videoglue,
  title={VideoGLUE: Video General Understanding Evaluation of Foundation Models}
  author={Yuan, Liangzhe and Gundavarapu, Nitesh Bharadwaj and Zhao, Long and
  Zhou, Hao and Cui, Yin and Jiang, Lu and Yang, Xuan and Jia, Menglin and
  Weyand, Tobias and Friedman, Luke and Sirotenko, Mikhail and Wang, Huisheng
  and Schroff, Florian and Adam, Hartwig and Yang, Ming-Hsuan and Liu, Ting and
  Gong, Boqing}
  booktitle={arXiv},
  year={2023}
}
```