# Quantization Aware Training for NLP Models

## Description

This project includes quantization aware training (QAT) code for NLP models.
These are examples to show how to apply the Model Optimization Toolkit's
[quantization aware training API](https://www.tensorflow.org/model_optimization/guide/quantization/training).
Compared to post-training quantization (PTQ), QAT can minimize the quality loss
from quantization, while still achieving the speed-up from integer quantization.

Currently, we support a limited number of NLP tasks & models. We will keep
adding support for other tasks and models in the next releases.

## Maintainers

- Jaehong Kim ([Xhark](https://github.com/Xhark))
- Rino Lee ([rino20](https://github.com/rino20))

## Requirements

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![tf-models-official PyPI](https://badge.fury.io/py/tf-models-official.svg)](https://badge.fury.io/py/tf-models-official)

## Results
### MobileBERT

Model name            | SQUAD F1 (float) | SQUAD F1 (PTQ) | SQUAD F1 (QAT) | download   | links
:-------------------- | ---------------: | -------------: | -------------: | ---------: | ----:
MobileBERT-EdgeTPU-XS | 88.02%           | 84.96%         | 85.42%         | [FP32](https://storage.googleapis.com/tf_model_garden/nlp/qat/mobilebert/model_fp32.tflite) \| [INT8](https://storage.googleapis.com/tf_model_garden/nlp/qat/mobilebert/model_int8_ptq.tflite) \| [QAT INT8](https://storage.googleapis.com/tf_model_garden/nlp/qat/mobilebert/model_qat.tflite) ([ckpt](https://storage.googleapis.com/tf_model_garden/nlp/qat/mobilebert/mobilebert_qat.tar.gz)) | [tensorboard](https://tensorboard.dev/experiment/ky0gSa6nQva2a5ppL4Mtzw/#scalars)

Please follow
[MobileBERT QAT Tutorial Colab notebook](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/qat/nlp/docs/MobileBERT_QAT_tutorial.ipynb)
to try exported models.

## Training

It can run on Google Cloud Platform using Cloud TPU.
[Here](https://cloud.google.com/tpu/docs/how-to) is the instruction of using
Cloud TPU. Follow the below instructions to set up Cloud TPU and launch
training, using mobilebert as an exmaple:

```shell

# First, Download the pre-trained floating point model as QAT needs to finetune it.
gsutil cp gs://tf_model_garden/nlp/qat/mobilebert/mobilebert_fp32_ckpt.tar.gz /tmp/qat/

# Extract the checkpoint.
tar -xvzf /tmp/qat/mobilebert_fp32_ckpt.tar.gz

# Convert the float checkpoint to QAT checkpoint.
$ python3 pretrained_checkpoint_converter.py \
  --experiment=bert/squad \
  --config_file=<fill in>/edgetpu/nlp/experiments/downstream_tasks/mobilebert_edgetpu_xs.yaml \
  --config_file=<fill in>/edgetpu/nlp/experiments/downstream_tasks/squad_v1.yaml \
  --experiment_qat=bert/squad_qat \
  --config_file_qat=<fill in>/edgetpu/nlp/experiments/downstream_tasks/mobilebert_edgetpu_xs.yaml \
  --config_file_qat=<fill in>/qat/nlp/configs/experiments/squad_v1_mobilebert_xs_qat_1gpu.yaml \
  --pretrained_checkpoint=<fill in> \  # Example: /tmp/qat/mobilebert_fp32_ckpt
  --output_checkpoint=<fill in>  # Example: /tmp/qat/mobilebert_fp32_ckpt_qat

# Launch training. Note that we override the checkpoint path in the config file by "params_override" to supply the correct checkpoint.
PARAMS_OVERRIDE="task.quantization.pretrained_original_checkpoint=/tmp/qat/mobilebert_fp32_ckpt_qat"
EXPERIMENT=bert/squad_qat  # Experiment type according to the subtask. Example: 'bert/squad_qat'
TPU_NAME="<tpu-name>"  # The name assigned while creating a Cloud TPU.
MODEL_DIR="gs://<path-to-model-directory>"  # Model artifacts directory for the training.
$ python3 train.py \
  --experiment=${EXPERIMENT} \
  --config_file_qat=<fill in>/edgetpu/nlp/experiments/downstream_tasks/mobilebert_edgetpu_xs.yaml \
  --config_file_qat=<fill in>/qat/nlp/configs/experiments/squad_v1_mobilebert_xs_qat_1gpu.yaml \
  --model_dir=${MODEL_DIR} \
  --tpu=$TPU_NAME \
  --params_override=${PARAMS_OVERRIDE}
  --mode=train
```

## Evaluation

Please Run below command for evaluation.

```shell
EXPERIMENT=bert/squad_qat  # Experiment type according to the subtask. Example: 'bert/squad_qat'
TPU_NAME="<tpu-name>"  # The name assigned while creating a Cloud TPU.
MODEL_DIR="gs://<path-to-model-directory>"  # Model artifacts directory for the training.
$ python3 train.py \
  --experiment=${EXPERIMENT} \
  --config_file_qat=<fill in>/edgetpu/nlp/experiments/downstream_tasks/mobilebert_edgetpu_xs.yaml \
  --config_file_qat=<fill in>/qat/nlp/configs/experiments/squad_v1_mobilebert_xs_qat_1gpu.yaml \
  --model_dir=${MODEL_DIR} \
  --tpu=$TPU_NAME \
  --mode=eval
```

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the terms of the **Apache License 2.0**.



