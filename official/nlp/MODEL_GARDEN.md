# TF-NLP Model Garden

## Introduction

This TF-NLP library provides a collection of scripts for the training and
evaluation of transformer-based models, on various tasks such as sentence
classification, question answering, and translation. Additionally, we provide
checkpoints of pretrained models which can be finetuned on downstream tasks.

### How to Train Models

Model Garden can be easily installed using PIP
(`pip install tf-models-nightly`). After installation, check out
[this instruction](https://github.com/tensorflow/models/blob/master/official/nlp/docs/train.md)
on how to train models with this codebase.

## Available Tasks

There are two available model configs (we will add more) under
`configs/experiments/`:

| Dataset           | Task                     | Config  | Example command |
| ----------------- | ------------------------ | ------- | ---- |
| GLUE/MNLI-matched | bert/sentence_prediction | [glue_mnli_matched.yaml](https://github.com/tensorflow/models/blob/master/official/nlp/configs/experiments/glue_mnli_matched.yaml) | <details> <summary>finetune BERT-base on this task</summary> PARAMS=runtime.distribution_strategy=mirrored<br/>PARAMS=${PARAMS},task.train_data.input_path=/path-to-your-training-data/<br/>PARAMS=${PARAMS},task.hub_module_url=https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4<br/><br/>python3 train.py \\<br/>  --experiment=bert/sentence_prediction \\<br/>  --mode=train \\<br/>  --model_dir=/a-folder-to-hold-checkpoints-and-logs/ \\<br/>  --config_file=configs/models/bert_en_uncased_base.yaml \\<br/>  --config_file=configs/experiments/glue_mnli_matched.yaml \\<br/>  --params_override=${PARAMS}</details> |
| SQuAD v1.1        | bert/squad               | [squad_v1.yaml](https://github.com/tensorflow/models/blob/master/official/nlp/configs/experiments/squad_v1.yaml) | <details> <summary>finetune BERT-base on this task</summary> PARAMS=runtime.distribution_strategy=mirrored<br/>PARAMS=${PARAMS},task.train_data.input_path=/path-to-your-training-data/<br/>PARAMS=${PARAMS},task.hub_module_url=https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4<br/><br/>python3 train.py \\<br/>  --experiment=bert/squad \\<br/>  --mode=train \\<br/>  --model_dir=/a-folder-to-hold-checkpoints-and-logs/ \\<br/>  --config_file=configs/models/bert_en_uncased_base.yaml \\<br/>  --config_file=configs/experiments/squad_v1.yaml \\<br/>  --params_override=${PARAMS}</details> |

One example on how to use the config file: if you want to work on the SQuAD
question answering task, set
`--config_file=configs/experiments/squad_v1.yaml` and
`--experiment=bert/squad`
as arguments to `train.py`.

## Available Model Configs

There are two available model configs (we will add more) under
`configs/models/`:

| Model        | Config  | Pretrained checkpoint & Vocabulary | TF-HUB SavedModel | Example command |
| ------------ | ------- | ---------------------------------- | ----------------- | --------------- |
| BERT-base    | [bert_en_uncased_base.yaml](https://github.com/tensorflow/models/blob/master/official/nlp/configs/models/bert_en_uncased_base.yaml) | [uncased_L-12_H-768_A-12](https://storage.googleapis.com/tf_model_garden/nlp/bert/v3/uncased_L-12_H-768_A-12.tar.gz) | [uncased_L-12_H-768_A-12](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/) | <details> <summary>finetune on SQuAD v1.1</summary> PARAMS=runtime.distribution_strategy=mirrored<br/>PARAMS=${PARAMS},task.train_data.input_path=/path-to-your-training-data/<br/>PARAMS=${PARAMS},task.hub_module_url=https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4<br/><br/>python3 train.py \\<br/>  --experiment=bert/squad \\<br/>  --mode=train \\<br/>  --model_dir=/a-folder-to-hold-checkpoints-and-logs/ \\<br/>  --config_file=configs/models/bert_en_uncased_base.yaml \\<br/>  --config_file=configs/experiments/squad_v1.yaml \\<br/>  --params_override=${PARAMS}</details> |
| ALBERT-base  | [albert_base.yaml](https://github.com/tensorflow/models/blob/master/official/nlp/configs/models/albert_base.yaml) | [albert_en_base](https://storage.googleapis.com/tf_model_garden/nlp/albert/albert_base.tar.gz) | [albert_en_base](https://tfhub.dev/tensorflow/albert_en_base/3) | <details> <summary>finetune on SQuAD v1.1</summary> PARAMS=runtime.distribution_strategy=mirrored<br/>PARAMS=${PARAMS},task.train_data.input_path=/path-to-your-training-data/<br/>PARAMS=${PARAMS},task.hub_module_url=https://tfhub.dev/tensorflow/albert_en_base/3<br/><br/>python3 train.py \\<br/>  --experiment=bert/squad \\<br/>  --mode=train \\<br/>  --model_dir=/a-folder-to-hold-checkpoints-and-logs/ \\<br/>  --config_file=configs/models/albert_base.yaml \\<br/>  --config_file=configs/experiments/squad_v1.yaml \\<br/>  --params_override=${PARAMS}</details> |

One example on how to use the config file: if you want to train an ALBERT-base
model, set `--config_file=configs/models/albert_base.yaml` as an argument to
`train.py`.

## Useful links

[How to Train Models](https://github.com/tensorflow/models/blob/master/official/nlp/docs/train.md)

[List of Pretrained Models](https://github.com/tensorflow/models/blob/master/official/nlp/docs/pretrained_models.md)

[How to Publish Models](https://github.com/tensorflow/models/blob/master/official/nlp/docs/tfhub.md)

[TensorFlow blog on Model Garden](https://blog.tensorflow.org/2020/03/introducing-model-garden-for-tensorflow-2.html).
