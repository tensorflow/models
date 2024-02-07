# TF-NLP Model Garden
## Introduction

The TF-NLP library provides a collection of scripts for training and
evaluating transformer-based models, on various tasks such as sentence
classification, question answering, and translation. Additionally, we provide
checkpoints of pretrained models which can be finetuned on downstream tasks.

⚠️ Disclaimer: Checkpoints are based on training with publicly available datasets.
Some datasets contain limitations, including non-commercial use limitations. Please review the terms and conditions made available by third parties before using
the datasets provided. Checkpoints are licensed under
[Apache 2.0](https://github.com/tensorflow/models/blob/master/LICENSE).

⚠️ Disclaimer: Datasets hyperlinked from this page are not owned or distributed
by Google. Such datasets are made available by third parties. Please review the
terms and conditions made available by the third parties before using the data.

### How to Train Models

Model Garden can be easily installed with
`pip install tf-models-nightly`. After installation, check out
[this instruction](https://github.com/tensorflow/models/blob/master/official/nlp/docs/train.md)
on how to train models with this codebase.


By default, the experiment runs on GPUs. To run on TPUs, one should overwrite
`runtime.distribution_strategy` and set the tpu address. See [RuntimeConfig](https://github.com/tensorflow/models/blob/master/official/core/config_definitions.py) for details.

In general, the experiments can run with the following command by setting the
corresponding `${TASK}`, `${TASK_CONFIG}`, `${MODEL_CONFIG}`.
```
EXPERIMENT=???
TASK_CONFIG=???
MODEL_CONFIG=???
EXRTRA_PARAMS=???
MODEL_DIR=???  # a-folder-to-hold-checkpoints-and-logs
python3 train.py \
  --experiment=${EXPERIMENT} \
  --mode=train_and_eval \
  --model_dir=${MODEL_DIR} \ 
  --config_file=${TASK_CONFIG} \
  --config_file=${MODEL_CONFIG} \
  --params_override=${EXRTRA_PARAMS}
``` 

* `EXPERIMENT` can be found under `configs/`
* `TASK_CONFIG` can be found under `configs/experiments/`
* `MODEL_CONFIG` can be found under `configs/models/`

#### Order of params override:
1. `train.py` looks up the registered `ExperimentConfig` with `${EXPERIMENT}`
2. Overrides params in `TaskConfig` in `${TASK_CONFIG}`
3. Overrides params `model` in `TaskConfig` with `${MODEL_CONFIG}`
4. Overrides any params in `ExperimentConfig` with `${EXTRA_PARAMS}`

Note that 
1. `${TASK_CONFIG}`, `${MODEL_CONFIG}`, `${EXTRA_PARAMS}` can be optional when EXPERIMENT default is enough.
2. `${TASK_CONFIG}`, `${MODEL_CONFIG}`, `${EXTRA_PARAMS}` are only guaranteed to be compatible to it's `${EXPERIMENT}` that defines it.

## Experiments

| NAME          | EXPERIMENT                     | TASK_CONFIG  | MODEL_CONFIG | EXRTRA_PARAMS |
| ----------------- | ------------------------ | ------- | -------- | ----------- |
| BERT-base GLUE/MNLI-matched finetune | [bert/sentence_prediction](https://github.com/tensorflow/models/blob/master/official/nlp/configs/finetuning_experiments.py) | [glue_mnli_matched.yaml](https://github.com/tensorflow/models/blob/master/official/nlp/configs/experiments/glue_mnli_matched.yaml) | [bert_en_uncased_base.yaml](https://github.com/tensorflow/models/blob/master/official/nlp/configs/models/bert_en_uncased_base.yaml) | <details> <summary>data and bert-base hub init</summary>task.train_data.input_path=/path-to-your-training-data,task.validation_data.input_path=/path-to-your-val-data,task.hub_module_url=https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4 </details> |
| BERT-base GLUE/MNLI-matched finetune | [bert/sentence_prediction](https://github.com/tensorflow/models/blob/master/official/nlp/configs/finetuning_experiments.py) | [glue_mnli_matched.yaml](https://github.com/tensorflow/models/blob/master/official/nlp/configs/experiments/glue_mnli_matched.yaml) | [bert_en_uncased_base.yaml](https://github.com/tensorflow/models/blob/master/official/nlp/configs/models/bert_en_uncased_base.yaml) | <details> <summary>data and bert-base ckpt init</summary>task.train_data.input_path=/path-to-your-training-data,task.validation_data.input_path=/path-to-your-val-data,task.init_checkpoint=gs://tf_model_garden/nlp/bert/uncased_L-12_H-768_A-12/bert_model.ckpt </details> |
| BERT-base SQuAD v1.1 finetune        | [bert/squad](https://github.com/tensorflow/models/blob/master/official/nlp/configs/finetuning_experiments.py)               | [squad_v1.yaml](https://github.com/tensorflow/models/blob/master/official/nlp/configs/experiments/squad_v1.yaml) | [bert_en_uncased_base.yaml](https://github.com/tensorflow/models/blob/master/official/nlp/configs/models/bert_en_uncased_base.yaml) | <details> <summary>data and bert-base hub init</summary>task.train_data.input_path=/path-to-your-training-data,task.validation_data.input_path=/path-to-your-val-data,task.hub_module_url=https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4 </details> |
|ALBERT-base SQuAD v1.1 finetune | [bert/squad](https://github.com/tensorflow/models/blob/master/official/nlp/configs/finetuning_experiments.py)   | [squad_v1.yaml](https://github.com/tensorflow/models/blob/master/official/nlp/configs/experiments/squad_v1.yaml) | [albert_base.yaml](https://github.com/tensorflow/models/blob/master/official/nlp/configs/models/albert_base.yaml)| <details> <summary>data and albert-base hub init</summary>task.train_data.input_path=/path-to-your-training-data,task.validation_data.input_path=/path-to-your-val-data,task.hub_module_url=https://tfhub.dev/tensorflow/albert_en_base/3 </details>|
| Transformer-large WMT14/en-de scratch |[wmt_transformer/large](https://github.com/tensorflow/models/blob/master/official/nlp/configs/wmt_transformer_experiments.py)|  | | <details> <summary>ende-32k sentencepiece</summary>task.sentencepiece_model_path='gs://tf_model_garden/nlp/transformer_wmt/ende_bpe_32k.model'</details> |


## Useful links

[How to Train Models](https://github.com/tensorflow/models/blob/master/official/nlp/docs/train.md)

[List of Pre-trained Models for finetuning](https://github.com/tensorflow/models/blob/master/official/nlp/docs/pretrained_models.md)

[How to Publish Models](https://github.com/tensorflow/models/blob/master/official/nlp/docs/tfhub.md)

[TensorFlow blog on Model Garden](https://blog.tensorflow.org/2020/03/introducing-model-garden-for-tensorflow-2.html).
