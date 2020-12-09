# Model Garden NLP Common Training Driver

[train.py](train.py) is the common training driver that supports multiple
NLP tasks (e.g., pre-training, GLUE and SQuAD fine-tuning etc) and multiple
models (e.g., BERT, ALBERT, MobileBERT etc).

## Experiment Configuration

[train.py] is driven by configs defined by the [ExperimentConfig](../core/config_definitions.py)
including configurations for `task`, `trainer` and `runtime`. The pre-defined
NLP related [ExperimentConfig](../core/config_definitions.py) can be found in
[configs/experiment_configs.py](configs/experiment_configs.py).

## Experiment Registry

We use an [experiment registry](../core/exp_factory.py) to build a mapping
between experiment type to experiment configuration instance. For example,
[configs/finetuning_experiments.py](configs/finetuning_experiments.py)
registers `bert/sentence_prediction` and `bert/squad` experiments. User can use
`--experiment` FLAG to invoke a registered experiment configuration,
e.g., `--experiment=bert/sentence_prediction`.

## Overriding Configuration via Yaml and FLAGS

The registered experiment configuration can be overridden by one or
multiple Yaml files provided by `--config_file` FLAG. For example:

```shell
--config_file=configs/experiments/glue_mnli_matched.yaml \
--config_file=configs/models/bert_en_uncased_base.yaml
```

In addition, experiment configuration can be further overriden by
`params_override` FLAG. For example:

```shell
 --params_override=task.train_data.input_path=/some/path,task.hub_module_url=/some/tfhub
```

## Run on Cloud TPUs

Next, we will describe how to run the [train.py](train.py) on Cloud TPUs.

### Setup
First, you need to create a `tf-nightly` TPU with
[ctpu tool](https://github.com/tensorflow/tpu/tree/master/tools/ctpu):

```shell
export TPU_NAME=YOUR_TPU_NAME
ctpu up -name $TPU_NAME --tf-version=nightly --tpu-size=YOUR_TPU_SIZE --project=YOUR_PROJECT
```

and then install Model Garden and required dependencies:

```shell
git clone https://github.com/tensorflow/models.git
export PYTHONPATH=$PYTHONPATH:/path/to/models
pip3 install --user -r official/requirements.txt
```

### Fine-tuning Sentence Classification with BERT from TF-Hub

This example fine-tunes BERT-base from TF-Hub on the the Multi-Genre Natural
Language Inference (MultiNLI) corpus using TPUs.

Firstly, you can prepare the fine-tuning data using
[`data/create_finetuning_data.py`](data/create_finetuning_data.py) script.
Resulting training and evaluation datasets in `tf_record` format will be later
passed to [train.py](train.py).

Then you can execute the following commands to start the training and evaluation
job.

```shell
export INPUT_DATA_DIR=gs://some_bucket/datasets
export OUTPUT_DIR=gs://some_bucket/my_output_dir

# See tfhub BERT collection for more tfhub models:
# https://tfhub.dev/google/collections/bert/1
export BERT_HUB_URL=https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3

# Override the configurations by FLAGS. Alternatively, you can directly edit
# `configs/experiments/glue_mnli_matched.yaml` to specify corresponding fields.
export PARAMS=task.train_data.input_path=$INPUT_DATA_DIR/mnli_train.tf_record
export PARAMS=$PARAMS,task.validation_data.input_path=$INPUT_DATA_DIR/mnli_eval.tf_record
export PARAMS=$PARAMS,task.hub_module_url=$BERT_HUB_URL
export PARAMS=$PARAMS,runtime.distribution_strategy=tpu

python3 train.py \
 --experiment=bert/sentence_prediction \
 --mode=train_and_eval \
 --model_dir=$OUTPUT_DIR \
 --config_file=configs/experiments/glue_mnli_matched.yaml \
 --tfhub_cache_dir=$OUTPUT_DIR/hub_cache \
 --tpu=${TPU_NAME} \
 --params_override=$PARAMS

```

You can monitor the training progress in the console and find the output
models in `$OUTPUT_DIR`.

Note: More examples about pre-training and fine-tuning will come soon.
