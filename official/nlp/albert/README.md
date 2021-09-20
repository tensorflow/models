# ALBERT (ALBERT: A Lite BERT for Self-supervised Learning of Language Representations)

**WARNING**: We are on the way to deprecate this directory.
We will add documentation in `nlp/docs` to use the new code in `nlp/modeling`.

The academic paper which describes ALBERT in detail and provides full results on
a number of tasks can be found here: https://arxiv.org/abs/1909.11942.

This repository contains TensorFlow 2.x implementation for ALBERT.

## Contents
  * [Contents](#contents)
  * [Pre-trained Models](#pre-trained-models)
    * [Restoring from Checkpoints](#restoring-from-checkpoints)
  * [Set Up](#set-up)
  * [Process Datasets](#process-datasets)
  * [Fine-tuning with BERT](#fine-tuning-with-bert)
    * [Cloud GPUs and TPUs](#cloud-gpus-and-tpus)
    * [Sentence and Sentence-pair Classification Tasks](#sentence-and-sentence-pair-classification-tasks)
    * [SQuAD 1.1](#squad-1.1)


## Pre-trained Models

We released both checkpoints and tf.hub modules as the pretrained models for
fine-tuning. They are TF 2.x compatible and are converted from the ALBERT v2
checkpoints released in TF 1.x official ALBERT repository
[google-research/albert](https://github.com/google-research/albert)
in order to keep consistent with ALBERT paper.

Our current released checkpoints are exactly the same as TF 1.x official ALBERT
repository.

### Access to Pretrained Checkpoints

Pretrained checkpoints can be found in the following links:

**Note: We implemented ALBERT using Keras functional-style networks in [nlp/modeling](../modeling).
ALBERT V2 models compatible with TF 2.x checkpoints are:**

*   **[`ALBERT V2 Base`](https://storage.googleapis.com/cloud-tpu-checkpoints/albert/checkpoints/albert_v2_base.tar.gz)**:
    12-layer, 768-hidden, 12-heads, 12M parameters
*   **[`ALBERT V2 Large`](https://storage.googleapis.com/cloud-tpu-checkpoints/albert/checkpoints/albert_v2_large.tar.gz)**:
    24-layer, 1024-hidden, 16-heads, 18M parameters
*   **[`ALBERT V2 XLarge`](https://storage.googleapis.com/cloud-tpu-checkpoints/albert/checkpoints/albert_v2_xlarge.tar.gz)**:
    24-layer, 2048-hidden, 32-heads, 60M parameters
*   **[`ALBERT V2 XXLarge`](https://storage.googleapis.com/cloud-tpu-checkpoints/albert/checkpoints/albert_v2_xxlarge.tar.gz)**:
    12-layer, 4096-hidden, 64-heads, 235M parameters

We recommend to host checkpoints on Google Cloud storage buckets when you use
Cloud GPU/TPU.

### Restoring from Checkpoints

`tf.train.Checkpoint` is used to manage model checkpoints in TF 2. To restore
weights from provided pre-trained checkpoints, you can use the following code:

```python
init_checkpoint='the pretrained model checkpoint path.'
model=tf.keras.Model() # Bert pre-trained model as feature extractor.
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(init_checkpoint)
```

Checkpoints featuring native serialized Keras models
(i.e. model.load()/load_weights()) will be available soon.

### Access to Pretrained hub modules.

Pretrained tf.hub modules in TF 2.x SavedModel format can be found in the
following links:

*   **[`ALBERT V2 Base`](https://tfhub.dev/tensorflow/albert_en_base/1)**:
    12-layer, 768-hidden, 12-heads, 12M parameters
*   **[`ALBERT V2 Large`](https://tfhub.dev/tensorflow/albert_en_large/1)**:
    24-layer, 1024-hidden, 16-heads, 18M parameters
*   **[`ALBERT V2 XLarge`](https://tfhub.dev/tensorflow/albert_en_xlarge/1)**:
    24-layer, 2048-hidden, 32-heads, 60M parameters
*   **[`ALBERT V2 XXLarge`](https://tfhub.dev/tensorflow/albert_en_xxlarge/1)**:
    12-layer, 4096-hidden, 64-heads, 235M parameters

## Set Up

```shell
export PYTHONPATH="$PYTHONPATH:/path/to/models"
```

Install `tf-nightly` to get latest updates:

```shell
pip install tf-nightly-gpu
```

With TPU, GPU support is not necessary. First, you need to create a `tf-nightly`
TPU with [ctpu tool](https://github.com/tensorflow/tpu/tree/master/tools/ctpu):

```shell
ctpu up -name <instance name> --tf-version=”nightly”
```

Second, you need to install TF 2 `tf-nightly` on your VM:

```shell
pip install tf-nightly
```

Warning: More details TPU-specific set-up instructions and tutorial should come
along with official TF 2.x release for TPU. Note that this repo is not
officially supported by Google Cloud TPU team yet until TF 2.1 released.

## Process Datasets

### Pre-training

Pre-train ALBERT using TF2.x will come soon.
For now, please use [ALBERT research repo](https://github.com/google-research/ALBERT)
to pretrain the model and convert the checkpoint to TF2.x compatible ones using
[tf2_albert_encoder_checkpoint_converter.py](tf2_albert_encoder_checkpoint_converter.py).



### Fine-tuning

To prepare the fine-tuning data for final model training, use the
[`../data/create_finetuning_data.py`](../data/create_finetuning_data.py) script.
Note that different from BERT models that use word piece tokenzer,
ALBERT models employ sentence piece tokenizer. So the FLAG tokenizer_impl has
to be set to 'sentence_piece'.
Resulting datasets in `tf_record` format and training meta data should be later
passed to training or evaluation scripts. The task-specific arguments are
described in following sections:

* GLUE

Users can download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```shell
export GLUE_DIR=~/glue
export ALBERT_DIR=gs://cloud-tpu-checkpoints/albert/checkpoints/albert_v2_base

export TASK_NAME=MNLI
export OUTPUT_DIR=gs://some_bucket/datasets
python ../data/create_finetuning_data.py \
 --input_data_dir=${GLUE_DIR}/${TASK_NAME}/ \
 --sp_model_file=${ALBERT_DIR}/30k-clean.model \
 --train_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record \
 --eval_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data \
 --fine_tuning_task_type=classification --max_seq_length=128 \
 --classification_task_name=${TASK_NAME} \
 --tokenization=SentencePiece
```

* SQUAD

The [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) contains
detailed information about the SQuAD datasets and evaluation.

The necessary files can be found here:

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)
*   [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
*   [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
*   [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

```shell
export SQUAD_DIR=~/squad
export SQUAD_VERSION=v1.1
export ALBERT_DIR=gs://cloud-tpu-checkpoints/albert/checkpoints/albert_v2_base
export OUTPUT_DIR=gs://some_bucket/datasets

python ../data/create_finetuning_data.py \
 --squad_data_file=${SQUAD_DIR}/train-${SQUAD_VERSION}.json \
 --sp_model_file=${ALBERT_DIR}/30k-clean.model \
 --train_data_output_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_meta_data \
 --fine_tuning_task_type=squad --max_seq_length=384 \
 --tokenization=SentencePiece
```

## Fine-tuning with ALBERT

### Cloud GPUs and TPUs

* Cloud Storage

The unzipped pre-trained model files can also be found in the Google Cloud
Storage folder `gs://cloud-tpu-checkpoints/albert/checkpoints`. For example:

```shell
export ALBERT_DIR=gs://cloud-tpu-checkpoints/albert/checkpoints/albert_v2_base
export MODEL_DIR=gs://some_bucket/my_output_dir
```

Currently, users are able to access to `tf-nightly` TPUs and the following TPU
script should run with `tf-nightly`.

* GPU -> TPU

Just add the following flags to `run_classifier.py` or `run_squad.py`:

```shell
  --distribution_strategy=tpu
  --tpu=grpc://${TPU_IP_ADDRESS}:8470
```

### Sentence and Sentence-pair Classification Tasks

This example code fine-tunes `albert_v2_base` on the Microsoft Research
Paraphrase Corpus (MRPC) corpus, which only contains 3,600 examples and can
fine-tune in a few minutes on most GPUs.

We use the `albert_v2_base` as an example throughout the
workflow.


```shell
export ALBERT_DIR=gs://cloud-tpu-checkpoints/albert/checkpoints/albert_v2_base
export MODEL_DIR=gs://some_bucket/my_output_dir
export GLUE_DIR=gs://some_bucket/datasets
export TASK=MRPC

python run_classifier.py \
  --mode='train_and_eval' \
  --input_meta_data_path=${GLUE_DIR}/${TASK}_meta_data \
  --train_data_path=${GLUE_DIR}/${TASK}_train.tf_record \
  --eval_data_path=${GLUE_DIR}/${TASK}_eval.tf_record \
  --bert_config_file=${ALBERT_DIR}/albert_config.json \
  --init_checkpoint=${ALBERT_DIR}/bert_model.ckpt \
  --train_batch_size=4 \
  --eval_batch_size=4 \
  --steps_per_loop=1 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --model_dir=${MODEL_DIR} \
  --distribution_strategy=mirrored
```

Alternatively, instead of specifying `init_checkpoint`, you can specify
`hub_module_url` to employ a pretraind BERT hub module, e.g.,
` --hub_module_url=https://tfhub.dev/tensorflow/albert_en_base/1`.

To use TPU, you only need to switch distribution strategy type to `tpu` with TPU
information and use remote storage for model checkpoints.

```shell
export ALBERT_DIR=gs://cloud-tpu-checkpoints/albert/checkpoints/albert_v2_base
export TPU_IP_ADDRESS='???'
export MODEL_DIR=gs://some_bucket/my_output_dir
export GLUE_DIR=gs://some_bucket/datasets

python run_classifier.py \
  --mode='train_and_eval' \
  --input_meta_data_path=${GLUE_DIR}/${TASK}_meta_data \
  --train_data_path=${GLUE_DIR}/${TASK}_train.tf_record \
  --eval_data_path=${GLUE_DIR}/${TASK}_eval.tf_record \
  --bert_config_file=$ALBERT_DIR/albert_config.json \
  --init_checkpoint=$ALBERT_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --model_dir=${MODEL_DIR} \
  --distribution_strategy=tpu \
  --tpu=grpc://${TPU_IP_ADDRESS}:8470
```

### SQuAD 1.1

The Stanford Question Answering Dataset (SQuAD) is a popular question answering
benchmark dataset. See more in [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/).

We use the `albert_v2_base` as an example throughout the
workflow.

```shell
export ALBERT_DIR=gs://cloud-tpu-checkpoints/albert/checkpoints/albert_v2_base
export SQUAD_DIR=gs://some_bucket/datasets
export MODEL_DIR=gs://some_bucket/my_output_dir
export SQUAD_VERSION=v1.1

python run_squad.py \
  --input_meta_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_meta_data \
  --train_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
  --predict_file=${SQUAD_DIR}/dev-v1.1.json \
  --sp_model_file=${ALBERT_DIR}/30k-clean.model \
  --bert_config_file=$ALBERT_DIR/albert_config.json \
  --init_checkpoint=$ALBERT_DIR/bert_model.ckpt \
  --train_batch_size=4 \
  --predict_batch_size=4 \
  --learning_rate=8e-5 \
  --num_train_epochs=2 \
  --model_dir=${MODEL_DIR} \
  --distribution_strategy=mirrored
```

Similarily, you can replace `init_checkpoint` FLAGS with `hub_module_url` to
specify a hub module path.

To use TPU, you need switch distribution strategy type to `tpu` with TPU
information.

```shell
export ALBERT_DIR=gs://cloud-tpu-checkpoints/albert/checkpoints/albert_v2_base
export TPU_IP_ADDRESS='???'
export MODEL_DIR=gs://some_bucket/my_output_dir
export SQUAD_DIR=gs://some_bucket/datasets
export SQUAD_VERSION=v1.1

python run_squad.py \
  --input_meta_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_meta_data \
  --train_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
  --predict_file=${SQUAD_DIR}/dev-v1.1.json \
  --sp_model_file=${ALBERT_DIR}/30k-clean.model \
  --bert_config_file=$ALBERT_DIR/albert_config.json \
  --init_checkpoint=$ALBERT_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --learning_rate=8e-5 \
  --num_train_epochs=2 \
  --model_dir=${MODEL_DIR} \
  --distribution_strategy=tpu \
  --tpu=grpc://${TPU_IP_ADDRESS}:8470
```

The dev set predictions will be saved into a file called predictions.json in the
model_dir:

```shell
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ./squad/predictions.json
```
