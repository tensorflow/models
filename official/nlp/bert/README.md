# BERT (Bidirectional Encoder Representations from Transformers)

The academic paper which describes BERT in detail and provides full results on a
number of tasks can be found here: https://arxiv.org/abs/1810.04805.

This repository contains TensorFlow 2.0 implementation for BERT.

N.B. This repository is under active development. Though we intend
to keep the top-level BERT Keras model interface stable, expect continued
changes to the training code, utility function interface and flags.

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

Our current released checkpoints are exactly the same as TF 1.x official BERT
repository, thus inside `BertConfig`, there is `backward_compatible=True`. We
are going to release new pre-trained checkpoints soon.

### Access to Pretrained Checkpoints

We provide checkpoints that are converted from [google-research/bert](https://github.com/google-research/bert),
in order to keep consistent with BERT paper.

*   **[`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/wwm_uncased_L-24_H-1024_A-16.tar.gz)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/wwm_cased_L-24_H-1024_A-16.tar.gz)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Base, Uncased`](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/uncased_L-12_H-768_A-12.tar.gz)**:
    12-layer, 768-hidden, 12-heads, 110M parameters
*   **[`BERT-Large, Uncased`](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/uncased_L-24_H-1024_A-16.tar.gz)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Base, Cased`](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/cased_L-12_H-768_A-12.tar.gz)**:
    12-layer, 768-hidden, 12-heads , 110M parameters
*   **[`BERT-Large, Cased`](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/cased_L-24_H-1024_A-16.tar.gz)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters

We recommend to host checkpoints on Google Cloud storage buckets when you use
Cloud GPU/TPU. For example, in the following tutorial, we use:

```shell
export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/tf_20/uncased_L-24_H-1024_A-16
```

### Restoring from Checkpoints

`tf.train.Checkpoint` is used to manage model checkpoints in TF 2.0. To restore
weights from provided pre-trained checkpoints, you can use the following code:

```python
init_checkpoint='the pretrained model checkpoint path.'
model=tf.keras.Model() # Bert pre-trained model as feature extractor.
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(init_checkpoint)
```

Checkpoints featuring native serialized Keras models
(i.e. model.load()/load_weights()) will be available soon.

## Set Up

```shell
export PYTHONPATH="$PYTHONPATH:/path/to/models"
```

Install `tf-nightly` to get latest updates:

```shell
pip install tf-nightly-gpu-2.0-preview
```

With TPU, GPU support is not necessary. First, you need to create a `tf-nigthly`
TPU with [cptu tool](https://github.com/tensorflow/tpu/tree/master/tools/ctpu):

```shell
ctpu up -name <instance name> --tf-version=”nightly”
```

Second, you need to install TF 2.0 `tf-night` on your VM:

```shell
pip install tf-nightly-2.0-preview
```

Warning: More details TPU-specific set-up instructions and tutorial should come
along with official TF 2.x release for TPU. Note that this repo is not officially
supported by Google Cloud TPU team yet.

## Process Datasets

### Pre-training

There is no change to generate pre-training data. Please use the script
[`create_pretraining_data.py`](https://github.com/google-research/bert/blob/master/create_pretraining_data.py)
inside [BERT research repo](https://github.com/google-research/bert) to get
processed pre-training data.


### Fine-tuning

To prepare the fine-tuning data for final model training, use the
[`create_finetuning_data.py`](./create_finetuning_data.py) script.  Resulting
datasets in `tf_record` format and training meta data should be later passed to
training or evaluation scripts. The task-specific arguments are described in
following sections:

* GLUE

Users can download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```shell
export GLUE_DIR=~/glue
export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/tf_20/uncased_L-24_H-1024_A-16

export TASK_NAME=MNLI
export OUTPUT_DIR=gs://some_bucket/datasets
python create_finetuning_data.py \
 --input_data_dir=${GLUE_DIR}/${TASK_NAME}/ \
 --vocab_file=${BERT_BASE_DIR}/vocab.txt \
 --train_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record \
 --eval_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data \
 --fine_tuning_task_type=classification --max_seq_length=128 \
 --classification_task_name=${TASK_NAME}
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
export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/tf_20/uncased_L-24_H-1024_A-16
export OUTPUT_DIR=gs://some_bucket/datasets

python create_finetuning_data.py \
 --squad_data_file=${SQUAD_DIR}/train-${SQUAD_VERSION}.json \
 --vocab_file=${BERT_BASE_DIR}/vocab.txt \
 --train_data_output_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_meta_data \
 --fine_tuning_task_type=squad --max_seq_length=384
```

## Fine-tuning with BERT

### Cloud GPUs and TPUs

* Cloud Storage

The unzipped pre-trained model files can also be found in the Google Cloud
Storage folder `gs://cloud-tpu-checkpoints/bert/tf_20`. For example:

```shell
export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/tf_20/uncased_L-24_H-1024_A-16
export MODEL_DIR=gs://some_bucket/my_output_dir
```

Currently, users are able to access to `tf-nightly` TPUs and the following TPU
script should run with `tf-nightly`.

* GPU -> TPU

Just add the following flags to `run_classifier.py` or `run_squad.py`:

```shell
  --strategy_type=tpu
  --tpu=grpc://${TPU_IP_ADDRESS}:8470
```

### Sentence and Sentence-pair Classification Tasks

This example code fine-tunes `BERT-Large` on the Microsoft Research Paraphrase
Corpus (MRPC) corpus, which only contains 3,600 examples and can fine-tune in a
few minutes on most GPUs.

We use the `BERT-Large` (uncased_L-24_H-1024_A-16) as an example throughout the
workflow.
For GPU memory of 16GB or smaller, you may try to use `BERT-Base`
(uncased_L-12_H-768_A-12).

```shell
export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/tf_20/uncased_L-24_H-1024_A-16
export MODEL_DIR=gs://some_bucket/my_output_dir
export GLUE_DIR=gs://some_bucket/datasets
export TASK=MRPC

python run_classifier.py \
  --mode='train_and_eval' \
  --input_meta_data_path=${GLUE_DIR}/${TASK}_meta_data \
  --train_data_path=${GLUE_DIR}/${TASK}_train.tf_record \
  --eval_data_path=${GLUE_DIR}/${TASK}_eval.tf_record \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --train_batch_size=4 \
  --eval_batch_size=4 \
  --steps_per_loop=1 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --model_dir=${MODEL_DIR} \
  --strategy_type=mirror
```

To use TPU, you only need to switch distribution strategy type to `tpu` with TPU
information and use remote storage for model checkpoints.

```shell
export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/tf_20/uncased_L-24_H-1024_A-16
export TPU_IP_ADDRESS='???'
export MODEL_DIR=gs://some_bucket/my_output_dir
export GLUE_DIR=gs://some_bucket/datasets

python run_classifier.py \
  --mode='train_and_eval' \
  --input_meta_data_path=${GLUE_DIR}/${TASK}_meta_data \
  --train_data_path=${GLUE_DIR}/${TASK}_train.tf_record \
  --eval_data_path=${GLUE_DIR}/${TASK}_eval.tf_record \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --model_dir=${MODEL_DIR} \
  --strategy_type=tpu \
  --tpu=grpc://${TPU_IP_ADDRESS}:8470
```

### SQuAD 1.1

The Stanford Question Answering Dataset (SQuAD) is a popular question answering
benchmark dataset. See more in [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/).

We use the `BERT-Large` (uncased_L-24_H-1024_A-16) as an example throughout the
workflow.
For GPU memory of 16GB or smaller, you may try to use `BERT-Base`
(uncased_L-12_H-768_A-12).

```shell
export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/tf_20/uncased_L-24_H-1024_A-16
export SQUAD_DIR=gs://some_bucket/datasets
export MODEL_DIR=gs://some_bucket/my_output_dir
export SQUAD_VERSION=v1.1

python run_squad.py \
  --input_meta_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_meta_data \
  --train_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
  --predict_file=${SQUAD_DIR}/dev-v1.1.json \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=4 \
  --predict_batch_size=4 \
  --learning_rate=8e-5 \
  --num_train_epochs=2 \
  --model_dir=${MODEL_DIR} \
  --strategy_type=mirror
```

To use TPU, you need switch distribution strategy type to `tpu` with TPU
information.

```shell
export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/tf_20/uncased_L-24_H-1024_A-16
export TPU_IP_ADDRESS='???'
export MODEL_DIR=gs://some_bucket/my_output_dir
export SQUAD_DIR=gs://some_bucket/datasets
export SQUAD_VERSION=v1.1

python run_squad.py \
  --input_meta_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_meta_data \
  --train_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
  --predict_file=${SQUAD_DIR}/dev-v1.1.json \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --learning_rate=8e-5 \
  --num_train_epochs=2 \
  --model_dir=${MODEL_DIR} \
  --strategy_type=tpu \
  --tpu=grpc://${TPU_IP_ADDRESS}:8470
```

The dev set predictions will be saved into a file called predictions.json in the
model_dir:

```shell
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ./squad/predictions.json
```


