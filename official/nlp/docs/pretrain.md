# Model Garden NLP Pre-training Experiments

This user guide describes experiments on pre-training models in TF-NLP. Here we demonstrate how to run the pre-training experiments on TPU/GPU environment. Please refer to the corresponding experiment to get more detailed instructions.

## Pre-train a BERT from scratch

</details>

This example pre-trains a BERT model with Wikipedia and Books datasets used by
the original BERT paper.
The [BERT repo](https://github.com/tensorflow/models/blob/master/official/nlp/data/create_pretraining_data.py)
contains detailed information about the Wikipedia dump and
[BookCorpus](https://yknzhu.wixsite.com/mbweb). Of course, the pre-training
recipe is generic and you can apply the same recipe to your own corpus.

Please use the script
[`create_pretraining_data.py`](https://github.com/tensorflow/models/blob/master/official/nlp/data/create_pretraining_data.py)
which is essentially branched from [BERT research repo](https://github.com/google-research/bert)
to get processed pre-training data and it adapts to TF2 symbols and python3
compatibility.

Running the pre-training script requires an input and output directory, as well
as a vocab file. Note that `max_seq_length` will need to match the sequence
length parameter you specify when you run pre-training.

```shell
export WORKING_DIR='local disk or cloud location'
export BERT_DIR='local disk or cloud location'
python models/official/nlp/data/create_pretraining_data.py \
  --input_file=$WORKING_DIR/input/input.txt \
  --output_file=$WORKING_DIR/output/tf_examples.tfrecord \
  --vocab_file=$BERT_DIR/wwm_uncased_L-24_H-1024_A-16/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

Then, you can update the yaml configuration file, e.g.
`configs/experiments/wiki_books_pretrain.yaml` to specify your data paths and
update masking-related hyper parameters to match with your specification for 
the pretraining data. When your data have multiple shards, you can
use `*` to include multiple files.

To train different BERT sizes, you need to adjust:

```
model:
  cls_heads: [{activation: tanh, cls_token_idx: 0, dropout_rate: 0.1, inner_dim: 768, name: next_sentence, num_classes: 2}]
```

to match the hidden dimensions.

Then, you can start the training and evaluation jobs, which runs the
[`bert/pretraining`](https://github.com/tensorflow/models/blob/master/official/nlp/configs/pretraining_experiments.py#L51)
experiment:

```shell
export OUTPUT_DIR=gs://some_bucket/my_output_dir
export PARAMS=$PARAMS,runtime.distribution_strategy=tpu

python3 train.py \
 --experiment=bert/pretraining \
 --mode=train_and_eval \
 --model_dir=$OUTPUT_DIR \
 --config_file=configs/models/bert_en_uncased_base.yaml \
 --config_file=configs/experiments/wiki_books_pretrain.yaml \
 --tpu=${TPU_NAME} \
 --params_override=$PARAMS
```

## Pre-train BERT MLM with TFDS datasets

This example pre-trains a BERT MLM model with tensorflow_datasets (TFDS) and use tf.text for pre-processing using TPUs. Note that: only wikipedia english corpus is used.

You can start the training and evaluation jobs, which runs the
[`bert/text_wiki_pretraining`](https://github.com/tensorflow/models/blob/master/official/nlp/configs/pretraining_experiments.py#L88)
experiment:

```shell
export OUTPUT_DIR=gs://some_bucket/my_output_dir

# See the following link for more pre-trained checkpoints:
# https://github.com/tensorflow/models/blob/master/official/nlp/docs/pretrained_models.md
export BERT_DIR=~/cased_L-12_H-768_A-12

# Override the configurations by FLAGS. Alternatively, you can directly edit
# `configs/experiments/wiki_tfds_pretrain.yaml` to specify corresponding fields.
export PARAMS=$PARAMS,task.validation_data.vocab_file_path=$BERT_DIR/vocab.txt
export PARAMS=$PARAMS,task.train_data.vocab_file_path=$BERT_DIR/vocab.txt
export PARAMS=$PARAMS,runtime.distribution_strategy=tpu

python3 train.py \
 --experiment=bert/text_wiki_pretraining \
 --mode=train_and_eval \
 --model_dir=$OUTPUT_DIR \
 --config_file=configs/experiments/wiki_tfds_pretrain.yaml \
 --tpu=${TPU_NAME} \
 --params_override=$PARAMS
```