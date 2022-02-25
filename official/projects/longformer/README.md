# Longformer: The Long-Document Transformer

## Modifications from Huggingface's Implementation
All models require a `global_attention_size` specified in the config, 
setting a global attention for all first `global_attention_size` tokens in any sentence. 
Individual different global attention sizes for sentences are not supported.
This setting allows running on TPUs where tensor sizes have to be determined.

`_get_global_attn_indices` in `longformer_attention.py` contains how the new global attention indices are specified.
Changed all `tf.cond` to if confiditions, since global attention is specified in the start now.

`sentence_prediction_with_checkpoint_convert.py` now contains a `initial_parameters_from_pk` parameter that
specified a pk file containing all pre-trained weights from a pytorch longformer, which can be loaded into the 
tf model.
The pk file can be generated from `utils/get_parameters_from_pretrained_pytorch_checkpoint.py`.
There is also a `longformer_tokenizer_to_tfrecord.py` that transformers pytorch longformer tokenized data to tf_records.

## Steps to Fine-tune on MNLI
#### Prepare the pre-trained checkpoint
Option 1. Use our saved checkpoint of `allenai/longformer-base-4096` stored in cloud storage
```bash
gsutil cp gs://model-garden-ucsd-zihan/allenai.pk allenai_longformer-base-4096.pk
```
Option 2. Create it directly
```bash
python3 utils/get_parameters_from_pretrained_pytorch_checkpoint.py
```
#### [Optional] Prepare the input file
```bash
python3 longformer_tokenizer_to_tfrecord.py
```
#### Training
Here, we use the training data of MNLI that were uploaded to the cloud storage, you can replace it with the input files you generated.
```bash
TRAIN_DATA=task.train_data.input_path=gs://model-garden-ucsd-zihan/longformer_allenai_mnli_train.tf_record,task.validation_data.input_path=gs://model-garden-ucsd-zihan/longformer_allenai_mnli_eval.tf_record
PYTHONPATH=/path/to/model/garden \
    python3 train.py \
    --experiment=longformer/glue \
    --config_file=experiments/glue_mnli_allenai.yaml \
    --params_override="${TRAIN_DATA},runtime.distribution_strategy=tpu,task.initial_parameters_from_pk=allenai_longformer-base-4096.pk" \
    --tpu=local \
    --model_dir=/path/to/outputdir \
    --mode=train_and_eval 
```
This should take an hour or two to run, and give a performance of ~86.