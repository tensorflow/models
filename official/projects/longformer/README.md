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

## Running
```bash
python utils/get_parameters_from_pretrained_pytorch_checkpoint.py
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
