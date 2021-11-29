Code for Roformer.

Run with
```bash
python3 train.py \
  --experiment=roformer/pretraining \
  --config_file=experiments/roformer_base.yaml \
  --params_override="task.validation_data.input_path=gs://tf_model_garden/nlp/data/research_data/bert_pretrain/wikipedia.tfrecord-00000-of-00500,runtime.distribution_strategy=tpu" \
  --tpu=local \
  --model_dir=<OUTPUT_DIR> \
  --mode=train_and_eval
```