Code for Roformer.

Run with
```bash
DATA_PATH=???
OUTPUT_DIR=???
python3 train.py \
  --experiment=roformer/pretraining \
  --config_file=experiments/roformer_base.yaml \
  --params_override="task.validation_data.input_path=${DATA_PATH},runtime.distribution_strategy=tpu" \
  --tpu=local \
  --model_dir=${OUTPUT_DIR} \
  --mode=train_and_eval
```
