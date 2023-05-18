# Long Range Arena

This repository contains TensorFlow 2.x implementation for Long Range Arena
Tasks, including baseline for Transformer and Linformer. The codebase is adapted
from (https://github.com/google-research/long-range-arena).

## Training on LRA Tasks
Example job script to train Transformer on ListOPs task:

```bash
TRAIN_DATA=task.train_data.input_path=gs://model-garden-ucsd-zihan/lra_listops_train.tf_record,task.validation_data.input_path=gs://model-garden-ucsd-zihan/lra_listops_eval.tf_record

PYTHONPATH=[/PATH/TO/MODEL_GARDEN] \
    python3 train.py \
    --experiment=transformer/lra_listops \
    --config_file=../experiments/lra_listops.yaml \
    --params_override="${TRAIN_DATA},runtime.distribution_strategy=tpu" \
    --tpu=local \
    --model_dir=[OUTPUT_DIR] \
    --mode=train_and_eval
```

To train Linformer on ListOPs task:

```bash
TRAIN_DATA=task.train_data.input_path=gs://model-garden-ucsd-zihan/lra_listops_train.tf_record,task.validation_data.input_path=gs://model-garden-ucsd-zihan/lra_listops_eval.tf_record

PYTHONPATH=[/PATH/TO/MODEL_GARDEN] \
    python3 train.py \
    --experiment=linformer/lra_listops \
    --config_file=../experiments/lra_listops_linformer.yaml \
    --params_override="${TRAIN_DATA},runtime.distribution_strategy=tpu" \
    --tpu=local \
    --model_dir=[OUTPUT_DIR] \
    --mode=train_and_eval
```

## Data Paths and Experiment Configs
Dataset Paths are listed below:

|            | Path                                                                    |
|------------|-------------------------------------------------------------------------|
| ListOps    | gs://model-garden-ucsd-zihan/lra_listops_[train/eval/test].tf_record    |
| IMDB       | gs://model-garden-ucsd-zihan/lra_imdb_[train/eval/test].tf_record       |
| AAN        | gs://model-garden-ucsd-zihan/lra_aan_[train/eval/test].tf_record        |
| CIFAR10    | gs://model-garden-ucsd-zihan/lra_cifar_[train/eval/test].tf_record      |
| Pathfinder | gs://model-garden-ucsd-zihan/lra_pathfinder_[train/eval/test].tf_record |

Experiment Configs can be found in the `experiments` subfolder.


