# Formulating Few-shot Fine-tuning Towards Language Model Pre-training: A Pilot Study on Named Entity Recognition

**DISCLAIMER**: This implementation is still under development.

[![Paper](http://img.shields.io/badge/Paper-arXiv.2205.11799-B3181B?logo=arXiv)](https://arxiv.org/abs/2205.11799)

This repository is the official implementation of the following
paper.

* [Formulating Few-shot Fine-tuning Towards Language Model Pre-training: A Pilot Study on Named Entity Recognition](https://arxiv.org/abs/2205.11799)

## Description

FFF-NER is a training task for effective Few-shot Named Entity Recognition.
You can also refer to the author's [GitHub repository](https://github.com/ZihanWangKi/fffner).

## Maintainers

* Zihan Wang ([zihanwangki](https://github.com/zihanwangki))

## Requirements

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![tf-models-official PyPI](https://badge.fury.io/py/tf-models-official.svg)](https://badge.fury.io/py/tf-models-official)

## Training & Evaluation

It can run on Google Cloud Platform using Cloud TPU.
[Here](https://cloud.google.com/tpu/docs/how-to) is the instruction of using
Cloud TPU.

### Setup
You will need to first convert a pre-trained language model to the encoder
format we are using. The following command by default converts a base size
bert uncased model.
```shell
python3 utils/convert_checkpoint_tensorflow.py
```
Then, you will need to convert the dataset into a tf_record for training.
`utils/create_data.py` contains the script to do so. Example dataset and
dataset format can be found in the 
[official repo](https://github.com/ZihanWangKi/fffner/tree/main/dataset)
Suppose the dataset is stored as 
`/data/fffner_datasets/conll2003/few_shot_5_0.words` and
`/data/fffner_datasets/conll2003/few_shot_5_0.ner`,
where `/data/fffner_datasets/conll2003/` also contains the dataset
configuration and testing data,
then,
```
export PATH_TO_DATA_FOLDER=/data/fffner_datasets/
export DATASET_NAME=conll2003
export TRAINING_FOLD=few_shot_5_0
```
and
```shell
python3 utils/create_data.py $PATH_TO_DATA_FOLDER $DATASET_NAME $TRAINING_FOLD
```
creates the training fold. 

### Training

```shell
PATH_TO_TRAINING_RECORD=conll2003_few_shot_5_0.tf_record # path to the training record
PATH_TO_TESTING_RECORD=conll2003_test.tf_record # path to the evaluation record
TPU_NAME="<tpu-name>"  # The name assigned while creating a Cloud TPU
MODEL_DIR=/tmp/conll2003_ew_shot_5_0 # directory to store the experiment
# Now launch the experiment.
python3 -m official.projects.mosaic.train \
  --experiment=fffner/ner \
  --config_file=experiments/base_conll2003.yaml \
  --params_override="task.train_data.input_path=${PATH_TO_TRAINING_RECORD},task.validation_data.input_path=${PATH_TO_TESTING_RECORD},runtime.distribution_strategy=tpu"
  --mode=train_and_eval \
  --tpu=$TPU_NAME \
  --model_dir=$MODEL_DIR
```
## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the terms of the **Apache License 2.0**.

## Citation

If you want to cite this repository in your work, please consider citing the
paper.

```
@article{wang2022formulating,
  title={Formulating Few-shot Fine-tuning Towards Language Model Pre-training: A Pilot Study on Named Entity Recognition},
  author={Wang, Zihan and Zhao, Kewen and Wang, Zilong and Shang, Jingbo},
  journal={arXiv preprint arXiv:2205.11799},
  year={2022}
}
```
