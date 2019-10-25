
# Adversarial logit pairing

This directory contains implementation of
[Adversarial logit pairing](https://arxiv.org/abs/1803.06373) paper as well as
few models pre-trained on ImageNet and Tiny ImageNet.

Please contact [Alexey Kurakin](https://github.com/AlexeyKurakin) regarding
this code.

## Pre-requesites

Code dependencies:

* TensorFlow 1.8 and Python 2.7 (other versions may work, but were not tested)
* [Abseil Python](https://github.com/abseil/abseil-py).
* Script which converts Tiny Imagenet dataset into TFRecord format also
  depends on [Pandas](https://pandas.pydata.org/).

## Datasets

To use this code you need to download datasets. You only need to download
those datasets which you're going to use. Following list of datasets is
supported:

* [ImageNet](http://www.image-net.org/). Follow
  [Preparing the datasets](https://github.com/tensorflow/models/tree/master/research/slim#Data)
  instructions in TF-Slim documentation to download and convert ImageNet dataset
  to TFRecord format.

* [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/).
  To obtain Tiny ImageNet dataset do following:

  ```
  # Download zip archive with TinyImagenet
  curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip

  # Extract archive
  unzip tiny-imagenet-200.zip

  # Convert dataset to TFRecord format
  mkdir tiny-imagenet-tfrecord
  python tiny_imagenet_converter/converter.py \
    --input_dir=tiny-imagenet-200 \
    --output_dir=tiny-imagenet-tfrecord
  ```

## Running the code

NOTE: Provided code supports distributed training on multiple machines,
and all provided checkpoints were trained in a distributed way. However it is
beyond the scope of this document to describe how to do distributed training.
Readed should refer to
[other material](https://www.tensorflow.org/deploy/distributed) to learn
about it.

### Training

Following command runs training:

```
# Following arguments has to be specified for training:
# - MAX_NUMBER_OF_TRAINING_STEPS - maximum number of training steps,
#     omit this flag or set it to -1 to have unlimited number of training steps.
# - MODEL_NAME - name of the model, now only "resnet_v2_50" is supported.
# - MOVING_AVG_DECAY - decay rate for exponential moving average of the
#     trainable variables. Training with exponential moving average usually
#     leads to better accuracy. Default of 0.9999. -1 disable exponential moving
#     average. Default works well, so typically you set it only if you want
#     to disable this feature.
# - HYPERPARAMETERS - string with hyperparameters,
#     see model_lib.py for full list of hyperparameters.
# - DATASET - dataset, either "imagenet" or "tiny_imagenet".
# - IMAGE_SIZE - size of the image (single number).
# - OUTPUT_DIRECTORY - directory where to write results.
# - IMAGENET_DIR - directory with ImageNet dataset in TFRecord format.
# - TINY_IMAGENET_DIR - directory with Tiny ImageNet dataset in TFRecord format.
#
# Note that only one of IMAGENET_DIR or TINY_IMAGENET_DIR has to be provided
# depending on which dataset you use.
#
python train.py \
  --max_steps="${MAX_NUMBER_OF_TRAINING_STEPS}" \
  --model_name="${MODEL_NAME}" \
  --moving_average_decay="${MOVING_AVG_DECAY}" \
  --hparams="${HYPERPARAMETERS}" \
  --dataset="${DATASET}" \
  --dataset_image_size="${IMAGE_SIZE}" \
  --output_dir="${OUTPUT_DIRECTORY}" \
  --imagenet_data_dir="${IMAGENET_DIR}" \
  --tiny_imagenet_data_dir="${TINY_IMAGENET_DIR}"
```

Full list of training hyperparameters could be found in `model_lib.py`.
These hyperparameters control learning rate schedule, optimizer, weight decay,
label smoothing and adversarial training.

Adversarial training is controlled by following hyperparameters:

* `train_adv_method` - method which is used to craft adversarial examples during
  training. Could be one of the following:

  * `clean` - perform regular training with clean examples;
  * `pgd_EPS_STEP_NITER` - use non targeted PGD with maximum size of
    perturbation equal to `EPS`, step size equal to `STEP`
    and number of iterations equal to `NITER`. Size of perturbation and step
    size are expected to be integers between 1 and 255.
  * `pgdll_EPS_STEP_NITER` - use targeted PGD, where target class is least
    likely prediction of the network.
  * `pgdrnd_EPS_STEP_NITER` - use targeted PGD, where target class is chosen
    randomly.

* `train_lp_weight` - weight of adversarial logit pairing loss. If zero or
  negarive, then no logit pairing is performed and training is done using
  mixed minibatch PGD. If positive then adversarial logit pairing term is added
  to the loss.

Below is example of how to run training with adversarial logit pairing on
ImageNet 64x64:

```
python train.py \
  --model_name="resnet_v2_50" \
  --hparams="train_adv_method=pgdll_16_2_10,train_lp_weight=0.5" \
  --dataset="imagenet" \
  --dataset_image_size=64 \
  --output_dir="/tmp/adv_train" \
  --imagenet_data_dir="${IMAGENET_DIR}"
```

### Fine tuning

Provided trainin script could be used to fine tune pre-trained checkpoint.
Following command does this:

```
# Fine tuning adds following additional arguments:
# - SCOPES_DO_NOT_LOAD_FROM_CHECKPOINT - comma separates list of scopes of
#     variables, which should not be loadeded from checkpoint (and default
#     initialization should be used instead).
#     SCOPES_DO_NOT_LOAD_FROM_CHECKPOINT should be either same or a subset of
#     LIST_OF_SCOPES_OF_TRAINABLE_VARS.
# - LIST_OF_SCOPES_OF_TRAINABLE_VARS - comma separated list of scopes of
#     trainable variables. Only variables which are prefixed with these scopes
#     will be trained.
# - PATH_TO_PRETRAINED_CHECKPOINT - directory with pretrained checkpoint which
#     is used as initialization for fine tuning.
#
python train.py \
  --max_steps="${MAX_NUMBER_OF_TRAINING_STEPS}" \
  --model_name="${MODEL_NAME}" \
  --moving_average_decay="${MOVING_AVG_DECAY}" \
  --hparams="${HYPERPARAMETERS}" \
  --dataset="${DATASET}" \
  --dataset_image_size="${IMAGE_SIZE}" \
  --output_dir="${OUTPUT_DIRECTORY}" \
  --imagenet_data_dir="${IMAGENET_DIR}" \
  --tiny_imagenet_data_dir="${TINY_IMAGENET_DIR}" \
  --finetune_exclude_pretrained_scopes="${SCOPES_DO_NOT_LOAD_FROM_CHECKPOINT}" \
  --finetune_trainable_scopes="${LIST_OF_SCOPES_OF_TRAINABLE_VARS}" \
  --finetune_checkpoint_path="${PATH_TO_PRETRAINED_CHECKPOINT}"
```

Below is an example of how to fine tune last few layers of the model on
Tiny Imagenet dataset:

```
python train.py \
  --model_name="resnet_v2_50" \
  --hparams="train_adv_method=pgdll_16_2_10,train_lp_weight=0.5,learning_rate=0.02" \
  --dataset="tiny_imagenet" \
  --dataset_image_size=64 \
  --output_dir="/tmp/adv_finetune" \
  --tiny_imagenet_data_dir="${TINY_IMAGENET_DIR}" \
  --finetune_exclude_pretrained_scopes="resnet_v2_50/logits" \
  --finetune_trainable_scopes="resnet_v2_50/logits,resnet_v2_50/postnorm" \
  --finetune_checkpoint_path="/tmp/adv_train"
```

### Evaluation

Following command runs evaluation:

```
# Following arguments should be provided for eval:
# - TRAINING_DIRECTORY - directory where training checkpoints are saved.
# - TRAINABLE_SCOPES - when loading checkpoint which was obtained by fine tuning
#     this argument should be the same as LIST_OF_SCOPES_OF_TRAINABLE_VARS
#     during training. Otherwise it should be empty.
#     This is needed to properly load exponential moving average variables.
#     If exponential moving averages are disabled then this flag could be
#     omitted.
# - EVAL_SUBDIR_NAME - name of the subdirectory inside TRAINING_DIRECTORY
#     where evaluation code will be saving event files.
# - DATASET - name of the dataset.
# - IMAGE_SIZE - size of the image in the dataset.
# - DATSET_SPLIT_NAME - name of the split in the dataset,
#     either 'train' or 'validation'. Default is 'validation'.
# - MODEL_NAME - name of the model.
# - MOVING_AVG_DECAY - decay rate for exponential moving average.
# - ADV_METHOD_FOR_EVAL - should be "clean" to evaluate on clean example or
#     description of the adversarial method to evaluate on adversarial examples.
# - HYPERPARAMETERS - hyperparameters, only "eval_batch_size" matters for eval
# - NUMBER_OF_EXAMPLES - how many examples from the dataset use for evaluation,
#     specify -1 to use all examples.
# - EVAL_ONCE - if True then evaluate only once, otherwise keep evaluation
#     running repeatedly on new checkpoints. Repeated evaluation might be useful
#     when running concurrent with training.
# - IMAGENET_DIR - directory with ImageNet dataset in TFRecord format.
# - TINY_IMAGENET_DIR - directory with Tiny ImageNet dataset in TFRecord format.
#
python eval.py \
  --train_dir="${TRAINING_DIRECTORY} \
  --trainable_scopes="${TRAINABLE_SCOPES}" \
  --eval_name="${EVAL_SUBDIR_NAME}" \
  --dataset="${DATASET}" \
  --dataset_image_size="${IMAGE_SIZE}" \
  --split_name="${DATSET_SPLIT_NAME}" \
  --model_name="${MODEL_NAME}" \
  --moving_average_decay="${MOVING_AVG_DECAY}" \
  --adv_method="${ADV_METHOD_FOR_EVAL}" \
  --hparams="${HYPERPARAMETERS}" \
  --num_examples="${NUMBER_OF_EXAMPLES}" \
  --eval_once="${EVAL_ONCE}" \
  --imagenet_data_dir="${IMAGENET_DIR}" \
  --tiny_imagenet_data_dir="${TINY_IMAGENET_DIR}"
```

Example of running evaluation on 10000 of clean examples from ImageNet
training set:

```
python eval.py \
  --train_dir=/tmp/adv_train \
  --dataset=imagenet \
  --dataset_image_size=64 \
  --split_name=train \
  --adv_method=clean \
  --hparams="eval_batch_size=50" \
  --num_examples=10000 \
  --eval_once=True \
  --imagenet_data_dir="${IMAGENET_DIR}"
```

Example of running evaluatin on adversarial images generated from Tiny ImageNet
validation set using fine-tuned checkpoint:

```
python eval.py \
  --train_dir=tmp/adv_finetune \
  --trainable_scopes="resnet_v2_50/logits,resnet_v2_50/postnorm" \
  --dataset=tiny_imagenet \
  --dataset_image_size=64 \
  --adv_method=pgdrnd_16_2_10 \
  --hparams="eval_batch_size=50" \
  --eval_once=True \
  --tiny_imagenet_data_dir="${TINY_IMAGENET_DIR}"
```

### Pre-trained models

Following set of pre-trained checkpoints released with this code:

| Model       |    Dataset   |  Accuracy on<br>clean images | Accuracy on<br>`pgdll_16_1_20` | Accuracy on<br>`pgdll_16_2_10` |
| ----------- | ------------ | --------------- | --------------------------- | -------------- |
| [Baseline ResNet-v2-50](http://download.tensorflow.org/models/adversarial_logit_pairing/imagenet64_base_2018_06_26.ckpt.tar.gz) | ImageNet 64x64 | 60.5% | 1.8% | 3.5% |
| [ALP-trained ResNet-v2-50](http://download.tensorflow.org/models/adversarial_logit_pairing/imagenet64_alp025_2018_06_26.ckpt.tar.gz) | ImageNet 64x64 | 55.7% | 27.5% | 27.8% |
| [Baseline ResNet-v2-50](http://download.tensorflow.org/models/adversarial_logit_pairing/tiny_imagenet_base_2018_06_26.ckpt.tar.gz) | Tiny ImageNet | 69.2% | 0.1% | 0.3% |
| [ALP-trained ResNet-v2-50](http://download.tensorflow.org/models/adversarial_logit_pairing/tiny_imagenet_alp05_2018_06_26.ckpt.tar.gz) | Tiny ImageNet | 72.0% | 41.3% | 40.8% |


* All provided checkpoints were initially trained with exponential moving
  average. However for ease of use they were re-saved without it.
  So to load and use provided checkpoints you need to specify
  `--moving_average_decay=-1` flag.
* All ALP models were trained with `pgdll_16_2_10` adversarial examples.
* All Tiny Imagenet models were obtained by fine tuning corresponding
  ImageNet 64x64 models. ALP-trained models were fine tuned with ALP.
