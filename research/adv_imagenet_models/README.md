# Adversarially trained ImageNet models

Pre-trained ImageNet models from the following papers:

* [Adversarial Machine Learning at Scale](https://arxiv.org/abs/1611.01236)
* [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204)

## Contact

Author: Alexey Kurakin,
github: [AlexeyKurakin](https://github.com/AlexeyKurakin)

## Pre-requesites and installation

Ensure that you have installed TensorFlow 1.1 or greater
([instructions](https://www.tensorflow.org/install/)).

You also need copy of ImageNet dataset if you want to run provided example.
Follow
[Preparing the dataset](https://github.com/tensorflow/models/tree/master/research/slim#Data)
instructions in TF-Slim library to get and preprocess ImageNet data.

## Available models

Following pre-trained models are available:

Network Architecture | Adversarial training | Checkpoint
---------------------|----------------------|----------------
Inception v3 | Step L.L. | [adv_inception_v3_2017_08_18.tar.gz](http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz)
Inception v3 | Step L.L. on ensemble of 3 models | [ens3_adv_inception_v3_2017_08_18.tar.gz](http://download.tensorflow.org/models/ens3_adv_inception_v3_2017_08_18.tar.gz)
Inception v3 | Step L.L. on ensemble of 4 models| [ens4_adv_inception_v3_2017_08_18.tar.gz](http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz)
Inception ResNet v2 | Step L.L. | [adv_inception_resnet_v2_2017_12_18.tar.gz](http://download.tensorflow.org/models/adv_inception_resnet_v2_2017_12_18.tar.gz)
Inception ResNet v2 | Step L.L. on ensemble of 3 models | [ens_adv_inception_resnet_v2_2017_08_18.tar.gz](http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz)

All checkpoints are compatible with
[TF-Slim](https://github.com/tensorflow/models/tree/master/research/slim)
implementation of Inception v3 and Inception Resnet v2.

## How to evaluate models on ImageNet test data

Python script `eval_on_adversarial.py` allow you to evaluate provided models
on white-box adversarial examples generated from ImageNet test set.

Usage is following:

```bash
# ${MODEL_NAME} - type of network architecture,
#     either "inception_v3" or "inception_resnet_v2"
# ${CHECKPOINT_PATH} - path to model checkpoint
# ${DATASET_DIR} - directory with ImageNet test set
# ${ADV_METHOD} - which method to use to generate adversarial images,
#   supported method:
#     "none" - use clean images from the dataset
#     "stepll" - one step towards least likely class method (StepLL),
#         see https://arxiv.org/abs/1611.01236 for details
#     "stepllnoise" - RAND+StepLL method from https://arxiv.org/abs/1705.07204
# ${ADV_EPS} - size of adversarial perturbation, ignored when method is none
python eval_on_adversarial.py \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --dataset_dir=${DATASET_DIR} \
  --batch_size=50 \
  --adversarial_method=${ADV_METHOD} \
  --adversarial_eps=${ADV_EPS}
```

Below is an example how to evaluate one of the models on RAND+StepLL adversarial
examples:

```bash
# Download checkpoint
CHECKPOINT_DIR=/tmp/checkpoints
mkdir ${CHECKPOINT_DIR}
wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar -xvf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
mv ens_adv_inception_resnet_v2.ckpt* ${CHECKPOINT_DIR}
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# Run evaluation
python eval_on_adversarial.py \
  --model_name=inception_v3 \
  --checkpoint_path=${CHECKPOINT_DIR}/ens_adv_inception_resnet_v2.ckpt \
  --dataset_dir=${DATASET_DIR} \
  --batch_size=50 \
  --adversarial_method=stepllnoise \
  --adversarial_eps=16
```
