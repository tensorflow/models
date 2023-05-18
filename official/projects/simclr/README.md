# Simple Framework for Contrastive Learning

[![Paper](http://img.shields.io/badge/Paper-arXiv.2002.05709-B3181B?logo=arXiv)](https://arxiv.org/abs/2002.05709)
[![Paper](http://img.shields.io/badge/Paper-arXiv.2006.10029-B3181B?logo=arXiv)](https://arxiv.org/abs/2006.10029)

<div align="center">
  <img width="50%" alt="SimCLR Illustration" src="https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s1600/image4.gif">
</div>
<div align="center">
  An illustration of SimCLR (from <a href="https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html">our blog here</a>).
</div>

## Environment setup

The code can be run on multiple GPUs or TPUs with different distribution
strategies. See the TensorFlow distributed training
[guide](https://www.tensorflow.org/guide/distributed_training) for an overview
of `tf.distribute`.

The code is compatible with TensorFlow 2.4+. See requirements.txt for all
prerequisites, and you can also install them using the following command. `pip
install -r ./official/requirements.txt`

## Pretraining
To pretrain the model on Imagenet, try the following command:

```
python3 -m official.projects.simclr.train \
  --mode=train_and_eval \
  --experiment=simclr_pretraining \
  --model_dir={MODEL_DIR} \
  --config_file={CONFIG_FILE}
```

An example of the config file can be found [here](./configs/experiments/imagenet_simclr_pretrain_gpu.yaml)


## Semi-supervised learning and fine-tuning the whole network

You can access 1% and 10% ImageNet subsets used for semi-supervised learning via
[tensorflow datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012_subset).
You can also find image IDs of these subsets in `imagenet_subsets/`.

To fine-tune the whole network, refer to the following command:

```
python3 -m official.projects.simclr.train \
  --mode=train_and_eval \
  --experiment=simclr_finetuning \
  --model_dir={MODEL_DIR} \
  --config_file={CONFIG_FILE}
```

An example of the config file can be found [here](./configs/experiments/imagenet_simclr_finetune_gpu.yaml).

## Cite

[SimCLR paper](https://arxiv.org/abs/2002.05709):

```
@article{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020}
}
```

[SimCLRv2 paper](https://arxiv.org/abs/2006.10029):

```
@article{chen2020big,
  title={Big Self-Supervised Models are Strong Semi-Supervised Learners},
  author={Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2006.10029},
  year={2020}
}
```
