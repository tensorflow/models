# MobileNet

In this folder we have provided the full model building code for
[MobileNetV1], [MobileNetV2] and [MobilenetV3] networks in [TF2.X with 
Keras] high level API. This project is an effort of migrating the MobileNet 
TF1.X implementation in [TensorFlow-Slim](../slim/nets/mobilenet) to TF2.X. 

In sum, this module consists of
- architectural definition of various MobileNet versions in 
[configs/archs.py](configs/archs.py).
- complete model building codes for
    - MobilenetV1 in [mobilenet_v1_model.py](mobilenet_v1_model.py)
    - MobilenetV2 in [mobilenet_v2_model.py](mobilenet_v2_model.py)
    - MobilenetV3 in [mobilenet_v3_model.py](mobilenet_v3_model.py)
- utilities helping load the pre-trained 
[TF1.X checkpoints](../slim/nets/mobilenet) into TF2.x Keras defined versions
    - MobilenetV1 TF1 loader in [mobilenet_v1_loader](tf1_loader/mobilenet_v1_loader.py) 
    - MobilenetV2 TF1 loader in [mobilenet_v2_loader](tf1_loader/mobilenet_v2_loader.py)
    - MobilenetV3 TF1 loader in [mobilenet_v3_loader](tf1_loader/mobilenet_v3_loader.py)
- a sample training pipeline for image classification problem defined in
[mobilenet_trainer.py](mobilenet_trainer.py), which also includes
    - pre-configured datasets: [ImageNet] and [Imagenette], 
    in [dataset.py](configs/dataset.py)
    - dataset loading and preprocessing pipeline defined 
    in [dataset_loader.py](dataset_loader.py)
- a set of bash scripts in folder [scripts/](scripts) to help launch training
jobs for various MobileNet versions.

## How to run training job
Run the following command from the root directory of the repository
```shell
python -m research.mobilenet.mobilenet_trainer \
  --model_name [MODEL_NAME] \
  --dataset_name [DATASET_NAME] \
  --data_dir [DATA_DIR] \
  --model_dir [MODEL_DIR]
```
where 
* --model_name: MobileNet version name: `mobilenet_v1`, `mobilenet_v2`, 
`mobilenet_v3_small` and `mobilenet_v3_large`. 
The default value is `mobilenet_v1`;
* --dataset_name: dataset name from train on: imagenette, imagenet2012, which
should be preconfigured in [dataset.py](configs/dataset.py). The default
value is `imagenette`;
* --data_dir: directory for training data. This is required if training data
is not directly downloaded from [TDFS]. The default value is `None`;
* --model_dir: the directory to save the model checkpoint.

There are more optional flags you can specify to modify the hyperparameters:
```shell
  --optimizer_name rmsprop \
  --learning_scheduler_name exponential \
  --op_momentum 0.9 \
  --op_decay_rate 0.9 \
  --lr 0.045 \
  --lr_decay_rate 0.94 \
  --lr_decay_epochs 2.5 \
  --label_smoothing 0.1 \
  --ma_decay_rate 0.1 \
  --dropout_rate 0.2 \
  --std_weight_decay 0.00002 \
  --truncated_normal_stddev 0.09 \
  --batch_norm_decay 0.9997 \
  --batch_size 128 \
  --epochs 30
```
where
* --optimizer_name: name of the optimizer used for training;
* --learning_scheduler_name: name of the learning rate scheduler;
* --op_momentum: optimizer's momentum;
* --op_decay_rate: optimizer discounting factor for gradient;
* --lr: base learning rate;
* --lr_decay_rate: magnitude of learning rate decay;
* --lr_decay_epochs: frequency of learning rate decay;
* --label_smoothing: amount of label smoothing;
* --ma_decay_rate: exponential moving average decay rate for trained parameters;
* --dropout_rate: dropout rate;
* --std_weight_decay: standard weight decay;
* --truncated_normal_stddev: the standard deviation of 
the truncated normal weight initializer;
* --batch_norm_decay: batch norm decay rate;
* --batch_size: batch size;
* --epochs: number of training epochs.

### Run training job using provided scripts
We have provided a set of bash scripts in folder [scripts/](scripts) to 
help launch training or testing jobs for various MobileNet versions.
- [train_mbnv1.sh](scripts/train_mbnv1.sh) for MobileNetV1
- [train_mbnv2.sh](scripts/train_mbnv2.sh) for MobileNetV2
- [train_mbnv3.sh](scripts/train_mbnv3.sh) for MobileNetV3 
- [start_tensorboard.sh](scripts/start_tensorboard.sh) for launching 
Tensorboard to monitor training process.

And the
`PYTHONPATH` has been properly set such that the script can be directly
launched within [scripts/](scripts). For example
```shell
sh train_mbnv1.sh train
```

## How to build various sizes of MobileNet
Width multiplier `alpha` is the key parameter to control the size of MobileNets.
For a given layer, and width multiplier, the number of input channels `M` becomes 
`alpha * M` and the number of output channels `N` becomes `alpha * N`. `alpha` 
is in the range of (0, 1] with typical settings of `1`, `0.75`, `0.5` and `0.25`. 
Note that `alpha=1` is the baseline MobileNet.

The architectural of various MobileNet versions are defined  
[configs/archs.py](configs/archs.py). One example is as follows
```python
class MobileNetV1Config(MobileNetConfig):
  """Configuration for the MobileNetV1 model.

    Attributes:
      name: name of the target model.
      blocks: base architecture

  """
  name: Text = 'MobileNetV1'
  width_multiplier: float = 1.0
  ......
```
Therefore, to train a desired version of MobileNet with customized architecture,
it is required to find the corresponding class definition in 
[configs/archs.py](configs/archs.py) and modify accordingly.

## How to load TF1 trained checkpoints
The utilities helping load the pre-trained 
[TF1.X checkpoints](../slim/nets/mobilenet) into TF2.x Keras versions are:
- MobilenetV1 TF1 loader in [mobilenet_v1_loader](tf1_loader/mobilenet_v1_loader.py) 
- MobilenetV2 TF1 loader in [mobilenet_v2_loader](tf1_loader/mobilenet_v2_loader.py)
- MobilenetV3 TF1 loader in [mobilenet_v3_loader](tf1_loader/mobilenet_v3_loader.py)

For each `mobilenet_vX_loader.py`, a model_load_function is defined with the same
signature as below:
```python
keras_model = [model_load_function](
    checkpoint_path=checkpoint_path,
    config=model_config)
```
where 
- [model_load_function] could be: `load_mobilenet_v1`, `load_mobilenet_v2`,
`load_mobilenet_v3_small`, `load_mobilenet_v3_large`;
- checkpoint_path: path of TF1 checkpoint;
- model_config: config used to build TF2 Keras model, which should be an
instance of `MobileNetV1Config` for MobileNetV1.

After loading the TF1 checkpoint into TF2 Keras model, we need to compile the
model before running evaluation. For example
```python
# compile model
if d_config.one_hot:
    loss_obj = tf.keras.losses.CategoricalCrossentropy()
else:
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

keras_model.compile(
    optimizer='rmsprop',
    loss=loss_obj,
    metrics=[mobilenet_trainer._get_metrics(one_hot=d_config.one_hot)['acc']])

# run evaluation
eval_result = keras_model.evaluate(eval_dataset)
```

Lastly, to save a TF2 compatible checkpoint from the restored Keras model, 
the following code can be used
```python
checkpoint = tf.train.Checkpoint(model=keras_model)
manager = tf.train.CheckpointManager(checkpoint,
                                     directory=save_path,
                                     max_to_keep=1)
manager.save()
```

We have already tested and reproduce the published results as follows:

| Checkpoint | Evaluation Top1 Accuracy |
|:-----------:|:----------:|
| [mobilenet_v1_1.0_224] | 0.710099995136261 |
| [mobilenet_v2_1.0_224] | 0.7184000015258789 |
| [mobilenet_v3_large_224_1.0_float] | 0.7521799802780151 |

## Example
See this [ipython notebook](notebooks/MNV1_Migrate_TF1_to_TF2_Keras.ipynb).

[MobileNetV1]: https://arxiv.org/abs/1704.04861
[MobilenetV2]: https://arxiv.org/abs/1801.04381
[MobilenetV3]: https://arxiv.org/abs/1905.02244
[TF2.X with Keras]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras
[TDFS]: https://www.tensorflow.org/datasets/catalog/overview
[ImageNet]: https://www.tensorflow.org/datasets/catalog/imagenet2012
[Imagenette]: https://www.tensorflow.org/datasets/catalog/imagenette
[mobilenet_v1_1.0_224]: http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz
[mobilenet_v2_1.0_224]: https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz 
[mobilenet_v3_large_224_1.0_float]: https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz