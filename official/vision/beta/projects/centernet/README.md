# Centernet

[![Paper](http://img.shields.io/badge/Paper-arXiv.1904.07850-B3181B?logo=arXiv)](https://arxiv.org/abs/1904.07850)

Centernet builds upon CornerNet, an anchor-free model for object detection.

Many other models, such as YOLO and RetinaNet, use anchor boxes. These anchor
boxes are predefined to be close to the aspect ratios and scales of the objects
in the training dataset. Anchor-based models do not predict the bounding boxes
of objects directly. They instead predict the location and size/shape
refinements to a predefined anchor box. The detection generator then computes
the final confidences, positions, and size of the detection.

CornerNet eliminates the need for anchor boxes. RetinaNet needs thousands of
anchor boxes in order to cover the most common ground truth boxes. This adds
unnecessary complexity to the model which slow down training and create
imbalances in positive and negative anchor boxes. Instead, CornerNet creates
heatmaps for each of the corners and pools them together in order to get the
final detection boxes for the objects. CenterNet removes even more complexity
by using the center instead of the corners, meaning that only one set of
heatmaps (one heatmap for each class) is needed to predict the object. CenterNet
proves that this can be done without a significant difference in accuracy.


## Environment setup

The code can be run on multiple GPUs or TPUs with different distribution
strategies. See the TensorFlow distributed training
[guide](https://www.tensorflow.org/guide/distributed_training) for an overview
of `tf.distribute`.

The code is compatible with TensorFlow 2.5+. See requirements.txt for all
prerequisites, and you can also install them using the following command. `pip
install -r ./official/requirements.txt`

## Training
To train the model on Coco, try the following command:

```
python3 -m official.vision.beta.projects.centernet.train \
  --mode=train_and_eval \
  --experiment=centernet_hourglass_coco \
  --model_dir={MODEL_DIR} \
  --config_file={CONFIG_FILE}
```

## Configurations

In the following table, we report the mAP measured on the `coco-val2017` set.

Backbone         | Config name                                     | mAP
:--------------- | :-----------------------------------------------| -------:
Hourglass-104    | `coco-centernet-hourglass-gpu.yaml`             | 40.01
Hourglass-104    | `coco-centernet-hourglass-tpu.yaml`             | 40.5

**Note:** `float16` (`bfloat16` for TPU) is used in the provided configurations.


## Cite

[Centernet](https://arxiv.org/abs/1904.07850):
```
@article{Zhou2019ObjectsAP,
  title={Objects as Points},
  author={Xingyi Zhou and Dequan Wang and Philipp Kr{\"a}henb{\"u}hl},
  journal={ArXiv},
  year={2019},
  volume={abs/1904.07850}
}
```

[CornerNet](https://arxiv.org/abs/1808.01244):
```
@article{Law2019CornerNetDO,
  title={CornerNet: Detecting Objects as Paired Keypoints},
  author={Hei Law and J. Deng},
  journal={International Journal of Computer Vision},
  year={2019},
  volume={128},
  pages={642-656}
}
```
