# TF Vision Model Garden

## Introduction
TF Vision model garden provides a large collection of baselines and checkpoints for image classification, object detection, and instance segmentation.


## Image Classification
### ImageNet Baselines
#### Models trained with vanilla settings:
* Models are trained from scratch with batch size 4096 and 1.6 initial learning rate.
* Linear warmup is applied for the first 5 epochs.
* Models trained with l2 weight regularization and ReLU activation.

| model        | resolution    | epochs  |  Top-1  |  Top-5  | download |
| ------------ |:-------------:|--------:|--------:|---------:|---------:|
| ResNet-50    | 224x224       |    90    | 76.1 | 92.9 | config |
| ResNet-50    | 224x224       |    200   | 77.1 | 93.5 | config |
| ResNet-101   | 224x224       |    200   | 78.3 | 94.2 | config |
| ResNet-152   | 224x224       |    200   | 78.7 | 94.3 | config |

#### Models trained with training features including:
* Label smoothing 0.1.
* Swish activation.

| model        | resolution    | epochs  |   Top-1  |  Top-5  | download |
| ------------ |:-------------:| ---------:|--------:|---------:|---------:|
| ResNet-50    | 224x224       |    200    | 78.1 | 93.9 | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnet50_tpu.yaml) |
| ResNet-101   | 224x224       |    200    | 79.1 | 94.5 | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnet101_tpu.yaml) |
| ResNet-152   | 224x224       |    200    | 79.4 | 94.7 | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnet152_tpu.yaml) |
| ResNet-200   | 224x224       |    200    | 79.9 | 94.8 | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnet200_tpu.yaml) |



## Object Detection and Instance Segmentation
### Common Settings and Notes
* We provide models based on two detection frameworks, [RetinaNet](https://arxiv.org/abs/1708.02002) or [Mask R-CNN](https://arxiv.org/abs/1703.06870), and two backbones, [ResNet-FPN](https://arxiv.org/abs/1612.03144) or [SpineNet](https://arxiv.org/abs/1912.05027).
* Models are all trained on COCO train2017 and evaluated on COCO val2017.
* Training details:
  * Models finetuned from ImageNet pretrained checkpoints adopt the 12 or 36 epochs schedule. Models trained from scratch adopt the 350 epochs schedule.
  * The default training data augmentation implements horizontal flipping and scale jittering with a random scale between [0.5, 2.0].
  * Unless noted, all models are trained with l2 weight regularization and ReLU activation.
  * We use batch size 256 and stepwise learning rate that decays at the last 30 and 10 epoch.
  * We use square image as input by resizing the long side of an image to the target size then padding the short side with zeros.

### COCO Object Detection Baselines
#### RetinaNet (ImageNet pretrained)
| backbone        | resolution    | epochs  | FLOPs (B)     | params (M) |  box AP |   download |
| ------------ |:-------------:| ---------:|-----------:|--------:|--------:|-----------:|
| R50-FPN      | 640x640       |    12    | 97.0 | 34.0 | 34.3 | config|
| R50-FPN      | 640x640       |    36    | 97.0 | 34.0 | 37.3 | config|

#### RetinaNet (Trained from scratch) with training features including:
* Stochastic depth with drop rate 0.2.
* Swish activation.

| backbone        | resolution    | epochs  | FLOPs (B)     | params (M) |  box AP |   download |
| ------------ |:-------------:| ---------:|-----------:|--------:|---------:|-----------:|
| SpineNet-49  | 640x640       |    500    | 85.4| 28.5 | 44.2 | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/retinanet/coco_spinenet49_tpu.yaml)|
| SpineNet-96  | 1024x1024     |    500    | 265.4 | 43.0 | 48.5 |  [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/retinanet/coco_spinenet96_tpu.yaml) |
| SpineNet-143 | 1280x1280     |    500    | 524.0 | 67.0 | 50.0 | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/retinanet/coco_spinenet143_tpu.yaml)|


### Instance Segmentation Baselines
#### Mask R-CNN (ImageNet pretrained)


#### Mask R-CNN (Trained from scratch)
| backbone        | resolution    | epochs  | FLOPs (B)  | params (M)  |  box AP |  mask AP  |   download |
| ------------ |:-------------:| ---------:|-----------:|--------:|--------:|-----------:|-----------:|
| SpineNet-49  | 640x640       |    350    | 215.7 | 40.8 | 42.6 | 37.9 | config |
