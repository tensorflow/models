# TF Vision Model Garden

## Introduction
TF Vision model garden provides a large collection of baselines and checkpoints for image classification, object detection, and instance segmentation.


## Image Classification
### Common Settings and Notes
* We provide ImageNet checkpoints for [ResNet](https://arxiv.org/abs/1512.03385) models.
* Training details:
  * All models are trained from scratch for 90 epochs with batch size 4096 and 1.6 initial stepwise decay learning rate.
  * Unless noted, all models are trained with l2 weight regularization and ReLU activation.

### ImageNet Baselines
| model        | resolution    | epochs  | FLOPs (B)    | params (M)  |  Top-1  |  Top-5  | download |
| ------------ |:-------------:| ---------:|-----------:|--------:|--------:|---------:|---------:|
| ResNet-50    | 224x224       |    90    | 4.1 | 25.6 | 76.1 | 92.9 | config |



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

#### RetinaNet (Trained from scratch)
| backbone        | resolution    | epochs  | FLOPs (B)     | params (M) |  box AP |   download |
| ------------ |:-------------:| ---------:|-----------:|--------:|---------:|-----------:|
| SpineNet-49  | 640x640       |    350    | 85.4| 28.5 | 42.4| config|
| SpineNet-96  | 1024x1024     |    350    | 265.4 | 43.0 | 46.0 |  config |
| SpineNet-143 | 1280x1280     |    350    | 524.0 | 67.0 | 46.8 |config|


### Instance Segmentation Baselines
#### Mask R-CNN (ImageNet pretrained)


#### Mask R-CNN (Trained from scratch)
| backbone        | resolution    | epochs  | FLOPs (B)  | params (M)  |  box AP |  mask AP  |   download |
| ------------ |:-------------:| ---------:|-----------:|--------:|--------:|-----------:|-----------:|
| SpineNet-49  | 640x640       |    350    | 215.7 | 40.8 | 42.6 | 37.9 | config |
