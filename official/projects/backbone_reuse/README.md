# Proper Reuse of Image Classification Features Improves Object Detection

This project brings the backbone freezing training approach into the Mask-RCNN
architecture. Please see the paper for more details
\([arxiv](https://arxiv.org/abs/2204.00484) - selected for oral presentation at
CVPR 2022\).

### Training Mask-Rcnn Models with backbone frozen.

#### Freezing Resnet-RS-101 checkpoint (ImageNet pretrained).

1.  Download the ResNet-RS-101 pretrained checkpoint from
    [TF-Vision Model Garden](https://github.com/tensorflow/models/tree/master/official/vision#resnet-rs-models-trained-with-various-settings),
    \([checkpoint](https://storage.cloud.google.com/tf_model_garden/vision/resnet-rs/resnet-rs-101-i192.tar.gz)\)

2.  Config files used in our Resnet-101 ablations are included in the
    [configs folder](https://github.com/tensorflow/models/tree/master/official/projects/backbone_reuse/configs/experiments/faster_rcnn).
    Select one according to the target architecture (FPN, NASFPN, NASFPN +
    Cascades) and training schedule preference (shorter--72 epochs, or longer
    --600 epochs).

3.  Change the config flag `init_checkpoint` to point to the downloaded file.

You are all set. Follow the standard TFVision Mask-Rcnn training pipeline to
complete the training.

#### How does it work?

The config files set the task's flag `freeze_backbone: true`. This flag prevents
the pretrained backbone weights from being updated during the downstream model
training.

## Citation

```
@inproceedings{vasconcelos2022backbonefreeze,
      title = {Proper Reuse of Image Classification Features Improves Object Detection},
      author = {Cristina Vasconcelos and Vighnesh Birodkar and Vincent Dumoulin},
      booktitle={CVPR}
      year={2022},
```
