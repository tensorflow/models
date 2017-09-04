==========
Questions:
==========

a. Is Yolo a meta-architecture?
b. Depending on answer of question a., how to incorporate the trainable and non-trainable versions of YOLO in OD API?

===========
Milestones
===========


1. Implement non-trainable, off-the-shelf usable tiny-yolo 
2. Implement non-trainable, off-the-shelf usable yolov1
3. Implement non-trainable, off-the-shelf usable yolov2
4. Implement non-trainable, off-the-shelf usable yolov9000
5. Implement trainable versions of 1, 2 and 3

==============================================================
*YOLOv1*
==============================================================

General Notes for Clarification -

The grid size, S, is a hyper-parameter, but it isn't clear how the model architecture is to be adjusted for different grid sizes. The authors claim that their architecture is inspired by GoogLeNet, but this doesn't necessarily imply that the model can be replaced with any general purpose classifier, i.e. feature extractor. This is due to the fact that the final layer before the FC layers should have a dimension of S x S and feature extractors are not designed to correspond to a S X S grid that is overlaid over an image.

However, the number of detectable classes - C is indeed a hyperparameter and the YOLOv1 base network can be gracefully adjusted for any number of classes.

==================
Hyperparameters
==================

1. Number of bounding boxes per grid cell - B
2. Number of detectable classes           - C
3. Lambda_coord
4. Lambda_noobj
5. Number of epochs
6. Batch Size
7. Momentum
8. Decay
9. Learning Rate Schedule
10. Dropout layer rate in the first FC layer
11. Data Augmentations

