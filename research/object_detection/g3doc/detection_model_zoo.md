# Tensorflow detection model zoo

We provide a collection of detection models pre-trained on the [COCO
dataset](http://mscoco.org), the [Kitti dataset](http://www.cvlibs.net/datasets/kitti/),
the [Open Images dataset](https://github.com/openimages/dataset), the
[AVA v2.1 dataset](https://research.google.com/ava/) and the
[iNaturalist Species Detection Dataset](https://github.com/visipedia/inat_comp/blob/master/2017/README.md#bounding-boxes).
These models can be useful for out-of-the-box inference if you are interested in
categories already in those datasets. They are also useful for initializing your
models when training on novel datasets.

In the table below, we list each such pre-trained model including:

* a model name that corresponds to a config file that was used to train this
  model in the `samples/configs` directory,
* a download link to a tar.gz file containing the pre-trained model,
* model speed --- we report running time in ms per 600x600 image (including all
  pre and post-processing), but please be
  aware that these timings depend highly on one's specific hardware
  configuration (these timings were performed using an Nvidia
  GeForce GTX TITAN X card) and should be treated more as relative timings in
  many cases. Also note that desktop GPU timing does not always reflect mobile
  run time. For example Mobilenet V2 is faster on mobile devices than Mobilenet
  V1, but is slightly slower on desktop GPU.
* detector performance on subset of the COCO validation set or Open Images test split as measured by the dataset-specific mAP measure.
  Here, higher is better, and we only report bounding box mAP rounded to the
  nearest integer.
* Output types (`Boxes`, and `Masks` if applicable )

You can un-tar each tar.gz file via, e.g.,:

```
tar -xzvf ssd_mobilenet_v1_coco.tar.gz
```

Inside the un-tar'ed directory, you will find:

* a graph proto (`graph.pbtxt`)
* a checkpoint
  (`model.ckpt.data-00000-of-00001`, `model.ckpt.index`, `model.ckpt.meta`)
* a frozen graph proto with weights baked into the graph as constants
  (`frozen_inference_graph.pb`) to be used for out of the box inference
    (try this out in the Jupyter notebook!)
* a config file (`pipeline.config`) which was used to generate the graph.  These
  directly correspond to a config file in the
  [samples/configs](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)) directory but often with a modified score threshold.  In the case
  of the heavier Faster R-CNN models, we also provide a version of the model
  that uses a highly reduced number of proposals for speed.

Some remarks on frozen inference graphs:

* If you try to evaluate the frozen graph, you may find performance numbers for
  some of the models to be slightly lower than what we report in the below
  tables.  This is because we discard detections with scores below a
  threshold (typically 0.3) when creating the frozen graph.  This corresponds
  effectively to picking a point on the precision recall curve of
  a detector (and discarding the part past that point), which negatively impacts
  standard mAP metrics.
* Our frozen inference graphs are generated using the
  [v1.8.0](https://github.com/tensorflow/tensorflow/tree/v1.8.0)
  release version of Tensorflow and we do not guarantee that these will work
  with other versions; this being said, each frozen inference graph can be
  regenerated using your current version of Tensorflow by re-running the
  [exporter](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md),
  pointing it at the model directory as well as the corresponding config file in
  [samples/configs](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).


## COCO-trained models

| Model name  | Speed (ms) | COCO mAP[^1] | Outputs |
| ------------ | :--------------: | :--------------: | :-------------: |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | 30 | 21 | Boxes |
| [ssd_mobilenet_v1_0.75_depth_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz) | 26 | 18 | Boxes |
| [ssd_mobilenet_v1_quantized_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz) | 29 | 18 | Boxes |
| [ssd_mobilenet_v1_0.75_depth_quantized_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz) | 29 | 16 | Boxes |
| [ssd_mobilenet_v1_ppn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz) | 26 | 20 | Boxes |
| [ssd_mobilenet_v1_fpn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) | 56 | 32 | Boxes |
| [ssd_resnet_50_fpn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) | 76 | 35 | Boxes |
| [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) | 31 | 22 | Boxes |
| [ssdlite_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz) | 27 | 22 | Boxes |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz) | 42 | 24 | Boxes |
| [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) | 58 | 28 | Boxes |
| [faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) | 89 | 30 | Boxes |
| [faster_rcnn_resnet50_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz) | 64 |  | Boxes |
| [rfcn_resnet101_coco](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz)  | 92 | 30 | Boxes |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz) | 106 | 32 | Boxes |
| [faster_rcnn_resnet101_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz) | 82 |  | Boxes |
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz) | 620 | 37 | Boxes |
| [faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz) | 241 |  | Boxes |
| [faster_rcnn_nas](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz) | 1833 | 43 | Boxes |
| [faster_rcnn_nas_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz) | 540 |  | Boxes |
| [mask_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz) | 771 | 36 | Masks |
| [mask_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz) | 79 | 25 | Masks |
| [mask_rcnn_resnet101_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz) | 470 | 33 | Masks |
| [mask_rcnn_resnet50_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz) | 343 | 29 | Masks |

Note: The asterisk (☆) at the end of model name indicates that this model supports TPU training.

Note: If you download the tar.gz file of quantized models and un-tar, you will get different set of files - a checkpoint, a config file and tflite frozen graphs (txt/binary).

## Kitti-trained models

Model name                                                                                                                                                        | Speed (ms) | Pascal mAP@0.5 | Outputs
----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :-------------: | :-----:
[faster_rcnn_resnet101_kitti](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz) | 79  | 87              | Boxes

## Open Images-trained models

Model name                                                                                                                                                        | Speed (ms) | Open Images mAP@0.5[^2] | Outputs
----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :-------------: | :-----:
[faster_rcnn_inception_resnet_v2_atrous_oid](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz) | 727 | 37              | Boxes
[faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz) | 347  |               | Boxes

## iNaturalist Species-trained models

Model name                                                                                                                                                        | Speed (ms) | Pascal mAP@0.5 | Outputs
----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :-------------: | :-----:
[faster_rcnn_resnet101_fgvc](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_fgvc_2018_07_19.tar.gz) | 395  | 58              | Boxes
[faster_rcnn_resnet50_fgvc](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_fgvc_2018_07_19.tar.gz) | 366  | 55             | Boxes


## AVA v2.1 trained models

Model name                                                                                                                                                        | Speed (ms) | Pascal mAP@0.5 | Outputs
----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :-------------: | :-----:
[faster_rcnn_resnet101_ava_v2.1](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_ava_v2.1_2018_04_30.tar.gz) | 93  | 11              | Boxes


[^1]: See [MSCOCO evaluation protocol](http://cocodataset.org/#detections-eval).

[^2]: This is PASCAL mAP with a slightly different way of true positives computation: see [Open Images evaluation protocol](evaluation_protocols.md#open-images).

