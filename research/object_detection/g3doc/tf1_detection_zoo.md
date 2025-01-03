# TensorFlow 1 Detection Model Zoo

[![TensorFlow 1.15](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

We provide a collection of detection models pre-trained on the
[COCO dataset](http://cocodataset.org), the
[Kitti dataset](http://www.cvlibs.net/datasets/kitti/), the
[Open Images dataset](https://storage.googleapis.com/openimages/web/index.html),
the [AVA v2.1 dataset](https://research.google.com/ava/) the
[iNaturalist Species Detection Dataset](https://github.com/visipedia/inat_comp/blob/master/2017/README.md#bounding-boxes)
and the
[Snapshot Serengeti Dataset](http://lila.science/datasets/snapshot-serengeti).
These models can be useful for out-of-the-box inference if you are interested in
categories already in those datasets. They are also useful for initializing your
models when training on novel datasets.

In the table below, we list each such pre-trained model including:

*   a model name that corresponds to a config file that was used to train this
    model in the `samples/configs` directory,
*   a download link to a tar.gz file containing the pre-trained model,
*   model speed --- we report running time in ms per 600x600 image (including
    all pre and post-processing), but please be aware that these timings depend
    highly on one's specific hardware configuration (these timings were
    performed using an Nvidia GeForce GTX TITAN X card) and should be treated
    more as relative timings in many cases. Also note that desktop GPU timing
    does not always reflect mobile run time. For example Mobilenet V2 is faster
    on mobile devices than Mobilenet V1, but is slightly slower on desktop GPU.
*   detector performance on subset of the COCO validation set, Open Images test
    split, iNaturalist test split, or Snapshot Serengeti LILA.science test
    split. as measured by the dataset-specific mAP measure. Here, higher is
    better, and we only report bounding box mAP rounded to the nearest integer.
*   Output types (`Boxes`, and `Masks` if applicable )

You can un-tar each tar.gz file via, e.g.,:

```
tar -xzvf ssd_mobilenet_v1_coco.tar.gz
```

Inside the un-tar'ed directory, you will find:

*   a graph proto (`graph.pbtxt`)
*   a checkpoint (`model.ckpt.data-00000-of-00001`, `model.ckpt.index`,
    `model.ckpt.meta`)
*   a frozen graph proto with weights baked into the graph as constants
    (`frozen_inference_graph.pb`) to be used for out of the box inference (try
    this out in the Jupyter notebook!)
*   a config file (`pipeline.config`) which was used to generate the graph.
    These directly correspond to a config file in the
    [samples/configs](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs))
    directory but often with a modified score threshold. In the case of the
    heavier Faster R-CNN models, we also provide a version of the model that
    uses a highly reduced number of proposals for speed.
*   Mobile model only: a TfLite file (`model.tflite`) that can be deployed on
    mobile devices.

Some remarks on frozen inference graphs:

*   If you try to evaluate the frozen graph, you may find performance numbers
    for some of the models to be slightly lower than what we report in the below
    tables. This is because we discard detections with scores below a threshold
    (typically 0.3) when creating the frozen graph. This corresponds effectively
    to picking a point on the precision recall curve of a detector (and
    discarding the part past that point), which negatively impacts standard mAP
    metrics.
*   Our frozen inference graphs are generated using the
    [v1.12.0](https://github.com/tensorflow/tensorflow/tree/v1.12.0) release
    version of TensorFlow; this being said, each frozen inference graph can be
    regenerated using your current version of TensorFlow by re-running the
    [exporter](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md),
    pointing it at the model directory as well as the corresponding config file
    in
    [samples/configs](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).

## COCO-trained models

Model name                                                                                                                                                                                    | Speed (ms) | COCO mAP[^1] | Outputs
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------: | :----------: | :-----:
[ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)                                                                       | 30         | 21           | Boxes
[ssd_mobilenet_v1_0.75_depth_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz)                                | 26         | 18           | Boxes
[ssd_mobilenet_v1_quantized_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz)                                  | 29         | 18           | Boxes
[ssd_mobilenet_v1_0.75_depth_quantized_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz)            | 29         | 16           | Boxes
[ssd_mobilenet_v1_ppn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz)                         | 26         | 20           | Boxes
[ssd_mobilenet_v1_fpn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)                         | 56         | 32           | Boxes
[ssd_resnet_50_fpn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)                             | 76         | 35           | Boxes
[ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)                                                                       | 31         | 22           | Boxes
[ssd_mobilenet_v2_quantized_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz)                                           | 29         | 22           | Boxes
[ssdlite_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)                                                               | 27         | 22           | Boxes
[ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)                                                                       | 42         | 24           | Boxes
[faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)                                                       | 58         | 28           | Boxes
[faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)                                                               | 89         | 30           | Boxes
[faster_rcnn_resnet50_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz)                                     | 64         |              | Boxes
[rfcn_resnet101_coco](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz)                                                                           | 92         | 30           | Boxes
[faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)                                                             | 106        | 32           | Boxes
[faster_rcnn_resnet101_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz)                                   | 82         |              | Boxes
[faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)                           | 620        | 37           | Boxes
[faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz) | 241        |              | Boxes
[faster_rcnn_nas](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz)                                                                              | 1833       | 43           | Boxes
[faster_rcnn_nas_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz)                                               | 540        |              | Boxes
[mask_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)                               | 771        | 36           | Masks
[mask_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz)                                                           | 79         | 25           | Masks
[mask_rcnn_resnet101_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz)                                                   | 470        | 33           | Masks
[mask_rcnn_resnet50_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz)                                                     | 343        | 29           | Masks

Note: The asterisk (☆) at the end of model name indicates that this model
supports TPU training.

Note: If you download the tar.gz file of quantized models and un-tar, you will
get different set of files - a checkpoint, a config file and tflite frozen
graphs (txt/binary).

### Mobile models

Model name                                                                                                                                                                | Pixel 1 Latency (ms) | COCO mAP | Outputs
------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------: | :------: | :-----:
[ssd_mobiledet_cpu_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz)                                     | 113                  | 24.0     | Boxes
[ssd_mobilenet_v2_mnasfpn_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18.tar.gz) | 183                  | 26.6     | Boxes
[ssd_mobilenet_v3_large_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz)                                       | 119                  | 22.6     | Boxes
[ssd_mobilenet_v3_small_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_small_coco_2020_01_14.tar.gz)                                       | 43                   | 15.4     | Boxes

### Pixel4 Edge TPU models

Model name                                                                                                                                    | Pixel 4 Edge TPU Latency (ms) | COCO mAP (fp32/uint8) | Outputs
--------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------: | :-------------------: | :-----:
[ssd_mobiledet_edgetpu_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz) | 6.9                           | 25.9/25.6             | Boxes
[ssd_mobilenet_edgetpu_coco](https://storage.cloud.google.com/mobilenet_edgetpu/checkpoints/ssdlite_mobilenet_edgetpu_coco_quant.tar.gz)      | 6.6                           | -/24.3                | Boxes

### Pixel4 DSP models

Model name                                                                                                                            | Pixel 4 DSP Latency (ms) | COCO mAP (fp32/uint8) | Outputs
------------------------------------------------------------------------------------------------------------------------------------- | :----------------------: | :-------------------: | :-----:
[ssd_mobiledet_dsp_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19.tar.gz) | 12.3                     | 28.9/28.8             | Boxes

## Kitti-trained models

Model name                                                                                                                          | Speed (ms) | Pascal mAP@0.5 | Outputs
----------------------------------------------------------------------------------------------------------------------------------- | :--------: | :------------: | :-----:
[faster_rcnn_resnet101_kitti](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz) | 79         | 87             | Boxes

## Open Images-trained models

Model name                                                                                                                                                                                    | Speed (ms) | Open Images mAP@0.5[^2] | Outputs
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------: | :---------------------: | :-----:
[faster_rcnn_inception_resnet_v2_atrous_oidv2](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz)                           | 727        | 37                      | Boxes
[faster_rcnn_inception_resnet_v2_atrous_lowproposals_oidv2](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz) | 347        |                         | Boxes
[facessd_mobilenet_v2_quantized_open_image_v4](http://download.tensorflow.org/models/object_detection/facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz) [^3]                       | 20         | 73 (faces)              | Boxes

Model name                                                                                                                                                             | Speed (ms) | Open Images mAP@0.5[^4] | Outputs
---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------: | :---------------------: | :-----:
[faster_rcnn_inception_resnet_v2_atrous_oidv4](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz) | 425        | 54                      | Boxes
[ssd_mobilenetv2_oidv4](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz)                                              | 89         | 36                      | Boxes
[ssd_resnet_101_fpn_oidv4](http://download.tensorflow.org/models/object_detection/ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20.tar.gz)        | 237        | 38                      | Boxes

## iNaturalist Species-trained models

Model name                                                                                                                        | Speed (ms) | Pascal mAP@0.5 | Outputs
--------------------------------------------------------------------------------------------------------------------------------- | :--------: | :------------: | :-----:
[faster_rcnn_resnet101_fgvc](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_fgvc_2018_07_19.tar.gz) | 395        | 58             | Boxes
[faster_rcnn_resnet50_fgvc](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_fgvc_2018_07_19.tar.gz)   | 366        | 55             | Boxes

## AVA v2.1 trained models

Model name                                                                                                                                | Speed (ms) | Pascal mAP@0.5 | Outputs
----------------------------------------------------------------------------------------------------------------------------------------- | :--------: | :------------: | :-----:
[faster_rcnn_resnet101_ava_v2.1](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_ava_v2.1_2018_04_30.tar.gz) | 93         | 11             | Boxes

## Snapshot Serengeti Camera Trap trained models

Model name                                                                                                                                                      | COCO mAP@0.5 | Outputs
--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------: | :-----:
[faster_rcnn_resnet101_snapshot_serengeti](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_snapshot_serengeti_2020_06_10.tar.gz)   | 38           | Boxes
[context_rcnn_resnet101_snapshot_serengeti](http://download.tensorflow.org/models/object_detection/context_rcnn_resnet101_snapshot_serengeti_2020_06_10.tar.gz) | 56           | Boxes

## Pixel 6 Edge TPU models

Model name                                                                                                                    | Pixel 6 Edge TPU Speed (ms) | Pixel 6 Speed with Post-processing on CPU (ms) | COCO 2017 mAP (uint8) | Outputs
----------------------------------------------------------------------------------------------------------------------------- | :-------------------------: | :--------------------------------------------: | :-------------------: | :-----:
[spaghettinet_edgetpu_s](http://download.tensorflow.org/models/object_detection/tf1/spaghettinet_edgetpu_s_2021_10_13.tar.gz) | 1.3                         | 1.8                                            | 26.3                  | Boxes
[spaghettinet_edgetpu_m](http://download.tensorflow.org/models/object_detection/tf1/spaghettinet_edgetpu_m_2021_10_13.tar.gz) | 1.4                         | 1.9                                            | 27.4                  | Boxes
[spaghettinet_edgetpu_l](http://download.tensorflow.org/models/object_detection/tf1/spaghettinet_edgetpu_l_2021_10_13.tar.gz) | 1.7                         | 2.1                                            | 28.0                  | Boxes

[^1]: See [MSCOCO evaluation protocol](http://cocodataset.org/#detections-eval).
    The COCO mAP numbers, with the exception of the Pixel 6 Edge TPU models,
    are evaluated on COCO 14 minival set (note that our split is different
    from COCO 17 Val). A full list of image ids used in our split could be
    found
    [here](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_minival_ids.txt).
[^2]: This is PASCAL mAP with a slightly different way of true positives
    computation: see
    [Open Images evaluation protocols](evaluation_protocols.md),
    oid_V2_detection_metrics.
[^3]: Non-face boxes are dropped during training and non-face groundtruth boxes
    are ignored when evaluating.
[^4]: This is Open Images Challenge metric: see
    [Open Images evaluation protocols](evaluation_protocols.md),
    oid_challenge_detection_metrics.
