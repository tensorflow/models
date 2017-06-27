# Tensorflow detection model zoo

We provide a collection of detection models pre-trained on the
[COCO dataset](http://mscoco.org).
These models can be useful for out-of-the-box inference if you are interested
in categories already in COCO (e.g., humans, cars, etc).
They are also useful for initializing your models when training on novel
datasets.

In the table below, we list each such pre-trained model including:

* a model name that corresponds to a config file that was used to train this
  model in the `samples/configs` directory,
* a download link to a tar.gz file containing the pre-trained model,
* model speed (one of {slow, medium, fast}),
* detector performance on COCO data as measured by the COCO mAP measure.
  Here, higher is better, and we only report bounding box mAP rounded to the
  nearest integer.
* Output types (currently only `Boxes`)

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

| Model name  | Speed | COCO mAP | Outputs |
| ------------ | :--------------: | :--------------: | :-------------: |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz) | fast | 21 | Boxes |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz) | fast | 24 | Boxes |
| [rfcn_resnet101_coco](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz)  | medium | 30 | Boxes |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz) | medium | 32 | Boxes |
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz) | slow | 37 | Boxes |
