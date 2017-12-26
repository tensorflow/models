# Perspective Transformer Nets

## Introduction
This is the TensorFlow implementation for the NIPS 2016 work ["Perspective Transformer Nets: Learning Single-View 3D Object Reconstrution without 3D Supervision"](https://papers.nips.cc/paper/6206-perspective-transformer-nets-learning-single-view-3d-object-reconstruction-without-3d-supervision.pdf)

Re-implemented by Xinchen Yan, Arkanath Pathak, Jasmine Hsu, Honglak Lee

Reference: [Orginal implementation in Torch](https://github.com/xcyan/nips16_PTN)

## How to run this code

This implementation is ready to be run locally or ["distributed across multiple machines/tasks"](https://www.tensorflow.org/deploy/distributed).
You will need to set the task number flag for each task when running in a distributed fashion.
Please refer to the original paper for parameter explanations and training details.

### Installation
*   TensorFlow
    *   This code requires the latest open-source TensorFlow that you will need to build manually.
    The [documentation](https://www.tensorflow.org/install/install_sources) provides the steps required for that.
*   Bazel
    *   Follow the instructions [here](http://bazel.build/docs/install.html).
    *   Alternately, Download bazel from
        [https://github.com/bazelbuild/bazel/releases](https://github.com/bazelbuild/bazel/releases)
        for your system configuration.
    *   Check for the bazel version using this command: bazel version
*   matplotlib
    *   Follow the instructions [here](https://matplotlib.org/users/installing.html).
    *   You can use a package repository like pip.
*   scikit-image
    *   Follow the instructions [here](http://scikit-image.org/docs/dev/install.html).
    *   You can use a package repository like pip.
*   PIL
    *   Install from [here](https://pypi.python.org/pypi/Pillow/2.2.1).

### Dataset

This code requires the dataset to be in *tfrecords* format with the following features:
*   image
    *   Flattened list of image (float representations) for each view point.
*   mask
    *   Flattened list of image masks (float representations) for each view point.
*   vox
    *   Flattened list of voxels (float representations) for the object.
    *   This is needed for using vox loss and for prediction comparison.

You can download the ShapeNet Dataset in tfrecords format from [here](https://drive.google.com/file/d/0B12XukcbU7T7OHQ4MGh6d25qQlk)<sup>*</sup>.

<sup>*</sup> Disclaimer: This data is hosted personally by Arkanath Pathak for non-commercial research purposes. Please cite the [ShapeNet paper](https://arxiv.org/pdf/1512.03012.pdf) in your works when using ShapeNet for non-commercial research purposes.

### Pretraining: pretrain_rotator.py for each RNN step
$ bazel run -c opt :pretrain_rotator -- --step_size={} --init_model={}

Pass the init_model as the checkpoint path for the last step trained model.
You'll also need to set the inp_dir flag to where your data resides.

### Training: train_ptn.py with last pretrained model.
$ bazel run -c opt :train_ptn -- --init_model={}

### Example TensorBoard Visualizations

To compare the visualizations make sure to set the model_name flag different for each parametric setting:

This code adds summaries for each loss. For instance, these are the losses we encountered in the distributed pretraining for ShapeNet Chair Dataset with 10 workers and 16 parameter servers:
![ShapeNet Chair Pretraining](https://drive.google.com/uc?export=view&id=0B12XukcbU7T7bWdlTjhzbGJVaWs "ShapeNet Chair Experiment Pretraining Losses")

You can expect such images after fine tuning the training as "grid_vis" under **Image** summaries in TensorBoard:
![ShapeNet Chair experiments with projection weight of 1](https://drive.google.com/uc?export=view&id=0B12XukcbU7T7ZFV6aEVBSDdCMjQ "ShapeNet Chair Dataset Predictions")
Here the third and fifth columns are the predicted masks and voxels respectively, alongside their ground truth values.

A similar image for when trained on all ShapeNet Categories (Voxel visualizations might be skewed):
![ShapeNet All Categories experiments](https://drive.google.com/uc?export=view&id=0B12XukcbU7T7bDZKNFlkTVAzZmM "ShapeNet All Categories Dataset Predictions")
