![No Maintenance Intended](https://img.shields.io/badge/No%20Maintenance%20Intended-%E2%9C%95-red.svg)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# FEELVOS: Fast End-to-End Embedding Learning for Video Object Segmentation

FEELVOS is a fast model for video object segmentation which does not rely on fine-tuning on the
first frame.

For details, please refer to our paper. If you find the code useful, please
also consider citing it.

* FEELVOS:

```
@inproceedings{feelvos2019,
    title={FEELVOS: Fast End-to-End Embedding Learning for Video Object Segmentation},
    author={Paul Voigtlaender and Yuning Chai and Florian Schroff and Hartwig Adam and Bastian Leibe and Liang-Chieh Chen},
    booktitle={CVPR},
    year={2019}
}
```

## Dependencies

FEELVOS requires a good GPU with around 12 GB of memory and depends on the following libraries

* TensorFlow
* Pillow
* Numpy
* Scipy
* Scikit Learn Image
* tf Slim (which is included in the "tensorflow/models/research/" checkout)
* DeepLab (which is included in the "tensorflow/models/research/" checkout)
* correlation_cost (optional, see below)

For detailed steps to install Tensorflow, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/). A typical user can install
Tensorflow using the following command:

```bash
pip install tensorflow-gpu
```

The remaining libraries can also be installed with pip using:

```bash
pip install pillow scipy scikit-image
```

## Dependency on correlation_cost

For fast cross-correlation, we use correlation cost as an external dependency. By default FEELVOS
will use a slow and memory hungry fallback implementation without correlation_cost. If you care for
performance, you should set up correlation_cost by following the instructions in
correlation_cost/README and afterwards setting ```USE_CORRELATION_COST = True``` in
utils/embedding_utils.py.

## Pre-trained Models

We provide 2 pre-trained FEELVOS models, both are based on Xception-65:

* [Trained on DAVIS 2017](http://download.tensorflow.org/models/feelvos_davis17_trained.tar.gz)
* [Trained on DAVIS 2017 and YouTube-VOS](http://download.tensorflow.org/models/feelvos_davis17_and_youtubevos_trained.tar.gz)

Additionally, we provide a [DeepLab checkpoint for Xception-65 pre-trained on ImageNet and COCO](http://download.tensorflow.org/models/xception_65_coco_pretrained_2018_10_02.tar.gz),
which can be used as an initialization for training FEELVOS.

## Pre-computed Segmentation Masks

We provide [pre-computed segmentation masks](http://download.tensorflow.org/models/feelvos_precomputed_masks.zip)
for FEELVOS both for training with and without YouTube-VOS data for the following datasets:

* DAVIS 2017 validation set
* DAVIS 2017 test-dev set
* YouTube-Objects dataset

## Local Inference
For a demo of local inference on DAVIS 2017 run

```bash
# From tensorflow/models/research/feelvos
sh eval.sh
```

## Local Training
For a demo of local training on DAVIS 2017 run

```bash
# From tensorflow/models/research/feelvos
sh train.sh
```

## Contacts (Maintainers)
*   Paul Voigtlaender, github: [pvoigtlaender](https://github.com/pvoigtlaender)
*   Yuning Chai, github: [yuningchai](https://github.com/yuningchai)
*   Liang-Chieh Chen, github: [aquariusjay](https://github.com/aquariusjay)

## License

All the codes in feelvos folder is covered by the [LICENSE](https://github.com/tensorflow/models/blob/master/LICENSE)
under tensorflow/models. Please refer to the LICENSE for details.
