# End-to-End Object Detection with Transformers (DETR)

[![DETR](https://img.shields.io/badge/DETR-arXiv.2005.12872-B3181B?)](https://arxiv.org/abs/2005.12872).

TensorFlow 2 implementation of End-to-End Object Detection with Transformers

⚠️ Disclaimer: All datasets hyperlinked from this page are not owned or
distributed by Google. The dataset is made available by third parties.
Please review the terms and conditions made available by the third parties
before using the data.

## Scripts:

You can find the scripts to reproduce the following experiments in
detr/experiments.


## DETR [COCO](https://cocodataset.org) ([ImageNet](https://www.image-net.org) pretrained)

| Model     | Resolution | Batch size | Epochs | Decay@ | Params (M) | Box AP | Dashboard | Checkpoint | Experiment |
| --------- | :--------: | ----------:| ------:| -----: | ---------: | -----: | --------: | ---------: | ---------: |
| DETR-ResNet-50 | 1333x1333 |64|300| 200 |41 | 40.6 | [tensorboard](https://tensorboard.dev/experiment/o2IEZnniRYu6pqViBeopIg/#scalars) | [ckpt](https://storage.googleapis.com/tf_model_garden/vision/detr/detr_resnet_50_300.tar.gz) | [detr_r50_300epochs.sh](https://github.com/tensorflow/models/blob/master/official/projects/detr/experiments/detr_r50_300epochs.sh) |
| DETR-ResNet-50 | 1333x1333 |64|500| 400 |41 | 42.0| [tensorboard](https://tensorboard.dev/experiment/YFMDKpESR4yjocPh5HgfRw/) | [ckpt](https://storage.googleapis.com/tf_model_garden/vision/detr/detr_resnet_50_500.tar.gz) | [detr_r50_500epochs.sh](https://github.com/tensorflow/models/blob/master/official/projects/detr/experiments/detr_r50_500epochs.sh) |
| DETR-ResNet-50 | 1333x1333 |64|300| 200 |41 | 40.6 | paper | NA | NA |
| DETR-ResNet-50 | 1333x1333 |64|500| 400 |41 | 42.0 | paper | NA | NA |
| DETR-DC5-ResNet-50 | 1333x1333 |64|500| 400 |41 | 43.3 | paper | NA | NA |

## Need contribution:

*   Add DC5 support and update experiment table.


## Citing TensorFlow Model Garden

If you find this codebase helpful in your research, please cite this repository.

```
@misc{tensorflowmodelgarden2020,
  author = {Hongkun Yu and Chen Chen and Xianzhi Du and Yeqing Li and
            Abdullah Rashwan and Le Hou and Pengchong Jin and Fan Yang and
            Frederick Liu and Jaeyoun Kim and Jing Li},
  title = {{TensorFlow Model Garden}},
  howpublished = {\url{https://github.com/tensorflow/models}},
  year = {2020}
}
```
