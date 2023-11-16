# Pix2Seq: A Language Modeling Framework for Object Detection

[![Pix2Seq](https://img.shields.io/badge/Pix2Seq-arXiv.2109.10852-B3181B?)](https://arxiv.org/abs/2109.10852).

TensorFlow 2 implementation of A Language Modeling Framework for Object
Detection.

The official implementation of Pix2Seq in Tensorflow 2 is [Here]
(https://github.com/google-research/pix2seq).

⚠️ Disclaimer: All datasets hyperlinked from this page are not owned or
distributed by Google. The dataset is made available by third parties. Please
review the terms and conditions made available by the third parties before using
the data.

## Training
To train the model on MS-COCO, try the following command:

```
python3 train.py \
  --mode=train \
  --experiment=pix2seq_r50_coco  \
  --model_dir=$MODEL_DIR \
  --config_file=./configs/experiments/coco_pix2seq_r50_gpu.yaml
```

## Evaluation
To evaluate the model on MS-COCO, try the following command:

```
python3 train.py \
  --mode=eval \
  --experiment=pix2seq_r50_coco  \
  --model_dir=$MODEL_DIR \
  --config_file=./configs/experiments/coco_pix2seq_r50_gpu.yaml
```

## Cite

[Pix2seq paper](https://arxiv.org/abs/2109.10852):


```
@article{chen2021pix2seq,
  title={Pix2seq: A language modeling framework for object detection},
  author={Chen, Ting and Saxena, Saurabh and Li, Lala and Fleet, David J and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2109.10852},
  year={2021}
}
```

## Contributors

<!--- go/keep-sorted start -->
* Gunho Park ([Github @gunho1123](https://github.com/gunho1123))
* Jiageng Zhang ([Github @Zarjagen](https://github.com/Zarjagen))
* Shicheng Xu ([Github @lightxu](https://github.com/lightxu))
* Tyler Scott ([Github @tylersco](https://github.com/tylersco))
* Yu Lou ([Github @LouYu2015](https://github.com/LouYu2015))
<!--- go/keep-sorted end -->
