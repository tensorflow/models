![No Maintenance Intended](https://img.shields.io/badge/No%20Maintenance%20Intended-%E2%9C%95-red.svg)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# Module networks for question answering on knowledge graph

This code repository contains a TensorFlow model for question answering on
knowledge graph with end-to-end module networks. The original paper describing
end-to-end module networks is as follows.

R. Hu, J. Andreas, M. Rohrbach, T. Darrell, K. Saenko, *Learning to Reason:
End-to-End Module Networks for Visual Question Answering*. in arXiv preprint
arXiv:1704.05526, 2017. ([PDF](https://arxiv.org/pdf/1704.05526.pdf))

```
@article{hu2017learning,
  title={Learning to Reason: End-to-End Module Networks for Visual Question Answering},
  author={Hu, Ronghang and Andreas, Jacob and Rohrbach, Marcus and Darrell, Trevor and Saenko, Kate},
  journal={arXiv preprint arXiv:1704.05526},
  year={2017}
}
```

The code in this repository is based on the original
[implementation](https://github.com/ronghanghu/n2nmn) for this paper.

## Requirements

1.  Install TensorFlow 1.0.0. Follow the [official
    guide](https://www.tensorflow.org/install/). Please note that newer or older
    versions of TensorFlow may fail to work due to incompatibility with
    TensorFlow Fold.
2.  Install TensorFlow Fold. Follow the
    [setup instructions](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/setup.md).
    TensorFlow Fold only supports Linux platform. We have not tested
    the code on other platforms.

## Data

1.  Download the [MetaQA dataset](https://goo.gl/f3AmcY). Click the button
    `MetaQA` and then click `Download` in the drop-down list. Extract the zip
    file after downloading completed. Read the documents there for dataset
    details.
2.  Move the `MetaQA` folder to the root directory of this repository.

## How to use this code

We provide an experiment folder `exp_1_hop`, which applies the implemented model
to the 1-hop vanilla dataset in MetaQA. More experiment folders are coming soon.

Currently, we provide code for training with ground truth layout, and testing
the saved model. Configurations can be modified in `config.py`. They can also be
set via command line parameters.

To train the model:

```
python exp_1_hop/train_gt_layout.py
```

To test the saved model (need to provide the snapshot name):

```
python exp_1_hop/test.py --snapshot_name 00010000
```

## Model introduction

1.  In this model, we store the knowledge graph in a key-value based memory. For
    each knowledge graph edge (subject, relation, object), we use the (subject,
    relation) as the key and the object as the value.
2.  All entities and relations are embedded as fixed-dimension vectors. These
    embeddings are also end-to-end learned.
3.  Neural modules can separately operate on either the key side or the value
    side.
4.  The attention is shared between keys and corresponding values.
5.  The answer output is based on the attention-weighted sum over keys or
    values, depending on the output module.

## Contact
Authors: Yuyu Zhang, Xin Pan

Pull requests and issues: @yuyuz
