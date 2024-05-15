# TF-Vision Data Augmentation and Model Regularization

## Data augmentation methods

TF-Vision provides a rich collection of advanced SoTA data augmentation methods
for various vision tasks. Default augmentation methods such as random flipping
or cropping will not be discussed here.

### RandAugment and AutoAugment

[RandAugment](https://arxiv.org/abs/1909.13719) or
[AutoAugment](https://arxiv.org/abs/1805.09501) combines multiple common image
augmentations/transformations that adjust the contrast, orientation, color,
brightness and sharpness. It randomly selects the augmentations as well as the
augmentation strength. The TF-Vision implementations are designed to work out of
the box for a wide range of datasets and come with low overhead. RandAugment
and AutoAugment nowadays are commonly adopted for SoTA image classification
model training.

Supported vision tasks:

*   Image classification
*   Video action classification
*   Object detection

[Definitions](https://github.com/tensorflow/models/blob/master/official/vision/configs/common.py)
in TF-Vision:

```python
@dataclasses.dataclass
class RandAugment(hyperparams.Config):
  """Configuration for RandAugment."""
  num_layers: int = 2
  magnitude: float = 10
  cutout_const: float = 40
  translate_const: float = 10
  magnitude_std: float = 0.0
  prob_to_apply: Optional[float] = None
  exclude_ops: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class AutoAugment(hyperparams.Config):
  """Configuration for AutoAugment."""
  augmentation_name: str = 'v0'
  cutout_const: float = 100
  translate_const: float = 250
```

Yaml template of using the techniques in TF-Vision:

```python
task:
  train_data:
    aug_type:
      type: 'randaug'
      randaug:
        magnitude: 10
        magnitude_std: 0.0
        num_layers: 2
```

### Image scale jittering

Image scale jittering randomly selects a scale from a user-specified range. It
either upsamples the image if the scale is greater than 1.0 or downsamples the image if
the scale is smaller than 1.0. Randomly cropping or zero-padding is further
performed to tailor the scaled image to a desired size. Scale jittering is one
of the most effective augmentation methods for training SoTA object detection
model. It has been widely used in recent publications on object detection such
as [SpineNet](https://arxiv.org/abs/1912.05027),
[EfficientDet](https://arxiv.org/abs/1911.09070),
[Detection-RS](https://arxiv.org/pdf/2107.00057.pdf) and
[Copy-Paste](https://arxiv.org/abs/2012.07177).

Supported vision tasks:

*   Object detection
*   Semantic segmentation

[Definition](https://github.com/tensorflow/models/blob/ae79c473faf78bfa068ecbb57504483eb81eef21/official/vision/configs/retinanet.py#L51C1-L67C35)
in TF-Vision:

```python
@dataclasses.dataclass
class Parser(hyperparams.Config):
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
```

Yaml template of using the techniques in TF-Vision:

```python
task:
  train_data:
    parser:
      aug_scale_max: 2.0
      aug_scale_min: 0.5
```

### Mixup and Cutmix

[Mixup](https://arxiv.org/pdf/1710.09412.pdf) combines two input images through
a random convex combination:

```python
img = img_1 * a + img_2 * (1.0 - a),
```

where a is drawn from a beta distribution). Analogously, the labels are
calculated with:

```python
label = label_1 * a + label_2 * (1.0 - a).
```

[Cutmix](https://arxiv.org/pdf/1905.04899.pdf) also combines two training
instances. Instead of linearly interpolating, here we paste a random rectangular
area of img_2 and into img_1. The labels are derived similarly to mixup:

```python
label = label_1 * a + label_2 * (1.0 - a),
```

where a compensates the area of the inserted rectangle. The two methods are key
factors in training transformer based image models
([ViT](https://arxiv.org/abs/2010.11929),
[DEIT](https://arxiv.org/abs/2012.12877)) with limited labeled data.

Supported vision tasks:

*   Image classification

[Definitions](https://github.com/tensorflow/models/blob/ae79c473faf78bfa068ecbb57504483eb81eef21/official/vision/configs/common.py#L93C1-L100C40)
in TF-Vision:

```python
@dataclasses.dataclass
class MixupAndCutmix(hyperparams.Config):
  """Configuration for MixupAndCutmix."""
  mixup_alpha: float = .8
  cutmix_alpha: float = 1.
  prob: float = 1.0
  switch_prob: float = 0.5
  label_smoothing: float = 0.1
```

Yaml template of using the methods in TF-Vision:

```python
task:
  train_data:
    mixup_and_cutmix:
      cutmix_alpha: 1.0
      label_smoothing: 0.1
      mixup_alpha: 0.8
      prob: 1.0
      switch_prob: 0.5
```

### Random erasing

[Random erasing](https://arxiv.org/pdf/1708.04896.pdf) samples random rectangles
of different aspect ratios (default is one per image). Then, the pixels in these
rectangles are replaced by random Gaussian noise. Note that random erasing is
applied after normalizing the input images to zero mean and a variance of one.
To apply random erasing , we exclude the Cutout augmentation in RandAugment.
Cutout fills random square regions of the image.

Supported vision tasks:

*   Image classification

[Definition](https://github.com/tensorflow/models/blob/ae79c473faf78bfa068ecbb57504483eb81eef21/official/vision/configs/common.py#L80C1-L90C14)
in TF-Vision:

```python
@dataclasses.dataclass
class RandomErasing(hyperparams.Config):
  """Configuration for RandomErasing."""
  probability: float = 0.25
  min_area: float = 0.02
  max_area: float = 1 / 3
  min_aspect: float = 0.3
  max_aspect = None
  min_count = 1
  max_count = 1
  trials = 10
```

Yaml template of using the methods in TF-Vision:

```python
task:
  train_data:
    random_erasing:
      min_area: float: 0.02
      max_area: float: 0.33
```

## Model Regularization methods

TF-Vision provides a rich collection of advanced SoTA model regularization
methods for different models and tasks. Default regularization methods such as
weight decay and dropout will not be described here.

### Stochastic depth

[Stochastic depth](https://arxiv.org/abs/1603.09382) complements the idea of
residual networks and is conceptually similar to dropout. In each mini-batch a
random set of layers is bypassed/replaced with the identity function (aka
residual connection). The probability of bypassing a layer increases linearly
with the depth of the network. This procedure reduces the training time and
improves generalization.

Supported models:

*   ResNet, ResNet-RS
*   SpineNet, SpineNet-mobile, SpineNet-seg
*   Vision Transformer (ViT)
*   ResNet-RS-3D

Yaml template of using the methods in TF-Vision:

```python
task:
  model:
    backbone:
      resnet:
        init_stochastic_depth_rate: 0.2
```

### Label smoothing

[Label smoothing](https://arxiv.org/pdf/1512.00567.pdf) aims to reduce the issue
of overconfidence and consequently overfitting. Instead of one-hot labeling, we
assign a high value x (close to 1) to the correct class and evenly distribute
(1 - x) to the remaining classes.

Supported models:

*   Image classification
*   Video action classification

Yaml template of using the methods in TF-Vision:

```python
task
  losses:
    label_smoothing: 0.1
```
