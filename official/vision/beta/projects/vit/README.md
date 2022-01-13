# Vision Transformer (ViT) and Data-Efficient Image Transformer (DEIT)

**DISCLAIMER**: This implementation is still under development. No support will
be provided during the development phase.

- [![ViT Paper](http://img.shields.io/badge/Paper-arXiv.2010.11929-B3181B?logo=arXiv)](https://arxiv.org/abs/2010.11929)
- [![DEIT Paper](http://img.shields.io/badge/Paper-arXiv.2012.12877-B3181B?logo=arXiv)](https://arxiv.org/abs/2012.12877)

This repository is the implementations of Vision Transformer (ViT) and
Data-Efficient Image Transformer (DEIT) in TensorFlow 2.

**Paper title:**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf).
- [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf).

# Pretrained Model

[Pretrained DEIT (without distillation and repeated augmentation) with 81.76 Top-1 Accuracy (95.65 Top 5 Accuracy).](https://storage.googleapis.com/tf_model_garden/vision/vit/deit_ckpt.zip) For more details see run 15 below.

# Augmentation and Regularization for ViT Training (DEIT-style)

Vision Transformer ([ViT](https://arxiv.org/abs/2010.11929)) is a popular transformer based architecture for image classification. In contrast to Convolutional Neural Networks (CNNs), VIT uses multiple [transformer encoder](https://arxiv.org/pdf/1706.03762.pdf) for the predictions `y=f(x)` , where `y` denotes the predicted confidence scores, `f`  the neural network, and `x` the input image. Training Vision Transformer models on ImageNet is challenging due to the large model scale and insufficient data. ViT model only outperforms state of the art CNNs when trained with a huge amount of training data. For this reason, this project aims to introduce effective data augmentation and regularization methods (introduced in [DEIT](https://arxiv.org/abs/2012.12877)) to TF Vision to better train ViT models on the comparably small ImageNet-1k.

## Methods

In this section we introduce the important regularization and augmentation methods used by DEIT. These methods are essential for DEIT to outperform ViT even if trained on less data. We implemented or adapted all methods (but repeated augmentation) below to match the DEIT implementation (with the exception for RandAugment where we only implemented the random severity). The different augmentation methods are visualized in [this notebook].

### Stochastic Depth

[Stochastic depth](https://arxiv.org/abs/1603.09382) complements the idea of residual networks and is conceptually similar to dropout. In each mini-batch a random set of layers is bypassed/replaced with the identity function (aka residual connection). The probability of bypassing a layer increases linearly with the depth of the network. According to the authors, this procedure reduces the training time and improves generalization. Originally, it was proposed for CNNs but is easily extendable to Transformers. Here we simply randomly bypass the attention blocks.


### RandAugment

[RandAugment](https://arxiv.org/pdf/1909.13719.pdf) combines multiple common image augmentations/transformations. For example, it wraps random augmentations that adjust the contrast, orientation, color, brightness, contrast, and sharpness. It randomly selects the augmentations as well as the augmentation strength. It is designed to work out of the box for a wide range of datasets and comes with a low overhead.


### ColorJitter

[ColorJitter](https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#ColorJitter) is a preprocessing technique implemented in TorchVision. The input images are individually randomly jittered w.r.t. Brightness, contrast and saturation. Note that similar augmentations are also part of RandAugment. We study ColorJitter since DEIT also applies ColorJitter and RandAugment at the same time.


### Random Erasing

In our configuration/implementation of [random erasing](https://arxiv.org/pdf/1708.04896.pdf), we sample random rectangles of different aspect ratios (default is one per image). Then, the pixels in these rectangles are replaced by random Gaussian noise. Note that random erasing is applied after normalizing the input images to zero mean and a variance of one. In our experiments, we follow DEIT and the [timm library](https://rwightman.github.io/pytorch-image-models/) to exclude the [Cutout](https://arxiv.org/pdf/1708.04552.pdf) augmentation in [RandAugment](https://arxiv.org/pdf/1909.13719.pdf). Cutout fills random square regions of the image.


### Label Smoothing

[Label smoothing](https://arxiv.org/pdf/1512.00567.pdf) aims to reduce the issue of overconfidence and consequently overfitting. We generalize the common cross entropy with a smoothing term: 
![(1-\alpha) \log(\hat{y})^\top y + \alpha \cdot C^{-1}](http://mathurl.com/render.cgi?%281-%5Calpha%29%20%5Clog%28%5Chat%7By%7D%29%5E%5Ctop%20y%20+%20%5Calpha%20%5Ccdot%20C%5E%7B-1%7D%5Cnocache) where `α` denotes the strength of the uniform prior ![C^{-1} = \frac{1}{C}](http://mathurl.com/render.cgi?C%5E%7B-1%7D%20%3D%20%5Cfrac%7B1%7D%7BC%7D%5Cnocache) with the number of classes `C`. Moreover, `y` is the ground truth one hot vector. We refer to [this recent paper](https://arxiv.org/pdf/1906.02629.pdf) for a discussion about when and why label smoothing improves over the vanilla cross entropy ![\log(\hat{y})^\top y](http://mathurl.com/render.cgi?%5Clog%28%5Chat%7By%7D%29%5E%5Ctop%20y%5Cnocache).


### Mixup

[Mixup](https://arxiv.org/pdf/1710.09412.pdf) combines two input images through a random convex combination ![x_{\text{aug}} = \lambda \cdot x_1 + (1 - \lambda) \cdot x_2](http://mathurl.com/render.cgi?x_%7B%5Ctext%7Baug%7D%7D%20%3D%20%5Clambda%20%5Ccdot%20x_1%20+%20%281%20-%20%5Clambda%29%20%5Ccdot%20x_2%5Cnocache) (`λ` is drawn from a beta distribution). Analogously, the labels are calculated with ![y_{\text{aug}} = \lambda \cdot y_1 + (1 - \lambda) \cdot y_2](http://mathurl.com/render.cgi?y_%7B%5Ctext%7Baug%7D%7D%20%3D%20%5Clambda%20%5Ccdot%20y_1%20+%20%281%20-%20%5Clambda%29%20%5Ccdot%20y_2%5Cnocache). In our case the `y`s are smoothed labels (see above).

### CutMix

[Cutmix](https://arxiv.org/pdf/1905.04899.pdf) also combines two training instances. Instead of linearly interpolating, here we paste a random rectangular area of one image into the other. The labels are derived similarly to mixup. Here `λ` compensates the area of the inserted rectangle.


### Repeated Augmentation

[Repeated Augmentation](https://arxiv.org/pdf/1902.05509.pdf) duplicates (here threefold) the training images within the same batch and consequently performs different random augmentations on the same images. This increases the number of batches per epoch, but does not increase the I/O overhead when training on a single node (i.e. each image is loaded once per epoch). However, in a distributed setup this is not simple to achieve because e.g. the parameters of [BatchNorm](https://arxiv.org/pdf/1502.03167.pdf) and [LayerNorm](https://arxiv.org/pdf/1607.06450.pdf) are typically not synchronized between the workers. For unbiased estimates, DEIT distributes the images to different workers and hence Repeated Augmentation effectively triples the compute cost.


## Experiments

We successively enable one augmentation/regularization at a time and report the differences (similarly to Table 8 in the DEIT paper). In each experiment, we pretrain the ViT/DEIT architecture with resolution 224. For simplicity, we add more techniques without tuning the other parameters. Note that constant hyperparameters will likely not yield the optimal accuracy since e.g. with less augmentation the accuracy is expected to increase with the regularization strength. In the following table, we use all train images (≅1.3 million images) and report the top-1 accuracy on the validation set (50,000 images).


<table>
  <tr>
   <td><strong>#</strong>
   </td>
   <td><strong>DEIT para.</strong>
   </td>
   <td><strong>Stoch-</strong>
<p>
<strong>astic Depth</strong>
   </td>
   <td><strong>Color Jitter</strong>
   </td>
   <td><strong>Rand Erase</strong>
   </td>
   <td><strong>Rand Aug. (w/o Cutout)</strong>
   </td>
   <td><strong>Rand Aug.</strong>
   </td>
   <td><strong>Mixup</strong>
   </td>
   <td><strong>Cutmix</strong>
   </td>
   <td><strong>Rep. Aug.</strong>
   </td>
   <td><strong>Top 1 Acc.</strong>
   </td>
  </tr>
  <tr>
   <td><strong>1</strong>
   </td>
   <td>𐄂 \
(tf-like weight init)
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>70.87
   </td>
  </tr>
  <tr>
   <td><strong>2</strong>
   </td>
   <td>𐄂  \
 (jax-like weight init)
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>72.26
   </td>
  </tr>
  <tr>
   <td><strong>3</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>67.10
   </td>
  </tr>
  <tr>
   <td><strong>4</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>70.06
   </td>
  </tr>
  <tr>
   <td><strong>5</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>72.43
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>6</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>79.84
   </td>
  </tr>
  <tr>
   <td><strong>7</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>75.71
   </td>
  </tr>
  <tr>
   <td><strong>8</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>75.79
   </td>
  </tr>
  <tr>
   <td><strong>9</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
<p>
(random severity)
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>75.86
   </td>
  </tr>
  <tr>
   <td><strong>10</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>75.89
   </td>
  </tr>
  <tr>
   <td><strong>11</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>76.41
   </td>
  </tr>
  <tr>
   <td><strong>12</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td>75.48
   </td>
  </tr>
  <tr>
   <td><strong>13</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td>80.05
   </td>
  </tr>
  <tr>
   <td><strong>14</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>81.69
   </td>
  </tr>
  <tr>
   <td><strong>15</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td><strong>81.77</strong>
<p>
(aver. of 5)
   </td>
  </tr>
  <tr>
   <td><strong>16</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>81.53
   </td>
  </tr>
  <tr>
   <td><strong>17</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>81.48
   </td>
  </tr>
  <tr>
   <td><strong>18</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>81.47
   </td>
  </tr>
  <tr>
   <td><strong>19</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td>𐄂
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td><strong>✓</strong>
   </td>
   <td>81.05
<p>
(aver. of 2)
   </td>
  </tr>
</table>


_15 - the average of five reruns, abs. 5.17% higher than reported in DEIT paper for the same set of augmentations. This configuration matches performance with best DEIT model (without distillation)_

_19 - the average of two reruns, abs. 0.75% lower than reported in DEIT paper_

We provide [training details for all runs through tensorboard.dev](https://tensorboard.dev/experiment/s6Yb8tYJSKmYlj1IE9gPtA/#scalars&regexInput=validation&runSelectionState=eyIwMS12aXQtdGZpbml0LTAxL3RyYWluIjp0cnVlLCIwMS12aXQtdGZpbml0LTAxL3ZhbGlkYXRpb24iOnRydWUsIjAyLXZpdC1qYXhpbml0LTAxL3ZpdC0wMy90cmFpbiI6dHJ1ZSwiMDItdml0LWpheGluaXQtMDEvdml0LTAzL3ZhbGlkYXRpb24iOnRydWUsIjAzLWRlaXQtbm9hdWctMDEvdHJhaW4iOnRydWUsIjAzLWRlaXQtbm9hdWctMDEvdmFsaWRhdGlvbiI6dHJ1ZSwiMDQtZGVpdC1zZC0wMS90cmFpbiI6dHJ1ZSwiMDQtZGVpdC1zZC0wMS92YWxpZGF0aW9uIjp0cnVlLCIwNC1kZWl0LXNkLTAyL3RyYWluIjp0cnVlLCIwNC1kZWl0LXNkLTAyL3ZhbGlkYXRpb24iOnRydWUsIjA1LWRlaXQtc2QtY29sb3JqLTAxL3RyYWluIjp0cnVlLCIwNS1kZWl0LXNkLWNvbG9yai0wMS92YWxpZGF0aW9uIjp0cnVlLCIwNi1kZWl0LXNkLW1peHVwLWN1dG1peC0wMS90cmFpbiI6dHJ1ZSwiMDYtZGVpdC1zZC1taXh1cC1jdXRtaXgtMDEvdmFsaWRhdGlvbiI6dHJ1ZSwiMDctZGVpdC1zZC1yYW5kYWNvbXBsZXRlLTAxL3RyYWluIjp0cnVlLCIwNy1kZWl0LXNkLXJhbmRhY29tcGxldGUtMDEvdmFsaWRhdGlvbiI6dHJ1ZSwiMDgtZGVpdC1zZC1yYW5kYS0wMS90cmFpbiI6dHJ1ZSwiMDgtZGVpdC1zZC1yYW5kYS0wMS92YWxpZGF0aW9uIjp0cnVlLCIwOS1kZWl0LXNkLXJhbmRhcmFuZG9tc2V2ZXJpdHktMDEvdHJhaW4iOnRydWUsIjA5LWRlaXQtc2QtcmFuZGFyYW5kb21zZXZlcml0eS0wMS92YWxpZGF0aW9uIjp0cnVlLCIxMC1kZWl0LXNkLWVyYXNlLXJhbmRhLTAxL3RyYWluIjp0cnVlLCIxMC1kZWl0LXNkLWVyYXNlLXJhbmRhLTAxL3ZhbGlkYXRpb24iOnRydWUsIjExLWRlaXQtc2QtY29sb3JqLWVyYXNlLXJhbmRhLTAxL3RyYWluIjp0cnVlLCIxMS1kZWl0LXNkLWNvbG9yai1lcmFzZS1yYW5kYS0wMS92YWxpZGF0aW9uIjp0cnVlLCIxMi1kZWl0LXNkLXJhbmRhLXJlcGEtMDEvdHJhaW4iOnRydWUsIjEyLWRlaXQtc2QtcmFuZGEtcmVwYS0wMS92YWxpZGF0aW9uIjp0cnVlLCIxMy1kZWl0LXNkLXJhbmRhLW1peHVwLTAxL3RyYWluIjp0cnVlLCIxMy1kZWl0LXNkLXJhbmRhLW1peHVwLTAxL3ZhbGlkYXRpb24iOnRydWUsIjE0LWRlaXQtc2QtcmFuZGEtY3V0bWl4LTAxL3RyYWluIjp0cnVlLCIxNC1kZWl0LXNkLXJhbmRhLWN1dG1peC0wMS92YWxpZGF0aW9uIjp0cnVlLCIxNS1kZWl0LXNkLXJhbmRhLW1peHVwLWN1dG1peC0wMS90cmFpbiI6dHJ1ZSwiMTUtZGVpdC1zZC1yYW5kYS1taXh1cC1jdXRtaXgtMDEvdmFsaWRhdGlvbiI6dHJ1ZSwiMTUtZGVpdC1zZC1yYW5kYS1taXh1cC1jdXRtaXgtMDIvdHJhaW4iOnRydWUsIjE1LWRlaXQtc2QtcmFuZGEtbWl4dXAtY3V0bWl4LTAyL3ZhbGlkYXRpb24iOnRydWUsIjE1LWRlaXQtc2QtcmFuZGEtbWl4dXAtY3V0bWl4LTAzL3RyYWluIjp0cnVlLCIxNS1kZWl0LXNkLXJhbmRhLW1peHVwLWN1dG1peC0wMy92YWxpZGF0aW9uIjp0cnVlLCIxNS1kZWl0LXNkLXJhbmRhLW1peHVwLWN1dG1peC0wNC90cmFpbiI6dHJ1ZSwiMTUtZGVpdC1zZC1yYW5kYS1taXh1cC1jdXRtaXgtMDQvdmFsaWRhdGlvbiI6dHJ1ZSwiMTUtZGVpdC1zZC1yYW5kYS1taXh1cC1jdXRtaXgtMDUvdHJhaW4iOnRydWUsIjE1LWRlaXQtc2QtcmFuZGEtbWl4dXAtY3V0bWl4LTA1L3ZhbGlkYXRpb24iOnRydWUsIjE2LWRlaXQtc2QtY29sb3JqLXJhbmRhLW1peHVwLWN1dG1peC0wMS90cmFpbiI6dHJ1ZSwiMTYtZGVpdC1zZC1jb2xvcmotcmFuZGEtbWl4dXAtY3V0bWl4LTAxL3ZhbGlkYXRpb24iOnRydWUsIjE3LWRlaXQtc2QtZXJhc2UtcmFuZGEtbWl4dXAtY3V0bWl4LTAxL3RyYWluIjp0cnVlLCIxNy1kZWl0LXNkLWVyYXNlLXJhbmRhLW1peHVwLWN1dG1peC0wMS92YWxpZGF0aW9uIjp0cnVlLCIxOC1kZWl0LXNkLWNvbG9yai1lcmFzZS1yYW5kYS1taXh1cC1jdXRtaXgtMDEvdHJhaW4iOnRydWUsIjE4LWRlaXQtc2QtY29sb3JqLWVyYXNlLXJhbmRhLW1peHVwLWN1dG1peC0wMS92YWxpZGF0aW9uIjp0cnVlLCIxOS1kZWl0LXNkLXJhbmRhLW1peHVwLWN1dG1peC1yZXBhLTAxL3RyYWluIjp0cnVlLCIxOS1kZWl0LXNkLXJhbmRhLW1peHVwLWN1dG1peC1yZXBhLTAxL3ZhbGlkYXRpb24iOnRydWV9&_smoothingWeight=0.797) _and also the pre-trained models will be made available_.
