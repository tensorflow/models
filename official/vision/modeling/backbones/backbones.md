# Vision Backbone Models

> This user guide presents the basic description of backbone models for computer
> vision tasks. The backbone models refer to the core component of a deep
> learning architecture responsible for extracting features from raw image data.
> These models act like a feature extractor, taking an image as input and
> generating a high-dimensional representation that captures prominent visual
> features. These are the foundational layer for tasks like image
> classification, object detection, and segmentation.


## Table of Contents

- [ResNet](#ResNet)
- [SpineNet](#SpineNet)
- [EfficientNet](#EfficientNet)
- [RevNet](#RevNet)
- [MobileNet](#MobileNet)
- [MobileDet](#MobileDet)

## ResNet

ResNet is a Residual Neural Network, a powerful convolutional neural network
(CNN) architecture widely used in computer vision tasks, particularly image
recognition, classification and segmentation.

**Core Idea:** ResNet introduced "skip connections" that allow gradients to flow
directly through the network, using identity mapping. The skip connection allows
the network to extract features from original data rather than transformed data
from the previous convolution layer, making training more efficient. Addressed
the vanishing gradient descent problem, a limitation in traditional deep neural
networks where gradients become very small during back propagation, that stops
the training progress.

**Building Blocks:** Utilizes residual blocks as the fundamental building block
in the architecture. Each residual block consists of: Few convolutional layers
for feature extraction. Activation functions for introducing non-linearity.
Batch normalization for improved stability and faster training. A "skip
connection" that adds the input of the block directly to its output.

**Variants(ResNet-RS):** ResNet-RS is a variant of ResNet, (builds upon ResNet)
which emphasizes training and scaling strategies. The primary focus is on
improving the efficiency of ResNet models while maintaining the similar
accuracy. The model introduced two novel scaling strategies. Scaling the model
depth in regimes where overfitting can occur for longer epochs and scaling the
image resolution more slowly. This results in more computationally efficient
models that can be trained without the need for expensive hardware accelerators.

Some of the improvements in Resnet-RS are: Cosine learning rate schedule, Label
smoothing, Stochastic depth, Randaugment, Decreased weight decay, Squeeze and
excitation. Model Garden supports building various ResNet and ResNetRS model
variants with different layers 10, 18, 26, 34, 50, 101, 152, 200, 270, 350, and
420 to make them suitable for real time applications and resource-constrained
environments as well. Deeper variants offer potentially higher accuracy but
require more training data and computational resources.

**Advantages of ResNet:**

*   **Reduced Vanishing Gradients:** Training deep neural networks can be
    challenging due to the vanishing gradient problem. In deeper networks, the
    gradients become zero as they are back propagating through layers and makes
    it difficult to train. ResNet's skip connections addresses this issue by
    providing a direct path for gradients to flow, allowing even deeper layers
    to learn and contribute to the overall model.

*   **Deeper Networks, Better Performance:** ResNet's core innovation is the use
    of residual blocks with skip connections. These connections allow the
    network to learn from much deeper layers compared to traditional models.
    This increased depth enables ResNet to capture more complex and minute
    features in data, leading to better performance in tasks like image
    classification and object detection.

*   **Improved Generalization:** By incorporating information from earlier
    layers through skip connections, ResNets tend to learn more generalizable
    features from the data. This means they perform better on unseen data
    compared to models that struggle to capture underlying patterns due to
    vanishing gradients.

*   **Faster Convergence:** The skip connections in ResNets also contribute to
    faster training convergence. By allowing gradients to flow more directly,
    the network can adjust its weights and learn more efficiently. This
    translates to faster training times and lower computational costs.

**Source Code :**
[ResNet](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/resnet.py)

**References:**

*   Irwan Bello, William Fedus, Xianzhi Du, Ekin D. Cubuk, Aravind Srinivas ,
    Tsung-Yi Lin , Jonathon Shlens, Barret Zoph-
    [Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579v1)

*   Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun -
    [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385)

*   [Francois Chollet's GitHub repository:](https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py)

[TOC]

## SpineNet

SpineNet is a backbone architecture with scale-permuted intermediate features
and cross-scale connections learned through Neural Architecture Search (NAS). By
using blocks with varying feature resolutions, this backbone improves
information flow across a range of scales. It uses cross-scale connections to
further improve feature richness by allowing information to be exchanged between
blocks at different resolutions. Unlike the convolutional neural networks
(CNNs), the resolution of feature maps is gradually reduced as going deeper by
conventional backbones. Since it ignores fine-grained information that is
essential for precise localization.

**Neural Architecture Search (NAS):** NAS automates the process of finding
optimal network architectures by searching a vast space of possibilities. In
NAS, the scale permutations for the intermediate and output blocks are first
searched, then cross-scale connections between blocks are determined. The model
is further improved by adding block adjustments in the search space

*   **Scale Permuted Intermediate features:** Spinenet generates the features at
    various scales and permutes them throughout the network. This allows the
    network to access and utilize features from different scales at any point,
    improving its ability to capture multi-scale information crucial for object
    recognition and localization.

*   **Cross Scale Connections:** SpineNet incorporates cross-scale connections
    that directly link these permuted features across different scales. This
    enables information exchange between feature maps of different resolutions,
    allowing the network to learn a more comprehensive understanding of the
    object across various scales.

*   **Scale Permutations:** The scale permutations need to be determined first
    before searching for the remaining architecture. The search space of scale
    permutations are defined by permuting intermediate and output blocks
    respectively and the search space size is (N-5)!5!.

*   **Cross Scale Connections:** Two input connections are used for each
    block.The parent blocks can be any block with a lower ordering or block from
    the stem network. When connecting blocks at different feature levels,
    spatial and feature dimensions need to be resampled.

Model garden supports various configurations for different sizes: 49S, 49, 96,
143, 143L and 190 to improve performance. The filter size and number of filters
can be scaled by a factor to reduce the number of parameters in the model.

**SpineNet_Mobile- A variant of SpineNet:**

SpineNet-Mobile is a mobile compatible version of SpineNet , which is optimized
for efficiency on mobile and edge devices. This version has fewer parameters and
computations as compared to original SpineNet, making it more suitable for
resource-constrained environments. Model garden has allowed the SpineNet to
configure for different scaling maps of size: 49, 49S, 49XS for mobile
environments. Supported filter_map_size are { 8, 16, 24, 40, 80, 112}.

**Advantages of SpineNet:**

*   **Improved Localization Accuracy:** By preserving spatial information and
    enabling cross-scale communication, SpineNet can better localize objects
    within the image compared to traditional encoder-decoder architectures.
*   **Efficiency:** SpineNet can achieve state-of-the-art accuracy on object
    detection tasks while using fewer computations compared to some ResNet-based
    models with Feature Pyramid Networks (FPNs).

**Source Code:**
[SpineNet](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/spinenet.py)
,
[SpineNet_Mobile](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/spinenet_mobile.py)

**References:**

*   Xianzhi Du, Tsung-Yi Lin, Pengchong Jin, Golnaz Ghiasi, Mingxing Tan, Yin
    Cui, Quoc V. Le, Xiaodan Song-
    [SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization](https://arxiv.org/abs/1912.05027)

*   Xianzhi Du, Tsung-Yi Lin, Pengchong Jin, Yin Cui, Mingxing Tan, Quoc Le,
    Xiaodan Song -
    [Efficient Scale-Permuted Backbone with Learned Resource Distribution.](https://arxiv.org/abs/2010.11426)

[TOC]

## EfficientNet

EfficientNets uniformly scales all dimensions of depth/width/resolution using a
simple and effective compound coefficient and which achieve much better accuracy
and efficiency than previous ConvNets. EfficientNet introduces a novel scaling
method and its effectively designed architecture make it a valuable model for
various computer vision applications. Compound scaling method: Compound scaling
achieves balanced increase across a network's width, depth, and resolution by
applying a consistent scaling factor to each dimension.

The following equations demonstrate mathematical intuition.
<p style="text-align: center;"> Depth $$ d = α^Ø $$, Width $$w = β^Ø$$,
Resolution $$r = γ^Ø $$ </p> <p style="text-align: center;">such that $$α
.β^2.γ^2 \approx 2$$ where $$ α ≥ 1, β ≥ 1, γ ≥ 1$$ </p> **Width Scaling( α):**
Scales the network width proportionally to φ raised to this power.

**Depth Scaling( β):** Scales the network depth proportionally to φ raised to
this power.(commonly denoted as β)

**Resolution Scaling (γ):** Scales the input image resolution by multiplying by
φ raised to this power.

**Choosing the Right EfficientNet:** Various versions of EfficietNet (Bo-B7)
represent different values for the compound coefficient (φ), which controls the
overall increase in resources for scaling.

*   **Smaller φ values (B0, B1):** These models are lightweight and
    resource-efficient, making them suitable for deployment on devices with
    limited powe capability like mobile phones.
*   **Larger φ values (B6, B7):** These models are more powerful and capable of
    achieving higher accuracy on complex tasks, but they require more
    computational resources and are better suited for powerful hardware like
    GPUs.

Model garden provides three MobileNet variants: MobileNetV1, MobileNetV2, and
MobileNetV3 (including Large, Small, and EdgeTPU versions). Each variant has a
different set of block specifications defining the architecture.

**Advantages of EfficientNet:**

*   **Faster Training:** Models with fewer parameters train quicker, saving time
    and computational resources.
*   **Small Memory Footprint:** Lower parameter count means the model takes up
    less space in memory, making it suitable for deployment on devices with
    limited memory, such as mobile phones or embedded systems.
*   **Deployment on Edge Devices:** The lower computational demands allow
    EfficientNet models to run on devices with limited processing power,
    enabling image recognition capabilities at the network edge.

**Source Code:**
[EfficientNet](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/efficientnet.py)

**References:**

*   Mingxing Tan, Quoc V. Le -
    [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946v5)

[TOC]

## RevNet

RevNet is a Reversible Residual Networks, a memory-efficient variant of ResNets
that utilizes reversible residual blocks to reduce memory consumption during
backpropagation. The activations can be reconstructed from subsequent layers,
eliminating the need to store more activations during backpropagation. This
makes RevNets suitable for training deeper networks on resource-constrained
devices. In case of traditional Deep Residual Networks (ResNets), achieve high
accuracy but require significant memory to store activations during training,
limiting network size and efficiency.

**Core Components:**

*   **Reversible Residual Blocks:** These blocks are the building blocks of
    RevNet. They process the input features ($$x$$) by splitting them into two
    channels ($$x1$$ and $$x2$$) and passing them through separate functions
    ($$F$$ and $$G$$). The outputs ($$y1$$ and $$y2$$) are then combined with
    the original input ($$x$$) using element-wise addition.

*   **Reversibility:** The key innovation lies in the design of functions $$F$$
    and $$G$$. They are designed to be invertible, meaning the original input
    (x) can be recovered from the outputs ($$y1$$ and $$y2$$) using inverse
    functions $$F ^(-1)$$ and G ^1 .

It's important to note that reversible blocks are restricted to having a stride
of 1. In convolutional networks, stride refers to the step size of the filter as
it moves across the input. A stride greater than 1 would result in some
information being discarded, making it impossible to perfectly reconstruct the
input. This is a constraint compared to standard ResNets, which sometimes use
layers with larger strides.

The stride limitation means that in a RevNet architecture, any layers with
strides greater than 1 would need to have their activations stored explicitly
during training. However, in practice, these non-reversible layers are typically
few in number compared to the many reversible blocks, so the overall memory
savings are still significant.

Model garden provides implementation of different RevNet variants : 38, 56 and
104 layers.

**Advantages of RevNet:**

*   **Reduced Memory Consumption:** By eliminating the need to store most
    activations, RevNets can achieve significant memory savings compared to
    standard ResNets, especially for deeper networks.

*   **Maintains Accuracy:** RevNets have been shown to achieve comparable
    accuracy to ResNets on image classification tasks like CIFAR-10, CIFAR-100,
    and ImageNet.

**Source Code:**
[RevNet](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/revnet.py)

**References:**

Aidan N. Gomez, Mengye Ren, Raquel Urtasun, Roger B. Grosse-[The Reversible
Residual Network: Backpropagation Without Storing
Activations](https://arxiv.org/abs/1707.04585)

[TOC]

## MobileNet

MobileNet is a convolutional neural network for mobile vision applications that
is simple, efficient, and requires little computational power. MobileNet is
widely employed in a variety of real-world applications, including object
identification, fine-grained categorization, facial characteristics, and
localization. Mobilenet is specifically tailored for mobile and resource
constrained environments. Mobilenet introduced the inverted residual block with
linear bottleneck. This module takes a low-dimensional compressed representation
of input which is first expanded to high dimension and filtered with a
lightweight depthwise convolution. Features are subsequently projected back to a
low-dimensional representation with a linear convolution.

Depthwise separable convolutions to build more efficient neural networks.
Instead of using a single, computationally expensive convolution operation, they
factorize it into two separate steps, resulting in significant savings in terms
of computations and parameters.

**Key concepts of Mobilenet:**

*   **Depthwise Convolution:** A depthwise convolution applies a separate filter
    to each channel individually. This means each filter only focuses on the
    spatial features within its corresponding channel, making it a lightweight
    operation.

*   **Pointwise Convolution:** After the depthwise convolution, we have multiple
    outputs, one for each channel. To combine the information across these
    channels and build new features, we use 1x1 convolutions, also known as
    pointwise convolutions. These convolutions essentially mix and match the
    information from different channels, capturing complex cross-channel
    relationships.

**Advantages of Mobilenet:**

*   **Reduced Computation:** By separating the spatial and channel-wise
    computations, depthwise separable convolutions require significantly fewer
    multiplications and additions compared to standard convolutions. This
    translates to faster inference and lower power consumption, which are
    crucial for mobile and embedded applications.

*   **Fewer Parameters:** Since each channel has its own filter in the depthwise
    convolution, and the pointwise convolution uses small 1x1 filters, the total
    number of parameters in the model is reduced considerably. This makes the
    model smaller and more efficient to store and transfer.

Model garden offers different variants of mobilenet with different
configurations. **MobileNetV3** has **MNV3Large**, **MNV3Small** and
**MNV3EdgeTPU** variants specifically designed for Edge devices and
**MobileNetV4** variants **MobileNetV4ConvSmall**, **MobileNetV4ConvMedium**,
**MobileNetV4ConvLarge**, **MobileNetV4HybridMedium** are the Universal Models
for the Mobile Ecosystem. It provides multi - hardware mobile model support
through **MobileNetMultiMAX**, **MobileNetMultiAVG**, **MobileNetMultiAVGSeg**,
**MobileNetMultiMAXSeg**.

**Source Code:**
[MobileNet](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py)

**References:**

*   Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
    Tobias Weyand, Marco Andreetto, Hartwig Adam - [MobileNets: Efficient
    Convolutional Neural Networks for Mobile Vision
    Applications](https://arxiv.org/abs/1704.04861)

## MobileDet

MobileDet is an advanced object detection model specifically designed for mobile
and edge devices. This model makes considerable use of regular convolutions
throughout the network, especially in the early stages, to overcome the
constraints of depthwise convolution. It is optimized for mobile devices and
properly placed using neural architecture search.

**Key Components:**

Inverted Bottleneck Layers: These are the building blocks of MobileDet. They
process information efficiently by using depthwise convolutions, which focus on
individual channels within an image.

*   **Neural Architecture Search (NAS):** This technique automatically searches
    for the most optimal network design for a specific hardware platform (e.g.,
    Edge TPU). It considers different combinations of layers and operations to
    find the best balance between accuracy and speed.

*   **Strategic Regular Convolutions:** While depthwise convolutions are
    generally faster, regular convolutions can improve accuracy. NAS allows
    MobileDet to strategically incorporate regular convolutions in some parts of
    the network to boost performance without significantly impacting speed.

**Advantages of MobileDet:**

*   **Focus on Mobile Devices:** It's specifically created for mobile
    accelerators, like Google Edge TPU and Qualcomm Hexagon DSP, to achieve a
    good balance between accuracy and how fast it can process information.

*   **Optimized Architecture:** It uses a technique called neural architecture
    search to find the most efficient network design for these mobile devices.
    This search allows MobileDet to incorporate regular convolutions
    strategically, even though depthwise convolutions are generally more
    efficient for mobile devices. Regular convolutions can improve accuracy but
    may be slower, so MobileDet finds a way to use them effectively for better
    overall performance.

*   **State-of-the-Art Performance:** MobileDet achieves high accuracy in object
    detection tasks while running at low latency on various mobile platforms.

**Source Code:**
[MobileDet](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobiledet.py)

**Reference:**

*   Yunyang Xiong, Hanxiao Liu, Suyog Gupta, Berkin Akin, Gabriel Bender,
    Yongzhe Wang, Pieter-Jan Kindermans, Mingxing Tan, Vikas Singh, Bo Chen -
    [MobileDets: Searching for Object Detection Architectures for Mobile
    Accelerators](https://arxiv.org/abs/2004.14525)
