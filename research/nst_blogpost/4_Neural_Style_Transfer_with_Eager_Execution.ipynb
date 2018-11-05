{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Neural Style Transfer with Eager Execution",
      "version": "0.3.2",
      "provenance": [],
      "private_outputs": true,
      "collapsed_sections": [],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "metadata": {
        "id": "jo5PziEC4hWs",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "# Neural Style Transfer with tf.keras\n",
        "\n",
        "<table class=\"tfo-notebook-buttons\" align=\"left\">\n",
        "  <td>\n",
        "    <a target=\"_blank\" href=\"https://colab.research.google.com/github/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb\"><img src=\"https://www.tensorflow.org/images/colab_logo_32px.png\" />Run in Google Colab</a>\n",
        "  </td>\n",
        "  <td>\n",
        "    <a target=\"_blank\" href=\"https://github.com/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb\"><img src=\"https://www.tensorflow.org/images/GitHub-Mark-32px.png\" />View source on GitHub</a>\n",
        "  </td>\n",
        "</table>"
      ]
    },
    {
      "metadata": {
        "id": "aDyGj8DmXCJI",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Overview\n",
        "\n",
        "In this tutorial, we will learn how to use deep learning to compose images in the style of another image (ever wish you could paint like Picasso or Van Gogh?). This is known as **neural style transfer**! This is a technique outlined in [Leon A. Gatys' paper, A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), which is a great read, and you should definitely check it out. \n",
        "\n",
        "But, what is neural style transfer?\n",
        "\n",
        "Neural style transfer is an optimization technique used to take three images, a **content** image, a **style reference** image (such as an artwork by a famous painter), and the **input** image you want to style -- and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.\n",
        "\n",
        "\n",
        "For example, let’s take an image of this turtle and Katsushika Hokusai's *The Great Wave off Kanagawa*:\n",
        "\n",
        "<img src=\"https://github.com/tensorflow/models/blob/master/research/nst_blogpost/Green_Sea_Turtle_grazing_seagrass.jpg?raw=1\" alt=\"Drawing\" style=\"width: 200px;\"/>\n",
        "<img src=\"https://github.com/tensorflow/models/blob/master/research/nst_blogpost/The_Great_Wave_off_Kanagawa.jpg?raw=1\" alt=\"Drawing\" style=\"width: 200px;\"/>\n",
        "\n",
        "[Image of Green Sea Turtle](https://commons.wikimedia.org/wiki/File:Green_Sea_Turtle_grazing_seagrass.jpg)\n",
        "-By P.Lindgren [CC BY-SA 3.0  (https://creativecommons.org/licenses/by-sa/3.0)], from Wikimedia Common\n",
        "\n",
        "\n",
        "Now how would it look like if Hokusai decided to paint the picture of this Turtle exclusively with this style? Something like this?\n",
        "\n",
        "<img src=\"https://github.com/tensorflow/models/blob/master/research/nst_blogpost/wave_turtle.png?raw=1\" alt=\"Drawing\" style=\"width: 500px;\"/>\n",
        "\n",
        "Is this magic or just deep learning? Fortunately, this doesn’t involve any witchcraft: style transfer is a fun and interesting technique that showcases the capabilities and internal representations of neural networks.  \n",
        "\n",
        "The principle of neural style transfer is to define two distance functions, one that describes how different the content of two images are , $L_{content}$, and one that describes the difference between two images in terms of their style, $L_{style}$. Then, given three images, a desired style image, a desired content image, and the input image (initialized with the content image), we try to transform the input image to minimize the content distance with the content image and its style distance with the style image. \n",
        "In summary, we’ll take the base input image, a content image that we want to match, and the style image that we want to match. We’ll transform the base input image by minimizing the content and style distances (losses) with backpropagation, creating an image that matches the content of the content image and the style of the style image. \n",
        "\n",
        "### Specific concepts that will be covered:\n",
        "In the process, we will build practical experience and develop intuition around the following concepts\n",
        "\n",
        "* **Eager Execution** - use TensorFlow's imperative programming environment that evaluates operations immediately \n",
        "  * [Learn more about eager execution](https://www.tensorflow.org/programmers_guide/eager)\n",
        "  * [See it in action](https://www.tensorflow.org/get_started/eager)\n",
        "* ** Using [Functional API](https://keras.io/getting-started/functional-api-guide/) to define a model** - we'll build a subset of our model that will give us access to the necessary intermediate activations using the Functional API \n",
        "* **Leveraging feature maps of a pretrained model** - Learn how to use pretrained models and their feature maps \n",
        "* **Create custom training loops** - we'll examine how to set up an optimizer to minimize a given loss with respect to input parameters\n",
        "\n",
        "### We will follow the general steps to perform style transfer:\n",
        "\n",
        "1. Visualize data\n",
        "2. Basic Preprocessing/preparing our data\n",
        "3. Set up loss functions \n",
        "4. Create model\n",
        "5. Optimize for loss function\n",
        "\n",
        "**Audience:** This post is geared towards intermediate users who are comfortable with basic machine learning concepts. To get the most out of this post, you should: \n",
        "* Read [Gatys' paper](https://arxiv.org/abs/1508.06576) - we'll explain along the way, but the paper will provide a more thorough understanding of the task\n",
        "* [Understand reducing loss with gradient descent](https://developers.google.com/machine-learning/crash-course/reducing-loss/gradient-descent)\n",
        "\n",
        "**Time Estimated**: 30 min\n"
      ]
    },
    {
      "metadata": {
        "id": "U8ajP_u73s6m",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Setup\n",
        "\n",
        "### Download Images"
      ]
    },
    {
      "metadata": {
        "id": "riWE_b8k3s6o",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "import os\n",
        "img_dir = '/tmp/nst'\n",
        "if not os.path.exists(img_dir):\n",
        "    os.makedirs(img_dir)\n",
        "!wget --quiet -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg\n",
        "!wget --quiet -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg\n",
        "!wget --quiet -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg\n",
        "!wget --quiet -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg\n",
        "!wget --quiet -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg\n",
        "!wget --quiet -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "eqxUicSPUOP6",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Import and configure modules"
      ]
    },
    {
      "metadata": {
        "id": "sc1OLbOWhPCO",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import matplotlib as mpl\n",
        "mpl.rcParams['figure.figsize'] = (10,10)\n",
        "mpl.rcParams['axes.grid'] = False\n",
        "\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "import time\n",
        "import functools"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "RYEjlrYk3s6w",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "import tensorflow as tf\n",
        "import tensorflow.contrib.eager as tfe\n",
        "\n",
        "from tensorflow.python.keras.preprocessing import image as kp_image\n",
        "from tensorflow.python.keras import models \n",
        "from tensorflow.python.keras import losses\n",
        "from tensorflow.python.keras import layers\n",
        "from tensorflow.python.keras import backend as K"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "L7sjDODq67HQ",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "We’ll begin by enabling [eager execution](https://www.tensorflow.org/guide/eager). Eager execution allows us to work through this technique in the clearest and most readable way. "
      ]
    },
    {
      "metadata": {
        "id": "sfjsSAtNrqQx",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "tf.enable_eager_execution()\n",
        "print(\"Eager execution: {}\".format(tf.executing_eagerly()))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "IOiGrIV1iERH",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "# Set up some global values here\n",
        "content_path = '/tmp/nst/Green_Sea_Turtle_grazing_seagrass.jpg'\n",
        "style_path = '/tmp/nst/The_Great_Wave_off_Kanagawa.jpg'"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "xE4Yt8nArTeR",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Visualize the input"
      ]
    },
    {
      "metadata": {
        "id": "3TLljcwv5qZs",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def load_img(path_to_img):\n",
        "  max_dim = 512\n",
        "  img = Image.open(path_to_img)\n",
        "  long = max(img.size)\n",
        "  scale = max_dim/long\n",
        "  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)\n",
        "  \n",
        "  img = kp_image.img_to_array(img)\n",
        "  \n",
        "  # We need to broadcast the image array such that it has a batch dimension \n",
        "  img = np.expand_dims(img, axis=0)\n",
        "  return img"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "vupl0CI18aAG",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def imshow(img, title=None):\n",
        "  # Remove the batch dimension\n",
        "  out = np.squeeze(img, axis=0)\n",
        "  # Normalize for display \n",
        "  out = out.astype('uint8')\n",
        "  plt.imshow(out)\n",
        "  if title is not None:\n",
        "    plt.title(title)\n",
        "  plt.imshow(out)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "2yAlRzJZrWM3",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "These are input content and style images. We hope to \"create\" an image with the content of our content image, but with the style of the style image. "
      ]
    },
    {
      "metadata": {
        "id": "_UWQmeEaiKkP",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(10,10))\n",
        "\n",
        "content = load_img(content_path).astype('uint8')\n",
        "style = load_img(style_path).astype('uint8')\n",
        "\n",
        "plt.subplot(1, 2, 1)\n",
        "imshow(content, 'Content Image')\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "imshow(style, 'Style Image')\n",
        "plt.show()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "7qMVNvEsK-_D",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Prepare the data\n",
        "Let's create methods that will allow us to load and preprocess our images easily. We perform the same preprocessing process as are expected according to the VGG training process. VGG networks are trained on image with each channel normalized by `mean = [103.939, 116.779, 123.68]`and with channels BGR."
      ]
    },
    {
      "metadata": {
        "id": "hGwmTwJNmv2a",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def load_and_process_img(path_to_img):\n",
        "  img = load_img(path_to_img)\n",
        "  img = tf.keras.applications.vgg19.preprocess_input(img)\n",
        "  return img"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "xCgooqs6tAka",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "In order to view the outputs of our optimization, we are required to perform the inverse preprocessing step. Furthermore, since our optimized image may take its values anywhere between $- \\infty$ and $\\infty$, we must clip to maintain our values from within the 0-255 range.   "
      ]
    },
    {
      "metadata": {
        "id": "mjzlKRQRs_y2",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def deprocess_img(processed_img):\n",
        "  x = processed_img.copy()\n",
        "  if len(x.shape) == 4:\n",
        "    x = np.squeeze(x, 0)\n",
        "  assert len(x.shape) == 3, (\"Input to deprocess image must be an image of \"\n",
        "                             \"dimension [1, height, width, channel] or [height, width, channel]\")\n",
        "  if len(x.shape) != 3:\n",
        "    raise ValueError(\"Invalid input to deprocessing image\")\n",
        "  \n",
        "  # perform the inverse of the preprocessiing step\n",
        "  x[:, :, 0] += 103.939\n",
        "  x[:, :, 1] += 116.779\n",
        "  x[:, :, 2] += 123.68\n",
        "  x = x[:, :, ::-1]\n",
        "\n",
        "  x = np.clip(x, 0, 255).astype('uint8')\n",
        "  return x"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "GEwZ7FlwrjoZ",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Define content and style representations\n",
        "In order to get both the content and style representations of our image, we will look at some intermediate layers within our model. As we go deeper into the model, these intermediate layers represent higher and higher order features. In this case, we are using the network architecture VGG19, a pretrained image classification network. These intermediate layers are necessary to define the representation of content and style from our images. For an input image, we will try to match the corresponding style and content target representations at these intermediate layers. \n",
        "\n",
        "#### Why intermediate layers?\n",
        "\n",
        "You may be wondering why these intermediate outputs within our pretrained image classification network allow us to define style and content representations. At a high level, this phenomenon can be explained by the fact that in order for a network to perform image classification (which our network has been trained to do), it must understand the image. This involves taking the raw image as input pixels and building an internal representation through transformations that turn the raw image pixels into a complex understanding of the features present within the image. This is also partly why convolutional neural networks are able to generalize well: they’re able to capture the invariances and defining features within classes (e.g., cats vs. dogs) that are agnostic to background noise and other nuisances. Thus, somewhere between where the raw image is fed in and the classification label is output, the model serves as a complex feature extractor; hence by accessing intermediate layers, we’re able to describe the content and style of input images. \n",
        "\n",
        "\n",
        "Specifically we’ll pull out these intermediate layers from our network: \n"
      ]
    },
    {
      "metadata": {
        "id": "N4-8eUp_Kc-j",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "# Content layer where will pull our feature maps\n",
        "content_layers = ['block5_conv2'] \n",
        "\n",
        "# Style layer we are interested in\n",
        "style_layers = ['block1_conv1',\n",
        "                'block2_conv1',\n",
        "                'block3_conv1', \n",
        "                'block4_conv1', \n",
        "                'block5_conv1'\n",
        "               ]\n",
        "\n",
        "num_content_layers = len(content_layers)\n",
        "num_style_layers = len(style_layers)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "Jt3i3RRrJiOX",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Build the Model \n",
        "In this case, we load [VGG19](https://keras.io/applications/#vgg19), and feed in our input tensor to the model. This will allow us to extract the feature maps (and subsequently the content and style representations) of the content, style, and generated images.\n",
        "\n",
        "We use VGG19, as suggested in the paper. In addition, since VGG19 is a relatively simple model (compared with ResNet, Inception, etc) the feature maps actually work better for style transfer. "
      ]
    },
    {
      "metadata": {
        "id": "v9AnzEUU6hhx",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "In order to access the intermediate layers corresponding to our style and content feature maps, we get the corresponding outputs and using the Keras [**Functional API**](https://keras.io/getting-started/functional-api-guide/), we define our model with the desired output activations. \n",
        "\n",
        "With the Functional API defining a model simply involves defining the input and output: \n",
        "\n",
        "`model = Model(inputs, outputs)`"
      ]
    },
    {
      "metadata": {
        "id": "nfec6MuMAbPx",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def get_model():\n",
        "  \"\"\" Creates our model with access to intermediate layers. \n",
        "  \n",
        "  This function will load the VGG19 model and access the intermediate layers. \n",
        "  These layers will then be used to create a new model that will take input image\n",
        "  and return the outputs from these intermediate layers from the VGG model. \n",
        "  \n",
        "  Returns:\n",
        "    returns a keras model that takes image inputs and outputs the style and \n",
        "      content intermediate layers. \n",
        "  \"\"\"\n",
        "  # Load our model. We load pretrained VGG, trained on imagenet data\n",
        "  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')\n",
        "  vgg.trainable = False\n",
        "  # Get output layers corresponding to style and content layers \n",
        "  style_outputs = [vgg.get_layer(name).output for name in style_layers]\n",
        "  content_outputs = [vgg.get_layer(name).output for name in content_layers]\n",
        "  model_outputs = style_outputs + content_outputs\n",
        "  # Build model \n",
        "  return models.Model(vgg.input, model_outputs)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "kl6eFGa7-OtV",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "In the above code snippet, we’ll load our pretrained image classification network. Then we grab the layers of interest as we defined earlier. Then we define a Model by setting the model’s inputs to an image and the outputs to the outputs of the style and content layers. In other words, we created a model that will take an input image and output the content and style intermediate layers! \n"
      ]
    },
    {
      "metadata": {
        "id": "vJdYvJTZ4bdS",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Define and create our loss functions (content and style distances)"
      ]
    },
    {
      "metadata": {
        "id": "F2Hcepii7_qh",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Content Loss"
      ]
    },
    {
      "metadata": {
        "id": "1FvH-gwXi4nq",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "Our content loss definition is actually quite simple. We’ll pass the network both the desired content image and our base input image. This will return the intermediate layer outputs (from the layers defined above) from our model. Then we simply take the euclidean distance between the two intermediate representations of those images.  \n",
        "\n",
        "More formally, content loss is a function that describes the distance of content from our output image $x$ and our content image, $p$. Let $C_{nn}$ be a pre-trained deep convolutional neural network. Again, in this case we use [VGG19](https://keras.io/applications/#vgg19). Let $X$ be any image, then $C_{nn}(X)$ is the network fed by X. Let $F^l_{ij}(x) \\in C_{nn}(x)$ and $P^l_{ij}(p) \\in C_{nn}(p)$ describe the respective intermediate feature representation of the network with inputs $x$ and $p$ at layer $l$. Then we describe the content distance (loss) formally as: $$L^l_{content}(p, x) = \\sum_{i, j} (F^l_{ij}(x) - P^l_{ij}(p))^2$$\n",
        "\n",
        "We perform backpropagation in the usual way such that we minimize this content loss. We thus change the initial image until it generates a similar response in a certain layer (defined in content_layer) as the original content image.\n",
        "\n",
        "This can be implemented quite simply. Again it will take as input the feature maps at a layer L in a network fed by x, our input image, and p, our content image, and return the content distance.\n",
        "\n"
      ]
    },
    {
      "metadata": {
        "id": "6KsbqPA8J9DY",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Computing content loss\n",
        "We will actually add our content losses at each desired layer. This way, each iteration when we feed our input image through the model (which in eager is simply `model(input_image)`!) all the content losses through the model will be properly compute and because we are executing eagerly, all the gradients will be computed. "
      ]
    },
    {
      "metadata": {
        "id": "d2mf7JwRMkCd",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def get_content_loss(base_content, target):\n",
        "  return tf.reduce_mean(tf.square(base_content - target))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "lGUfttK9F8d5",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Style Loss"
      ]
    },
    {
      "metadata": {
        "id": "I6XtkGK_YGD1",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "Computing style loss is a bit more involved, but follows the same principle, this time feeding our network the base input image and the style image. However, instead of comparing the raw intermediate outputs of the base input image and the style image, we instead compare the Gram matrices of the two outputs. \n",
        "\n",
        "Mathematically, we describe the style loss of the base input image, $x$, and the style image, $a$, as the distance between the style representation (the gram matrices) of these images. We describe the style representation of an image as the correlation between different filter responses given by the Gram matrix  $G^l$, where $G^l_{ij}$ is the inner product between the vectorized feature map $i$ and $j$ in layer $l$. We can see that $G^l_{ij}$ generated over the feature map for a given image represents the correlation between feature maps $i$ and $j$. \n",
        "\n",
        "To generate a style for our base input image, we perform gradient descent from the content image to transform it into an image that matches the style representation of the original image. We do so by minimizing the mean squared distance between the feature correlation map of the style image and the input image. The contribution of each layer to the total style loss is described by\n",
        "$$E_l = \\frac{1}{4N_l^2M_l^2} \\sum_{i,j}(G^l_{ij} - A^l_{ij})^2$$\n",
        "\n",
        "where $G^l_{ij}$ and $A^l_{ij}$ are the respective style representation in layer $l$ of $x$ and $a$. $N_l$ describes the number of feature maps, each of size $M_l = height * width$. Thus, the total style loss across each layer is \n",
        "$$L_{style}(a, x) = \\sum_{l \\in L} w_l E_l$$\n",
        "where we weight the contribution of each layer's loss by some factor $w_l$. In our case, we weight each layer equally ($w_l =\\frac{1}{|L|}$)"
      ]
    },
    {
      "metadata": {
        "id": "F21Hm61yLKk5",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Computing style loss\n",
        "Again, we implement our loss as a distance metric . "
      ]
    },
    {
      "metadata": {
        "id": "N7MOqwKLLke8",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def gram_matrix(input_tensor):\n",
        "  # We make the image channels first \n",
        "  channels = int(input_tensor.shape[-1])\n",
        "  a = tf.reshape(input_tensor, [-1, channels])\n",
        "  n = tf.shape(a)[0]\n",
        "  gram = tf.matmul(a, a, transpose_a=True)\n",
        "  return gram / tf.cast(n, tf.float32)\n",
        "\n",
        "def get_style_loss(base_style, gram_target):\n",
        "  \"\"\"Expects two images of dimension h, w, c\"\"\"\n",
        "  # height, width, num filters of each layer\n",
        "  # We scale the loss at a given layer by the size of the feature map and the number of filters\n",
        "  height, width, channels = base_style.get_shape().as_list()\n",
        "  gram_style = gram_matrix(base_style)\n",
        "  \n",
        "  return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "pXIUX6czZABh",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Apply style transfer to our images\n"
      ]
    },
    {
      "metadata": {
        "id": "y9r8Lyjb_m0u",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Run Gradient Descent \n",
        "If you aren't familiar with gradient descent/backpropagation or need a refresher, you should definitely check out this [awesome resource](https://developers.google.com/machine-learning/crash-course/reducing-loss/gradient-descent).\n",
        "\n",
        "In this case, we use the [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)* optimizer in order to minimize our loss. We iteratively update our output image such that it minimizes our loss: we don't update the weights associated with our network, but instead we train our input image to minimize loss. In order to do this, we must know how we calculate our loss and gradients. \n",
        "\n",
        "\\* Note that L-BFGS, which if you are familiar with this algorithm is recommended, isn’t used in this tutorial because a primary motivation behind this tutorial was to illustrate best practices with eager execution, and, by using Adam, we can demonstrate the autograd/gradient tape functionality with custom training loops.\n"
      ]
    },
    {
      "metadata": {
        "id": "-kGzV6LTp4CU",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "We’ll define a little helper function that will load our content and style image, feed them forward through our network, which will then output the content and style feature representations from our model. "
      ]
    },
    {
      "metadata": {
        "id": "O-lj5LxgtmnI",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def get_feature_representations(model, content_path, style_path):\n",
        "  \"\"\"Helper function to compute our content and style feature representations.\n",
        "\n",
        "  This function will simply load and preprocess both the content and style \n",
        "  images from their path. Then it will feed them through the network to obtain\n",
        "  the outputs of the intermediate layers. \n",
        "  \n",
        "  Arguments:\n",
        "    model: The model that we are using.\n",
        "    content_path: The path to the content image.\n",
        "    style_path: The path to the style image\n",
        "    \n",
        "  Returns:\n",
        "    returns the style features and the content features. \n",
        "  \"\"\"\n",
        "  # Load our images in \n",
        "  content_image = load_and_process_img(content_path)\n",
        "  style_image = load_and_process_img(style_path)\n",
        "  \n",
        "  # batch compute content and style features\n",
        "  style_outputs = model(style_image)\n",
        "  content_outputs = model(content_image)\n",
        "  \n",
        "  \n",
        "  # Get the style and content feature representations from our model  \n",
        "  style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]\n",
        "  content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]\n",
        "  return style_features, content_features"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "3DopXw7-lFHa",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Computing the loss and gradients\n",
        "Here we use [**tf.GradientTape**](https://www.tensorflow.org/programmers_guide/eager#computing_gradients) to compute the gradient. It allows us to take advantage of the automatic differentiation available by tracing operations for computing the gradient later. It records the operations during the forward pass and then is able to compute the gradient of our loss function with respect to our input image for the backwards pass."
      ]
    },
    {
      "metadata": {
        "id": "oVDhSo8iJunf",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):\n",
        "  \"\"\"This function will compute the loss total loss.\n",
        "  \n",
        "  Arguments:\n",
        "    model: The model that will give us access to the intermediate layers\n",
        "    loss_weights: The weights of each contribution of each loss function. \n",
        "      (style weight, content weight, and total variation weight)\n",
        "    init_image: Our initial base image. This image is what we are updating with \n",
        "      our optimization process. We apply the gradients wrt the loss we are \n",
        "      calculating to this image.\n",
        "    gram_style_features: Precomputed gram matrices corresponding to the \n",
        "      defined style layers of interest.\n",
        "    content_features: Precomputed outputs from defined content layers of \n",
        "      interest.\n",
        "      \n",
        "  Returns:\n",
        "    returns the total loss, style loss, content loss, and total variational loss\n",
        "  \"\"\"\n",
        "  style_weight, content_weight = loss_weights\n",
        "  \n",
        "  # Feed our init image through our model. This will give us the content and \n",
        "  # style representations at our desired layers. Since we're using eager\n",
        "  # our model is callable just like any other function!\n",
        "  model_outputs = model(init_image)\n",
        "  \n",
        "  style_output_features = model_outputs[:num_style_layers]\n",
        "  content_output_features = model_outputs[num_style_layers:]\n",
        "  \n",
        "  style_score = 0\n",
        "  content_score = 0\n",
        "\n",
        "  # Accumulate style losses from all layers\n",
        "  # Here, we equally weight each contribution of each loss layer\n",
        "  weight_per_style_layer = 1.0 / float(num_style_layers)\n",
        "  for target_style, comb_style in zip(gram_style_features, style_output_features):\n",
        "    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)\n",
        "    \n",
        "  # Accumulate content losses from all layers \n",
        "  weight_per_content_layer = 1.0 / float(num_content_layers)\n",
        "  for target_content, comb_content in zip(content_features, content_output_features):\n",
        "    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)\n",
        "  \n",
        "  style_score *= style_weight\n",
        "  content_score *= content_weight\n",
        "\n",
        "  # Get total loss\n",
        "  loss = style_score + content_score \n",
        "  return loss, style_score, content_score"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "r5XTvbP6nJQa",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "Then computing the gradients is easy:"
      ]
    },
    {
      "metadata": {
        "id": "fwzYeOqOUH9_",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def compute_grads(cfg):\n",
        "  with tf.GradientTape() as tape: \n",
        "    all_loss = compute_loss(**cfg)\n",
        "  # Compute gradients wrt input image\n",
        "  total_loss = all_loss[0]\n",
        "  return tape.gradient(total_loss, cfg['init_image']), all_loss"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "T9yKu2PLlBIE",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Optimization loop"
      ]
    },
    {
      "metadata": {
        "id": "pj_enNo6tACQ",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "import IPython.display\n",
        "\n",
        "def run_style_transfer(content_path, \n",
        "                       style_path,\n",
        "                       num_iterations=1000,\n",
        "                       content_weight=1e3, \n",
        "                       style_weight=1e-2): \n",
        "  # We don't need to (or want to) train any layers of our model, so we set their\n",
        "  # trainable to false. \n",
        "  model = get_model() \n",
        "  for layer in model.layers:\n",
        "    layer.trainable = False\n",
        "  \n",
        "  # Get the style and content feature representations (from our specified intermediate layers) \n",
        "  style_features, content_features = get_feature_representations(model, content_path, style_path)\n",
        "  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]\n",
        "  \n",
        "  # Set initial image\n",
        "  init_image = load_and_process_img(content_path)\n",
        "  init_image = tfe.Variable(init_image, dtype=tf.float32)\n",
        "  # Create our optimizer\n",
        "  opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)\n",
        "\n",
        "  # For displaying intermediate images \n",
        "  iter_count = 1\n",
        "  \n",
        "  # Store our best result\n",
        "  best_loss, best_img = float('inf'), None\n",
        "  \n",
        "  # Create a nice config \n",
        "  loss_weights = (style_weight, content_weight)\n",
        "  cfg = {\n",
        "      'model': model,\n",
        "      'loss_weights': loss_weights,\n",
        "      'init_image': init_image,\n",
        "      'gram_style_features': gram_style_features,\n",
        "      'content_features': content_features\n",
        "  }\n",
        "    \n",
        "  # For displaying\n",
        "  num_rows = 2\n",
        "  num_cols = 5\n",
        "  display_interval = num_iterations/(num_rows*num_cols)\n",
        "  start_time = time.time()\n",
        "  global_start = time.time()\n",
        "  \n",
        "  norm_means = np.array([103.939, 116.779, 123.68])\n",
        "  min_vals = -norm_means\n",
        "  max_vals = 255 - norm_means   \n",
        "  \n",
        "  imgs = []\n",
        "  for i in range(num_iterations):\n",
        "    grads, all_loss = compute_grads(cfg)\n",
        "    loss, style_score, content_score = all_loss\n",
        "    opt.apply_gradients([(grads, init_image)])\n",
        "    clipped = tf.clip_by_value(init_image, min_vals, max_vals)\n",
        "    init_image.assign(clipped)\n",
        "    end_time = time.time() \n",
        "    \n",
        "    if loss < best_loss:\n",
        "      # Update best loss and best image from total loss. \n",
        "      best_loss = loss\n",
        "      best_img = deprocess_img(init_image.numpy())\n",
        "\n",
        "    if i % display_interval== 0:\n",
        "      start_time = time.time()\n",
        "      \n",
        "      # Use the .numpy() method to get the concrete numpy array\n",
        "      plot_img = init_image.numpy()\n",
        "      plot_img = deprocess_img(plot_img)\n",
        "      imgs.append(plot_img)\n",
        "      IPython.display.clear_output(wait=True)\n",
        "      IPython.display.display_png(Image.fromarray(plot_img))\n",
        "      print('Iteration: {}'.format(i))        \n",
        "      print('Total loss: {:.4e}, ' \n",
        "            'style loss: {:.4e}, '\n",
        "            'content loss: {:.4e}, '\n",
        "            'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))\n",
        "  print('Total time: {:.4f}s'.format(time.time() - global_start))\n",
        "  IPython.display.clear_output(wait=True)\n",
        "  plt.figure(figsize=(14,4))\n",
        "  for i,img in enumerate(imgs):\n",
        "      plt.subplot(num_rows,num_cols,i+1)\n",
        "      plt.imshow(img)\n",
        "      plt.xticks([])\n",
        "      plt.yticks([])\n",
        "      \n",
        "  return best_img, best_loss "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "vSVMx4burydi",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "best, best_loss = run_style_transfer(content_path, \n",
        "                                     style_path, num_iterations=1000)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "dzJTObpsO3TZ",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "Image.fromarray(best)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "dCXQ9vSnQbDy",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "To download the image from Colab uncomment the following code:"
      ]
    },
    {
      "metadata": {
        "id": "SSH6OpyyQn7w",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#from google.colab import files\n",
        "#files.download('wave_turtle.png')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "LwiZfCW0AZwt",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Visualize outputs\n",
        "We \"deprocess\" the output image in order to remove the processing that was applied to it. "
      ]
    },
    {
      "metadata": {
        "id": "lqTQN1PjulV9",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def show_results(best_img, content_path, style_path, show_large_final=True):\n",
        "  plt.figure(figsize=(10, 5))\n",
        "  content = load_img(content_path) \n",
        "  style = load_img(style_path)\n",
        "\n",
        "  plt.subplot(1, 2, 1)\n",
        "  imshow(content, 'Content Image')\n",
        "\n",
        "  plt.subplot(1, 2, 2)\n",
        "  imshow(style, 'Style Image')\n",
        "\n",
        "  if show_large_final: \n",
        "    plt.figure(figsize=(10, 10))\n",
        "\n",
        "    plt.imshow(best_img)\n",
        "    plt.title('Output Image')\n",
        "    plt.show()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "i6d6O50Yvs6a",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "show_results(best, content_path, style_path)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "tyGMmWh2Pss8",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Try it on other images\n",
        "Image of Tuebingen \n",
        "\n",
        "Photo By: Andreas Praefcke [GFDL (http://www.gnu.org/copyleft/fdl.html) or CC BY 3.0  (https://creativecommons.org/licenses/by/3.0)], from Wikimedia Commons"
      ]
    },
    {
      "metadata": {
        "id": "x2TePU39k9lb",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Starry night + Tuebingen"
      ]
    },
    {
      "metadata": {
        "id": "ES9dC6ZyJBD2",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "best_starry_night, best_loss = run_style_transfer('/tmp/nst/Tuebingen_Neckarfront.jpg',\n",
        "                                                  '/tmp/nst/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "X8w8WLkKvzXu",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "show_results(best_starry_night, '/tmp/nst/Tuebingen_Neckarfront.jpg',\n",
        "             '/tmp/nst/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "QcXwvViek4Br",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Pillars of Creation + Tuebingen"
      ]
    },
    {
      "metadata": {
        "id": "vJ3u2U-gGmgP",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "best_poc_tubingen, best_loss = run_style_transfer('/tmp/nst/Tuebingen_Neckarfront.jpg', \n",
        "                                                  '/tmp/nst/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "pQUq3KxpGv2O",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "show_results(best_poc_tubingen, \n",
        "             '/tmp/nst/Tuebingen_Neckarfront.jpg',\n",
        "             '/tmp/nst/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "bTZdTOdW3s8H",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Kandinsky Composition 7 + Tuebingen"
      ]
    },
    {
      "metadata": {
        "id": "bt9mbQfl7exl",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "best_kandinsky_tubingen, best_loss = run_style_transfer('/tmp/nst/Tuebingen_Neckarfront.jpg', \n",
        "                                                  '/tmp/nst/Vassily_Kandinsky,_1913_-_Composition_7.jpg')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "Qnz8HeXSXg6P",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "show_results(best_kandinsky_tubingen, \n",
        "             '/tmp/nst/Tuebingen_Neckarfront.jpg',\n",
        "             '/tmp/nst/Vassily_Kandinsky,_1913_-_Composition_7.jpg')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "cg68lW2A3s8N",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Pillars of Creation + Sea Turtle"
      ]
    },
    {
      "metadata": {
        "id": "dl0DUot_bFST",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "best_poc_turtle, best_loss = run_style_transfer('/tmp/nst/Green_Sea_Turtle_grazing_seagrass.jpg', \n",
        "                                                  '/tmp/nst/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "UzJfE0I1bQn8",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "show_results(best_poc_turtle, \n",
        "             '/tmp/nst/Green_Sea_Turtle_grazing_seagrass.jpg',\n",
        "             '/tmp/nst/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "sElaeNX-4Vnc",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Key Takeaways\n",
        "\n",
        "### What we covered:\n",
        "\n",
        "* We built several different loss functions and used backpropagation to transform our input image in order to minimize these losses\n",
        "  * In order to do this we had to load in a **pretrained model** and use its learned feature maps to describe the content and style representation of our images.\n",
        "    * Our main loss functions were primarily computing the distance in terms of these different representations\n",
        "* We implemented this with a custom model and **eager execution**\n",
        "  * We built our custom model with the Functional API \n",
        "  * Eager execution allows us to dynamically work with tensors, using a natural python control flow\n",
        "  * We manipulated tensors directly, which makes debugging and working with tensors easier. \n",
        "* We iteratively updated our image by applying our optimizers update rules using **tf.gradient**. The optimizer minimized a given loss with respect to our input image. "
      ]
    },
    {
      "metadata": {
        "id": "U-y02GWonqnD",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "\n",
        "**[Image of Tuebingen](https://commons.wikimedia.org/wiki/File:Tuebingen_Neckarfront.jpg)** \n",
        "Photo By: Andreas Praefcke [GFDL (http://www.gnu.org/copyleft/fdl.html) or CC BY 3.0  (https://creativecommons.org/licenses/by/3.0)], from Wikimedia Commons\n",
        "\n",
        "**[Image of Green Sea Turtle](https://commons.wikimedia.org/wiki/File:Green_Sea_Turtle_grazing_seagrass.jpg)**\n",
        "By P.Lindgren [CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0)], from Wikimedia Commons\n",
        "\n"
      ]
    },
    {
      "metadata": {
        "id": "IpUD9W6ZkeyM",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}
