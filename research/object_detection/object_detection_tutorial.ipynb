{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Object Detection Demo\n",
    "Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import os\n",
    "import six.moves.urllib as urllib\n",
    "import sys\n",
    "import tarfile\n",
    "import tensorflow as tf\n",
    "import zipfile\n",
    "\n",
    "from collections import defaultdict\n",
    "from io import StringIO\n",
    "from matplotlib import pyplot as plt\n",
    "from PIL import Image\n",
    "\n",
    "# This is needed since the notebook is stored in the object_detection folder.\n",
    "sys.path.append(\"..\")\n",
    "from object_detection.utils import ops as utils_ops\n",
    "\n",
    "if tf.__version__ < '1.4.0':\n",
    "  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Env setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# This is needed to display the images.\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Object detection imports\n",
    "Here are the imports from the object detection module."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from utils import label_map_util\n",
    "\n",
    "from utils import visualization_utils as vis_util"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model preparation "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Variables\n",
    "\n",
    "Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  \n",
    "\n",
    "By default we use an \"SSD with Mobilenet\" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# What model to download.\n",
    "MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'\n",
    "MODEL_FILE = MODEL_NAME + '.tar.gz'\n",
    "DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'\n",
    "\n",
    "# Path to frozen detection graph. This is the actual model that is used for the object detection.\n",
    "PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'\n",
    "\n",
    "# List of the strings that is used to add correct label for each box.\n",
    "PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')\n",
    "\n",
    "NUM_CLASSES = 90"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Download Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "opener = urllib.request.URLopener()\n",
    "opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)\n",
    "tar_file = tarfile.open(MODEL_FILE)\n",
    "for file in tar_file.getmembers():\n",
    "  file_name = os.path.basename(file.name)\n",
    "  if 'frozen_inference_graph.pb' in file_name:\n",
    "    tar_file.extract(file, os.getcwd())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load a (frozen) Tensorflow model into memory."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "detection_graph = tf.Graph()\n",
    "with detection_graph.as_default():\n",
    "  od_graph_def = tf.GraphDef()\n",
    "  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:\n",
    "    serialized_graph = fid.read()\n",
    "    od_graph_def.ParseFromString(serialized_graph)\n",
    "    tf.import_graph_def(od_graph_def, name='')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Loading label map\n",
    "Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "label_map = label_map_util.load_labelmap(PATH_TO_LABELS)\n",
    "categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)\n",
    "category_index = label_map_util.create_category_index(categories)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Helper code"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_image_into_numpy_array(image):\n",
    "  (im_width, im_height) = image.size\n",
    "  return np.array(image.getdata()).reshape(\n",
    "      (im_height, im_width, 3)).astype(np.uint8)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Detection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# For the sake of simplicity we will use only 2 images:\n",
    "# image1.jpg\n",
    "# image2.jpg\n",
    "# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.\n",
    "PATH_TO_TEST_IMAGES_DIR = 'test_images'\n",
    "TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]\n",
    "\n",
    "# Size, in inches, of the output images.\n",
    "IMAGE_SIZE = (12, 8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_inference_for_single_image(image, graph):\n",
    "  with graph.as_default():\n",
    "    with tf.Session() as sess:\n",
    "      # Get handles to input and output tensors\n",
    "      ops = tf.get_default_graph().get_operations()\n",
    "      all_tensor_names = {output.name for op in ops for output in op.outputs}\n",
    "      tensor_dict = {}\n",
    "      for key in [\n",
    "          'num_detections', 'detection_boxes', 'detection_scores',\n",
    "          'detection_classes', 'detection_masks'\n",
    "      ]:\n",
    "        tensor_name = key + ':0'\n",
    "        if tensor_name in all_tensor_names:\n",
    "          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(\n",
    "              tensor_name)\n",
    "      if 'detection_masks' in tensor_dict:\n",
    "        # The following processing is only for single image\n",
    "        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])\n",
    "        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])\n",
    "        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.\n",
    "        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)\n",
    "        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])\n",
    "        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])\n",
    "        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(\n",
    "            detection_masks, detection_boxes, image.shape[0], image.shape[1])\n",
    "        detection_masks_reframed = tf.cast(\n",
    "            tf.greater(detection_masks_reframed, 0.5), tf.uint8)\n",
    "        # Follow the convention by adding back the batch dimension\n",
    "        tensor_dict['detection_masks'] = tf.expand_dims(\n",
    "            detection_masks_reframed, 0)\n",
    "      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')\n",
    "\n",
    "      # Run inference\n",
    "      output_dict = sess.run(tensor_dict,\n",
    "                             feed_dict={image_tensor: np.expand_dims(image, 0)})\n",
    "\n",
    "      # all outputs are float32 numpy arrays, so convert types as appropriate\n",
    "      output_dict['num_detections'] = int(output_dict['num_detections'][0])\n",
    "      output_dict['detection_classes'] = output_dict[\n",
    "          'detection_classes'][0].astype(np.uint8)\n",
    "      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]\n",
    "      output_dict['detection_scores'] = output_dict['detection_scores'][0]\n",
    "      if 'detection_masks' in output_dict:\n",
    "        output_dict['detection_masks'] = output_dict['detection_masks'][0]\n",
    "  return output_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "for image_path in TEST_IMAGE_PATHS:\n",
    "  image = Image.open(image_path)\n",
    "  # the array based representation of the image will be used later in order to prepare the\n",
    "  # result image with boxes and labels on it.\n",
    "  image_np = load_image_into_numpy_array(image)\n",
    "  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]\n",
    "  image_np_expanded = np.expand_dims(image_np, axis=0)\n",
    "  # Actual detection.\n",
    "  output_dict = run_inference_for_single_image(image_np, detection_graph)\n",
    "  # Visualization of the results of a detection.\n",
    "  vis_util.visualize_boxes_and_labels_on_image_array(\n",
    "      image_np,\n",
    "      output_dict['detection_boxes'],\n",
    "      output_dict['detection_classes'],\n",
    "      output_dict['detection_scores'],\n",
    "      category_index,\n",
    "      instance_masks=output_dict.get('detection_masks'),\n",
    "      use_normalized_coordinates=True,\n",
    "      line_thickness=8)\n",
    "  plt.figure(figsize=IMAGE_SIZE)\n",
    "  plt.imshow(image_np)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "version": "0.3.2"
  },
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
