"""Create an image classification graph.

Script to download a pre-trained image classifier and tweak it so that
the model accepts raw bytes of an encoded image.

Doing so involves some model-specific normalization of an image.
Ideally, this would have been part of the image classifier model,
but the particular model being used didn't include this normalization,
so this script does the necessary tweaking.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import urllib
import os
import zipfile
import tensorflow as tf

URL = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
LABELS_FILE = 'imagenet_comp_graph_label_strings.txt'
GRAPH_FILE = 'tensorflow_inception_graph.pb'

GRAPH_INPUT_TENSOR = 'input:0'
GRAPH_PROBABILITIES_TENSOR = 'output:0'

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
MEAN = 117
SCALE = 1

LOCAL_DIR = 'src/main/resources'


def download():
  print('Downloading %s' % URL)
  zip_filename, _ = urllib.request.urlretrieve(URL)
  with zipfile.ZipFile(zip_filename) as zip:
    zip.extract(LABELS_FILE)
    zip.extract(GRAPH_FILE)
    os.rename(LABELS_FILE, os.path.join(LOCAL_DIR, 'labels.txt'))
    os.rename(GRAPH_FILE, os.path.join(LOCAL_DIR, 'graph.pb'))


def create_graph_to_decode_and_normalize_image():
  """See file docstring.

  Returns:
    input: The placeholder to feed the raw bytes of an encoded image.
    y: A Tensor (the decoded, normalized image) to be fed to the graph.
  """
  image = tf.placeholder(tf.string, shape=(), name='encoded_image_bytes')
  with tf.name_scope("preprocess"):
    y = tf.image.decode_image(image, channels=3)
    y = tf.cast(y, tf.float32)
    y = tf.expand_dims(y, axis=0)
    y = tf.image.resize_bilinear(y, (IMAGE_HEIGHT, IMAGE_WIDTH))
    y = (y - MEAN) / SCALE
  return (image, y)


def patch_graph():
  """Create graph.pb that applies the model in URL to raw image bytes."""
  with tf.Graph().as_default() as g:
    input_image, image_normalized = create_graph_to_decode_and_normalize_image()
    original_graph_def = tf.GraphDef()
    with open(os.path.join(LOCAL_DIR, 'graph.pb')) as f:
      original_graph_def.ParseFromString(f.read())
    softmax = tf.import_graph_def(
        original_graph_def,
        name='inception',
        input_map={GRAPH_INPUT_TENSOR: image_normalized},
        return_elements=[GRAPH_PROBABILITIES_TENSOR])
    # We're constructing a graph that accepts a single image (as opposed to a
    # batch of images), so might as well make the output be a vector of
    # probabilities, instead of a batch of vectors with batch size 1.
    output_probabilities = tf.squeeze(softmax, name='probabilities')
    # Overwrite the graph.
    with open(os.path.join(LOCAL_DIR, 'graph.pb'), 'w') as f:
      f.write(g.as_graph_def().SerializeToString())
    print('------------------------------------------------------------')
    print('MODEL GRAPH  : graph.pb')
    print('LABELS       : labels.txt')
    print('INPUT TENSOR : %s' % input_image.op.name)
    print('OUTPUT TENSOR: %s' % output_probabilities.op.name)


if __name__ == '__main__':
  if not os.path.exists(LOCAL_DIR):
    os.makedirs(LOCAL_DIR)
  download()
  patch_graph()
