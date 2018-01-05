# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Visualizes embeddings in tensorboard.

Usage:
root=experimental/users/sermanet/imitation/mirror && \
blaze build -c opt --copt=-mavx --config=cuda $root:visualize_embeddings && \
blaze-bin/$root/visualize_embeddings \
--checkpointdir $checkpointdir \
--checkpoint_iter $checkpoint_iter \
--embedding_records $embedding_records \
--outdir $outdir \
--num_embed 1000 \
--sprite_dim 64 \
--config_paths $configs \
--logtostderr

blaze build third_party/tensorboard && \
blaze-bin/third_party/tensorboard/tensorboard --logdir=$outdir
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import cv2
import numpy as np
from scipy.misc import imresize
from scipy.misc import imsave
from estimators.get_estimator import get_estimator
from utils import util
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_string(
    'config_paths', '',
    """
    Path to a YAML configuration files defining FLAG values. Multiple files
    can be separated by the `#` symbol. Files are merged recursively. Setting
    a key in these files is equivalent to setting the FLAG value with
    the same name.
    """)
tf.flags.DEFINE_string(
    'model_params', '{}', 'YAML configuration string for the model parameters.')
tf.app.flags.DEFINE_string(
    'checkpoint_iter', '', 'Evaluate this specific checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpointdir', '/tmp/tcn', 'Path to model checkpoints.')
tf.app.flags.DEFINE_string(
    'outdir', '/tmp/tcn', 'Path to write tensorboard info to.')
tf.app.flags.DEFINE_integer(
    'num_embed', 4000, 'Number of embeddings.')
tf.app.flags.DEFINE_integer(
    'num_sequences', -1, 'Number of sequences, -1 for all.')
tf.app.flags.DEFINE_integer(
    'sprite_dim', 64, 'Height, width of the square sprite image.')
tf.app.flags.DEFINE_string(
    'embedding_records', None, 'path to embedding records')
FLAGS = tf.app.flags.FLAGS


def images_to_sprite(data):
  """Creates the sprite image along with any necessary padding.

  Taken from: https://github.com/tensorflow/tensorflow/issues/6322

  Args:
    data: NxHxW[x3] tensor containing the images.

  Returns:
    data: Properly shaped HxWx3 image with any necessary padding.
  """
  if len(data.shape) == 3:
    data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
  data = data.astype(np.float32)
  min_v = np.min(data.reshape((data.shape[0], -1)), axis=1)
  data = (data.transpose(1, 2, 3, 0) - min_v).transpose(3, 0, 1, 2)
  max_v = np.max(data.reshape((data.shape[0], -1)), axis=1)
  data = (data.transpose(1, 2, 3, 0) / max_v).transpose(3, 0, 1, 2)
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = ((0, n ** 2 - data.shape[0]), (0, 0),
             (0, 0)) + ((0, 0),) * (data.ndim - 3)
  data = np.pad(data, padding, mode='constant',
                constant_values=0)
  # Tile the individual thumbnails into an image.
  data = data.reshape((n, n) + data.shape[1:]).transpose(
      (0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
  data = (data * 255).astype(np.uint8)
  return data


def main(_):
  """Runs main labeled eval loop."""
  # Parse config dict from yaml config files / command line flags.
  config = util.ParseConfigsToLuaTable(FLAGS.config_paths, FLAGS.model_params)

  # Choose an estimator based on training strategy.
  checkpointdir = FLAGS.checkpointdir
  checkpoint_path = os.path.join(
      '%s/model.ckpt-%s' % (checkpointdir, FLAGS.checkpoint_iter))
  estimator = get_estimator(config, checkpointdir)

  # Get records to embed.
  validation_dir = FLAGS.embedding_records
  validation_records = util.GetFilesRecursively(validation_dir)

  sequences_to_data = {}
  for (view_embeddings, view_raw_image_strings, seqname) in estimator.inference(
      validation_records, checkpoint_path, config.data.embed_batch_size,
      num_sequences=FLAGS.num_sequences):
    sequences_to_data[seqname] = {
        'embeddings': view_embeddings,
        'images': view_raw_image_strings,
    }

  all_embeddings = np.zeros((0, config.embedding_size))
  all_ims = []
  all_seqnames = []

  num_embeddings = FLAGS.num_embed
  # Concatenate all views from all sequences into a big flat list.
  for seqname, data in sequences_to_data.iteritems():
    embs = data['embeddings']
    ims = data['images']
    for v in range(config.data.num_views):
      for (emb, im) in zip(embs[v], ims[v]):
        all_embeddings = np.append(all_embeddings, [emb], axis=0)
        all_ims.append(im)
        all_seqnames.append(seqname)

  # Choose N indices uniformly from all images.
  random_indices = range(all_embeddings.shape[0])
  random.shuffle(random_indices)
  viz_indices = random_indices[:num_embeddings]

  # Extract embs.
  viz_embs = np.array(all_embeddings[viz_indices])

  # Extract and decode ims.
  viz_ims = list(np.array(all_ims)[viz_indices])
  decoded_ims = []

  sprite_dim = FLAGS.sprite_dim
  for i, im in enumerate(viz_ims):
    if i % 100 == 0:
      print('Decoding image %d/%d.' % (i, num_embeddings))
    nparr_i = np.fromstring(str(im), np.uint8)
    img_np = cv2.imdecode(nparr_i, 1)
    img_np = img_np[..., [2, 1, 0]]

    img_np = imresize(img_np, [sprite_dim, sprite_dim, 3])
    decoded_ims.append(img_np)
  decoded_ims = np.array(decoded_ims)

  # Extract sequence names.
  outdir = FLAGS.outdir

  # The embedding variable, which needs to be stored
  # Note this must a Variable not a Tensor!
  embedding_var = tf.Variable(viz_embs, name='viz_embs')

  with tf.Session() as sess:
    sess.run(embedding_var.initializer)
    summary_writer = tf.summary.FileWriter(outdir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Comment out if you don't want sprites
    embedding.sprite.image_path = os.path.join(outdir, 'sprite.png')
    embedding.sprite.single_image_dim.extend(
        [decoded_ims.shape[1], decoded_ims.shape[1]])

    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver([embedding_var])
    saver.save(sess, os.path.join(outdir, 'model2.ckpt'), 1)

  sprite = images_to_sprite(decoded_ims)
  imsave(os.path.join(outdir, 'sprite.png'), sprite)

if __name__ == '__main__':
  tf.app.run(main)
