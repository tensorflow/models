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

r"""Generates imitation videos.

Generate single pairwise imitation videos:
blaze build -c opt --config=cuda --copt=-mavx \
learning/brain/research/tcn/generate_videos && \
blaze-bin/learning/brain/research/tcn/generate_videos \
--logtostderr \
--config_paths $config_paths \
--checkpointdir $checkpointdir \
--checkpoint_iter $checkpoint_iter \
--query_records_dir $query_records_dir \
--target_records_dir $target_records_dir \
--outdir $outdir \
--mode single \
--num_query_sequences 1 \
--num_target_sequences -1

# Generate imitation videos with multiple sequences in the target set:
query_records_path
blaze build -c opt --config=cuda --copt=-mavx \
learning/brain/research/tcn/generate_videos && \
blaze-bin/learning/brain/research/tcn/generate_videos \
--logtostderr \
--config_paths $config_paths \
--checkpointdir $checkpointdir \
--checkpoint_iter $checkpoint_iter \
--query_records_dir $query_records_dir \
--target_records_dir $target_records_dir \
--outdir $outdir \
--num_multi_targets 1 \
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
import os
import matplotlib
matplotlib.use("pdf")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from estimators.get_estimator import get_estimator
from utils import util
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
    'checkpointdir', '/tmp/tcn', 'Path to model checkpoints.')
tf.app.flags.DEFINE_string(
    'checkpoint_iter', '', 'Checkpoint iter to use.')
tf.app.flags.DEFINE_integer(
    'num_multi_targets', -1,
    'Number of imitation vids in the target set per imitation video.')
tf.app.flags.DEFINE_string(
    'outdir', '/tmp/tcn', 'Path to write embeddings to.')
tf.app.flags.DEFINE_string(
    'mode', 'single', 'single | multi. Single means generate imitation vids'
                      'where query is being imitated by single sequence. Multi'
                      'means generate imitation vids where query is being'
                      'imitated by multiple.')
tf.app.flags.DEFINE_string('query_records_dir', '',
                           'Directory of image tfrecords.')
tf.app.flags.DEFINE_string('target_records_dir', '',
                           'Directory of image tfrecords.')
tf.app.flags.DEFINE_integer('query_view', 1,
                            'Viewpoint of the query video.')
tf.app.flags.DEFINE_integer('target_view', 0,
                            'Viewpoint of the imitation video.')
tf.app.flags.DEFINE_integer('smoothing_window', 5,
                            'Number of frames to smooth over.')
tf.app.flags.DEFINE_integer('num_query_sequences', -1,
                            'Number of query sequences to embed.')
tf.app.flags.DEFINE_integer('num_target_sequences', -1,
                            'Number of target sequences to embed.')
FLAGS = tf.app.flags.FLAGS


def SmoothEmbeddings(embs):
  """Temporally smoothes a sequence of embeddings."""
  new_embs = []
  window = int(FLAGS.smoothing_window)
  for i in range(len(embs)):
    min_i = max(i-window, 0)
    max_i = min(i+window, len(embs))
    new_embs.append(np.mean(embs[min_i:max_i, :], axis=0))
  return np.array(new_embs)


def MakeImitationVideo(
    outdir, vidname, query_im_strs, knn_im_strs, height=640, width=360):
  """Creates a KNN imitation video.

  For each frame in vid0, pair with the frame at index in knn_indices in
  vids1. Write video to disk.

  Args:
    outdir: String, directory to write videos.
    vidname: String, name of video.
    query_im_strs: Numpy array holding query image strings.
    knn_im_strs: Numpy array holding knn image strings.
    height: Int, height of raw images.
    width: Int, width of raw images.
  """
  if not tf.gfile.Exists(outdir):
    tf.gfile.MakeDirs(outdir)
  vid_path = os.path.join(outdir, vidname)
  combined = zip(query_im_strs, knn_im_strs)

  # Create and write the video.
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_aspect('equal')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  im = ax.imshow(
      np.zeros((height, width*2, 3)), cmap='gray', interpolation='nearest')
  im.set_clim([0, 1])
  plt.tight_layout(pad=0, w_pad=0, h_pad=0)
  # pylint: disable=invalid-name
  def update_img(pair):
    """Decode pairs of image strings, update a video."""
    im_i, im_j = pair
    nparr_i = np.fromstring(str(im_i), np.uint8)
    img_np_i = cv2.imdecode(nparr_i, 1)
    img_np_i = img_np_i[..., [2, 1, 0]]
    nparr_j = np.fromstring(str(im_j), np.uint8)
    img_np_j = cv2.imdecode(nparr_j, 1)
    img_np_j = img_np_j[..., [2, 1, 0]]

    # Optionally reshape the images to be same size.
    frame = np.concatenate([img_np_i, img_np_j], axis=1)
    im.set_data(frame)
    return im
  ani = animation.FuncAnimation(fig, update_img, combined, interval=15)
  writer = animation.writers['ffmpeg'](fps=15)
  dpi = 100
  tf.logging.info('Writing video to:\n %s \n' % vid_path)
  ani.save('%s.mp4' % vid_path, writer=writer, dpi=dpi)


def GenerateImitationVideo(
    vid_name, query_ims, query_embs, target_ims, target_embs, height, width):
  """Generates a single cross-sequence imitation video.

  For each frame in some query sequence, find the nearest neighbor from
  some target sequence in embedding space.

  Args:
    vid_name: String, the name of the video.
    query_ims: Numpy array of shape [query sequence length, height, width, 3].
    query_embs: Numpy array of shape [query sequence length, embedding size].
    target_ims: Numpy array of shape [target sequence length, height, width,
      3].
    target_embs: Numpy array of shape [target sequence length, embedding
      size].
    height: Int, height of the raw image.
    width: Int, width of the raw image.
  """
  # For each query frame, find the index of the nearest neighbor in the
  # target video.
  knn_indices = [util.KNNIds(q, target_embs, k=1)[0] for q in query_embs]

  # Create and write out the video.
  assert knn_indices
  knn_ims = np.array([target_ims[k] for k in knn_indices])
  MakeImitationVideo(FLAGS.outdir, vid_name, query_ims, knn_ims, height, width)


def SingleImitationVideos(
    query_records, target_records, config, height, width):
  """Generates pairwise imitation videos.

  This creates all pairs of target imitating query videos, where each frame
  on the left is matched to a nearest neighbor coming a single
  embedded target video.

  Args:
    query_records: List of Strings, paths to tfrecord datasets to use as
      queries.
    target_records: List of Strings, paths to tfrecord datasets to use as
      targets.
    config: A T object describing training config.
    height: Int, height of the raw image.
    width: Int, width of the raw image.
  """
  # Embed query and target data.
  (query_sequences_to_data,
   target_sequences_to_data) = EmbedQueryTargetData(
       query_records, target_records, config)

  qview = FLAGS.query_view
  tview = FLAGS.target_view

  # Loop over query videos.
  for task_i, data_i in query_sequences_to_data.iteritems():
    for task_j, data_j in target_sequences_to_data.iteritems():
      i_ims = data_i['images']
      i_embs = data_i['embeddings']
      query_embs = SmoothEmbeddings(i_embs[qview])
      query_ims = i_ims[qview]

      j_ims = data_j['images']
      j_embs = data_j['embeddings']
      target_embs = SmoothEmbeddings(j_embs[tview])
      target_ims = j_ims[tview]

      tf.logging.info('Generating %s imitating %s video.' % (task_j, task_i))
      vid_name = 'q%sv%s_im%sv%s' % (task_i, qview, task_j, tview)
      vid_name = vid_name.replace('/', '_')
      GenerateImitationVideo(vid_name, query_ims, query_embs,
                             target_ims, target_embs, height, width)


def MultiImitationVideos(
    query_records, target_records, config, height, width):
  """Creates multi-imitation videos.

  This creates videos where every frame on the left is matched to a nearest
  neighbor coming from a set of multiple embedded target videos.

  Args:
    query_records: List of Strings, paths to tfrecord datasets to use as
      queries.
    target_records: List of Strings, paths to tfrecord datasets to use as
      targets.
    config: A T object describing training config.
    height: Int, height of the raw image.
    width: Int, width of the raw image.
  """
  # Embed query and target data.
  (query_sequences_to_data,
   target_sequences_to_data) = EmbedQueryTargetData(
       query_records, target_records, config)

  qview = FLAGS.query_view
  tview = FLAGS.target_view

  # Loop over query videos.
  for task_i, data_i in query_sequences_to_data.iteritems():
    i_ims = data_i['images']
    i_embs = data_i['embeddings']
    query_embs = SmoothEmbeddings(i_embs[qview])
    query_ims = i_ims[qview]

    all_target_embs = []
    all_target_ims = []

    # If num_imitation_vids is -1, add all seq embeddings to the target set.
    if FLAGS.num_multi_targets == -1:
      num_multi_targets = len(target_sequences_to_data)
    else:
      # Else, add some specified number of seq embeddings to the target set.
      num_multi_targets = FLAGS.num_multi_targets
    for j in range(num_multi_targets):
      task_j = target_sequences_to_data.keys()[j]
      data_j = target_sequences_to_data[task_j]
      print('Adding %s to target set' % task_j)
      j_ims = data_j['images']
      j_embs = data_j['embeddings']

      target_embs = SmoothEmbeddings(j_embs[tview])
      target_ims = j_ims[tview]
      all_target_embs.extend(target_embs)
      all_target_ims.extend(target_ims)

    # Generate a "j imitating i" video.
    tf.logging.info('Generating all imitating %s video.' % task_i)
    vid_name = 'q%sv%s_multiv%s' % (task_i, qview, tview)
    vid_name = vid_name.replace('/', '_')
    GenerateImitationVideo(vid_name, query_ims, query_embs,
                           all_target_ims, all_target_embs, height, width)


def SameSequenceVideos(query_records, config, height, width):
  """Generate same sequence, cross-view imitation videos."""
  batch_size = config.data.embed_batch_size

  # Choose an estimator based on training strategy.
  estimator = get_estimator(config, FLAGS.checkpointdir)

  # Choose a checkpoint path to restore.
  checkpointdir = FLAGS.checkpointdir
  checkpoint_path = os.path.join(checkpointdir,
                                 'model.ckpt-%s' % FLAGS.checkpoint_iter)

  # Embed num_sequences query sequences, store embeddings and image strings in
  # query_sequences_to_data.
  sequences_to_data = {}
  for (view_embeddings, view_raw_image_strings, seqname) in estimator.inference(
      query_records, checkpoint_path, batch_size,
      num_sequences=FLAGS.num_query_sequences):
    sequences_to_data[seqname] = {
        'embeddings': view_embeddings,
        'images': view_raw_image_strings,
    }

  # Loop over query videos.
  qview = FLAGS.query_view
  tview = FLAGS.target_view
  for task_i, data_i in sequences_to_data.iteritems():
    ims = data_i['images']
    embs = data_i['embeddings']
    query_embs = SmoothEmbeddings(embs[qview])
    query_ims = ims[qview]

    target_embs = SmoothEmbeddings(embs[tview])
    target_ims = ims[tview]

    tf.logging.info('Generating %s imitating %s video.' % (task_i, task_i))
    vid_name = 'q%sv%s_im%sv%s' % (task_i, qview, task_i, tview)
    vid_name = vid_name.replace('/', '_')
    GenerateImitationVideo(vid_name, query_ims, query_embs,
                           target_ims, target_embs, height, width)


def EmbedQueryTargetData(query_records, target_records, config):
  """Embeds the full set of query_records and target_records.

  Args:
    query_records: List of Strings, paths to tfrecord datasets to use as
      queries.
    target_records: List of Strings, paths to tfrecord datasets to use as
      targets.
    config: A T object describing training config.

  Returns:
    query_sequences_to_data: A dict holding 'embeddings' and 'images'
    target_sequences_to_data: A dict holding 'embeddings' and 'images'
  """
  batch_size = config.data.embed_batch_size

  # Choose an estimator based on training strategy.
  estimator = get_estimator(config, FLAGS.checkpointdir)

  # Choose a checkpoint path to restore.
  checkpointdir = FLAGS.checkpointdir
  checkpoint_path = os.path.join(checkpointdir,
                                 'model.ckpt-%s' % FLAGS.checkpoint_iter)

  # Embed num_sequences query sequences, store embeddings and image strings in
  # query_sequences_to_data.
  num_query_sequences = FLAGS.num_query_sequences
  num_target_sequences = FLAGS.num_target_sequences
  query_sequences_to_data = {}
  for (view_embeddings, view_raw_image_strings, seqname) in estimator.inference(
      query_records, checkpoint_path, batch_size,
      num_sequences=num_query_sequences):
    query_sequences_to_data[seqname] = {
        'embeddings': view_embeddings,
        'images': view_raw_image_strings,
    }

  if (query_records == target_records) and (
      num_query_sequences == num_target_sequences):
    target_sequences_to_data = query_sequences_to_data
  else:
    # Embed num_sequences target sequences, store embeddings and image strings
    # in sequences_to_data.
    target_sequences_to_data = {}
    for (view_embeddings, view_raw_image_strings,
         seqname) in estimator.inference(
             target_records, checkpoint_path, batch_size,
             num_sequences=num_target_sequences):
      target_sequences_to_data[seqname] = {
          'embeddings': view_embeddings,
          'images': view_raw_image_strings,
      }
  return query_sequences_to_data, target_sequences_to_data


def main(_):
  # Parse config dict from yaml config files / command line flags.
  config = util.ParseConfigsToLuaTable(FLAGS.config_paths, FLAGS.model_params)

  # Get tables to embed.
  query_records_dir = FLAGS.query_records_dir
  query_records = util.GetFilesRecursively(query_records_dir)

  target_records_dir = FLAGS.target_records_dir
  target_records = util.GetFilesRecursively(target_records_dir)

  height = config.data.raw_height
  width = config.data.raw_width
  mode = FLAGS.mode
  if mode == 'multi':
    # Generate videos where target set is composed of multiple videos.
    MultiImitationVideos(query_records, target_records, config,
                         height, width)
  elif mode == 'single':
    # Generate videos where target set is a single video.
    SingleImitationVideos(query_records, target_records, config,
                          height, width)
  elif mode == 'same':
    # Generate videos where target set is the same as query, but diff view.
    SameSequenceVideos(query_records, config, height, width)
  else:
    raise ValueError('Unknown mode %s' % mode)

if __name__ == '__main__':
  tf.app.run()
