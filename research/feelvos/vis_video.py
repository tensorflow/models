# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Segmentation results evaluation and visualization for videos using attention.
"""

import math
import os
import time
import numpy as np

import tensorflow as tf

from feelvos import common
from feelvos import model
from feelvos.datasets import video_dataset
from feelvos.utils import embedding_utils
from feelvos.utils import eval_utils
from feelvos.utils import video_input_generator


slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('vis_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

flags.DEFINE_integer('output_stride', 8,
                     'The ratio of input to output spatial resolution.')

flags.DEFINE_string('dataset', 'davis_2016',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('vis_split', 'val',
                    'Which split of the dataset used for visualizing results')

flags.DEFINE_string(
    'dataset_dir',
    '/cns/is-d/home/lcchen/data/pascal_voc_seg/example_sstables',
    'Where the dataset resides.')

flags.DEFINE_integer('num_vis_examples', -1,
                     'Number of examples for visualization. If -1, use all '
                     'samples in the vis data.')

flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_bool('save_segmentations', False, 'Whether to save the '
                                               'segmentation masks as '
                                               'png images. Might be slow '
                                               'on cns.')

flags.DEFINE_bool('save_embeddings', False, 'Whether to save the embeddings as'
                                            'pickle. Might be slow on cns.')

flags.DEFINE_bool('eval_once_and_quit', False,
                  'Whether to just run the eval a single time and quit '
                  'afterwards. Otherwise, the eval is run in a loop with '
                  'new checkpoints.')

flags.DEFINE_boolean('first_frame_finetuning', False,
                     'Whether to only sample the first frame for fine-tuning.')

# the folder where segmentations are saved.
_SEGMENTATION_SAVE_FOLDER = 'segmentation'
_EMBEDDINGS_SAVE_FOLDER = 'embeddings'


def _process_seq_data(segmentation_dir, embeddings_dir, seq_name,
                      predicted_labels, gt_labels, embeddings):
  """Calculates the sequence IoU and optionally save the segmentation masks.

  Args:
    segmentation_dir: Directory in which the segmentation results are stored.
    embeddings_dir: Directory in which the embeddings are stored.
    seq_name: String, the name of the sequence.
    predicted_labels: Int64 np.array of shape [n_frames, height, width].
    gt_labels: Ground truth labels, Int64 np.array of shape
      [n_frames, height, width].
    embeddings: Float32 np.array of embeddings of shape
      [n_frames, decoder_height, decoder_width, embedding_dim], or None.

  Returns:
    The IoU for the sequence (float).
  """
  sequence_dir = os.path.join(segmentation_dir, seq_name)
  tf.gfile.MakeDirs(sequence_dir)
  embeddings_seq_dir = os.path.join(embeddings_dir, seq_name)
  tf.gfile.MakeDirs(embeddings_seq_dir)
  label_set = np.unique(gt_labels[0])
  ious = []
  assert len(predicted_labels) == len(gt_labels)
  if embeddings is not None:
    assert len(predicted_labels) == len(embeddings)
  for t, (predicted_label, gt_label) in enumerate(
      zip(predicted_labels, gt_labels)):
    if FLAGS.save_segmentations:
      seg_filename = os.path.join(segmentation_dir, seq_name, '%05d.png' % t)
      eval_utils.save_segmentation_with_colormap(seg_filename, predicted_label)
    if FLAGS.save_embeddings:
      embedding_filename = os.path.join(embeddings_dir, seq_name,
                                        '%05d.npy' % t)
      assert embeddings is not None
      eval_utils.save_embeddings(embedding_filename, embeddings[t])
    object_ious_t = eval_utils.calculate_multi_object_ious(
        predicted_label, gt_label, label_set)
    ious.append(object_ious_t)
  # First and last frame are excluded in DAVIS eval.
  seq_ious = np.mean(ious[1:-1], axis=0)
  tf.logging.info('seq ious: %s %s', seq_name, seq_ious)
  return seq_ious


def create_predictions(samples, reference_labels, first_frame_img,
                       model_options):
  """Predicts segmentation labels for each frame of the video.

  Slower version than create_predictions_fast, but does support more options.

  Args:
    samples: Dictionary of input samples.
    reference_labels: Int tensor of shape [1, height, width, 1].
    first_frame_img: Float32 tensor of shape [height, width, 3].
    model_options: An InternalModelOptions instance to configure models.

  Returns:
    predicted_labels: Int tensor of shape [time, height, width] of
      predicted labels for each frame.
    all_embeddings: Float32 tensor of shape
      [time, height, width, embedding_dim], or None.
  """

  def predict(args, imgs):
    """Predicts segmentation labels and softmax probabilities for each image.

    Args:
      args: A tuple of (predictions, softmax_probabilities), where predictions
        is an int tensor of shape [1, h, w] and softmax_probabilities is a
        float32 tensor of shape [1, h_decoder, w_decoder, n_objects].
      imgs: Either a one-tuple of the image to predict labels for of shape
        [h, w, 3], or pair of previous frame and current frame image.

    Returns:
      predictions: The predicted labels as int tensor of shape [1, h, w].
      softmax_probabilities: The softmax probabilities of shape
        [1, h_decoder, w_decoder, n_objects].
    """
    if FLAGS.save_embeddings:
      last_frame_predictions, last_softmax_probabilities, _ = args
    else:
      last_frame_predictions, last_softmax_probabilities = args

    if FLAGS.also_attend_to_previous_frame or FLAGS.use_softmax_feedback:
      ref_labels_to_use = tf.concat(
          [reference_labels, last_frame_predictions[..., tf.newaxis]],
          axis=0)
    else:
      ref_labels_to_use = reference_labels

    predictions, softmax_probabilities = model.predict_labels(
        tf.stack((first_frame_img,) + imgs),
        model_options=model_options,
        image_pyramid=FLAGS.image_pyramid,
        embedding_dimension=FLAGS.embedding_dimension,
        reference_labels=ref_labels_to_use,
        k_nearest_neighbors=FLAGS.k_nearest_neighbors,
        use_softmax_feedback=FLAGS.use_softmax_feedback,
        initial_softmax_feedback=last_softmax_probabilities,
        embedding_seg_feature_dimension=
        FLAGS.embedding_seg_feature_dimension,
        embedding_seg_n_layers=FLAGS.embedding_seg_n_layers,
        embedding_seg_kernel_size=FLAGS.embedding_seg_kernel_size,
        embedding_seg_atrous_rates=FLAGS.embedding_seg_atrous_rates,
        also_return_softmax_probabilities=True,
        num_frames_per_video=
        (3 if FLAGS.also_attend_to_previous_frame or
         FLAGS.use_softmax_feedback else 2),
        normalize_nearest_neighbor_distances=
        FLAGS.normalize_nearest_neighbor_distances,
        also_attend_to_previous_frame=FLAGS.also_attend_to_previous_frame,
        use_local_previous_frame_attention=
        FLAGS.use_local_previous_frame_attention,
        previous_frame_attention_window_size=
        FLAGS.previous_frame_attention_window_size,
        use_first_frame_matching=FLAGS.use_first_frame_matching
    )
    predictions = tf.cast(predictions[common.OUTPUT_TYPE], tf.int32)

    if FLAGS.save_embeddings:
      names = [n.name for n in tf.get_default_graph().as_graph_def().node]
      embedding_names = [x for x in names if 'embeddings' in x]
      # This will crash when multi-scale inference is used.
      assert len(embedding_names) == 1, len(embedding_names)
      embedding_name = embedding_names[0] + ':0'
      embeddings = tf.get_default_graph().get_tensor_by_name(embedding_name)
      return predictions, softmax_probabilities, embeddings
    else:
      return predictions, softmax_probabilities

  init_labels = tf.squeeze(reference_labels, axis=-1)
  init_softmax = embedding_utils.create_initial_softmax_from_labels(
      reference_labels, reference_labels, common.parse_decoder_output_stride(),
      reduce_labels=False)
  if FLAGS.save_embeddings:
    decoder_height = tf.shape(init_softmax)[1]
    decoder_width = tf.shape(init_softmax)[2]
    n_frames = (3 if FLAGS.also_attend_to_previous_frame
                or FLAGS.use_softmax_feedback else 2)
    embeddings_init = tf.zeros((n_frames, decoder_height, decoder_width,
                                FLAGS.embedding_dimension))
    init = (init_labels, init_softmax, embeddings_init)
  else:
    init = (init_labels, init_softmax)
  # Do not eval the first frame again but concat the first frame ground
  # truth instead.
  if FLAGS.also_attend_to_previous_frame or FLAGS.use_softmax_feedback:
    elems = (samples[common.IMAGE][:-1], samples[common.IMAGE][1:])
  else:
    elems = (samples[common.IMAGE][1:],)
  res = tf.scan(predict, elems,
                initializer=init,
                parallel_iterations=1,
                swap_memory=True)
  if FLAGS.save_embeddings:
    predicted_labels, _, all_embeddings = res
    first_frame_embeddings = all_embeddings[0, 0, tf.newaxis]
    other_frame_embeddings = all_embeddings[:, -1]
    all_embeddings = tf.concat(
        [first_frame_embeddings, other_frame_embeddings], axis=0)
  else:
    predicted_labels, _ = res
    all_embeddings = None
  predicted_labels = tf.concat([reference_labels[..., 0],
                                tf.squeeze(predicted_labels, axis=1)],
                               axis=0)
  return predicted_labels, all_embeddings


def create_predictions_fast(samples, reference_labels, first_frame_img,
                            model_options):
  """Predicts segmentation labels for each frame of the video.

  Faster version than create_predictions, but does not support all options.

  Args:
    samples: Dictionary of input samples.
    reference_labels: Int tensor of shape [1, height, width, 1].
    first_frame_img: Float32 tensor of shape [height, width, 3].
    model_options: An InternalModelOptions instance to configure models.

  Returns:
    predicted_labels: Int tensor of shape [time, height, width] of
      predicted labels for each frame.
    all_embeddings: Float32 tensor of shape
      [time, height, width, embedding_dim], or None.

  Raises:
    ValueError: If FLAGS.save_embeddings is True, FLAGS.use_softmax_feedback is
      False, or FLAGS.also_attend_to_previous_frame is False.
  """
  if FLAGS.save_embeddings:
    raise ValueError('save_embeddings does not work with '
                     'create_predictions_fast. Use the slower '
                     'create_predictions instead.')
  if not FLAGS.use_softmax_feedback:
    raise ValueError('use_softmax_feedback must be True for '
                     'create_predictions_fast. Use the slower '
                     'create_predictions instead.')
  if not FLAGS.also_attend_to_previous_frame:
    raise ValueError('also_attend_to_previous_frame must be True for '
                     'create_predictions_fast. Use the slower '
                     'create_predictions instead.')
  # Extract embeddings for first frame and prepare initial predictions.
  first_frame_embeddings = embedding_utils.get_embeddings(
      first_frame_img[tf.newaxis], model_options, FLAGS.embedding_dimension)
  init_labels = tf.squeeze(reference_labels, axis=-1)
  init_softmax = embedding_utils.create_initial_softmax_from_labels(
      reference_labels, reference_labels, common.parse_decoder_output_stride(),
      reduce_labels=False)
  init = (init_labels, init_softmax, first_frame_embeddings)

  def predict(args, img):
    """Predicts segmentation labels and softmax probabilities for each image.

    Args:
      args: tuple of
        (predictions, softmax_probabilities, last_frame_embeddings), where
        predictions is an int tensor of shape [1, h, w],
        softmax_probabilities is a float32 tensor of shape
        [1, h_decoder, w_decoder, n_objects],
        and last_frame_embeddings is a float32 tensor of shape
        [h_decoder, w_decoder, embedding_dimension].
      img: Image to predict labels for of shape [h, w, 3].

    Returns:
      predictions: The predicted labels as int tensor of shape [1, h, w].
      softmax_probabilities: The softmax probabilities of shape
        [1, h_decoder, w_decoder, n_objects].
    """
    (last_frame_predictions, last_softmax_probabilities,
     prev_frame_embeddings) = args
    ref_labels_to_use = tf.concat(
        [reference_labels, last_frame_predictions[..., tf.newaxis]],
        axis=0)

    predictions, softmax_probabilities, embeddings = model.predict_labels(
        img[tf.newaxis],
        model_options=model_options,
        image_pyramid=FLAGS.image_pyramid,
        embedding_dimension=FLAGS.embedding_dimension,
        reference_labels=ref_labels_to_use,
        k_nearest_neighbors=FLAGS.k_nearest_neighbors,
        use_softmax_feedback=FLAGS.use_softmax_feedback,
        initial_softmax_feedback=last_softmax_probabilities,
        embedding_seg_feature_dimension=
        FLAGS.embedding_seg_feature_dimension,
        embedding_seg_n_layers=FLAGS.embedding_seg_n_layers,
        embedding_seg_kernel_size=FLAGS.embedding_seg_kernel_size,
        embedding_seg_atrous_rates=FLAGS.embedding_seg_atrous_rates,
        also_return_softmax_probabilities=True,
        num_frames_per_video=1,
        normalize_nearest_neighbor_distances=
        FLAGS.normalize_nearest_neighbor_distances,
        also_attend_to_previous_frame=FLAGS.also_attend_to_previous_frame,
        use_local_previous_frame_attention=
        FLAGS.use_local_previous_frame_attention,
        previous_frame_attention_window_size=
        FLAGS.previous_frame_attention_window_size,
        use_first_frame_matching=FLAGS.use_first_frame_matching,
        also_return_embeddings=True,
        ref_embeddings=(first_frame_embeddings, prev_frame_embeddings)
    )
    predictions = tf.cast(predictions[common.OUTPUT_TYPE], tf.int32)
    return predictions, softmax_probabilities, embeddings

  # Do not eval the first frame again but concat the first frame ground
  # truth instead.
  # If you have a lot of GPU memory, you can try to set swap_memory=False,
  # and/or parallel_iterations=2.
  elems = samples[common.IMAGE][1:]
  res = tf.scan(predict, elems,
                initializer=init,
                parallel_iterations=1,
                swap_memory=True)
  predicted_labels, _, _ = res
  predicted_labels = tf.concat([reference_labels[..., 0],
                                tf.squeeze(predicted_labels, axis=1)],
                               axis=0)
  return predicted_labels


def main(unused_argv):
  if FLAGS.vis_batch_size != 1:
    raise ValueError('Only batch size 1 is supported for now')

  data_type = 'tf_sequence_example'
  # Get dataset-dependent information.
  dataset = video_dataset.get_dataset(
      FLAGS.dataset,
      FLAGS.vis_split,
      dataset_dir=FLAGS.dataset_dir,
      data_type=data_type)

  # Prepare for visualization.
  tf.gfile.MakeDirs(FLAGS.vis_logdir)
  segmentation_dir = os.path.join(FLAGS.vis_logdir, _SEGMENTATION_SAVE_FOLDER)
  tf.gfile.MakeDirs(segmentation_dir)
  embeddings_dir = os.path.join(FLAGS.vis_logdir, _EMBEDDINGS_SAVE_FOLDER)
  tf.gfile.MakeDirs(embeddings_dir)
  num_vis_examples = (dataset.num_videos if (FLAGS.num_vis_examples < 0)
                      else FLAGS.num_vis_examples)
  if FLAGS.first_frame_finetuning:
    num_vis_examples = 1

  tf.logging.info('Visualizing on %s set', FLAGS.vis_split)
  g = tf.Graph()
  with g.as_default():
    # Without setting device to CPU we run out of memory.
    with tf.device('cpu:0'):
      samples = video_input_generator.get(
          dataset,
          None,
          None,
          FLAGS.vis_batch_size,
          min_resize_value=FLAGS.min_resize_value,
          max_resize_value=FLAGS.max_resize_value,
          resize_factor=FLAGS.resize_factor,
          dataset_split=FLAGS.vis_split,
          is_training=False,
          model_variant=FLAGS.model_variant,
          preprocess_image_and_label=False,
          remap_labels_to_reference_frame=False)
      samples[common.IMAGE] = tf.cast(samples[common.IMAGE], tf.float32)
      samples[common.LABEL] = tf.cast(samples[common.LABEL], tf.int32)
      first_frame_img = samples[common.IMAGE][0]
      reference_labels = samples[common.LABEL][0, tf.newaxis]
      gt_labels = tf.squeeze(samples[common.LABEL], axis=-1)
      seq_name = samples[common.VIDEO_ID][0]

    model_options = common.VideoModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=None,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    all_embeddings = None
    predicted_labels = create_predictions_fast(
        samples, reference_labels, first_frame_img, model_options)
    # If you need more options like saving embeddings, replace the call to
    # create_predictions_fast with create_predictions.

    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(slim.get_variables_to_restore())
    sv = tf.train.Supervisor(graph=g,
                             logdir=FLAGS.vis_logdir,
                             init_op=tf.global_variables_initializer(),
                             summary_op=None,
                             summary_writer=None,
                             global_step=None,
                             saver=saver)
    num_batches = int(
        math.ceil(num_vis_examples / float(FLAGS.vis_batch_size)))
    last_checkpoint = None

    # Infinite loop to visualize the results when new checkpoint is created.
    while True:
      last_checkpoint = slim.evaluation.wait_for_new_checkpoint(
          FLAGS.checkpoint_dir, last_checkpoint)
      start = time.time()
      tf.logging.info(
          'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                       time.gmtime()))
      tf.logging.info('Visualizing with model %s', last_checkpoint)

      all_ious = []
      with sv.managed_session(FLAGS.master,
                              start_standard_services=False) as sess:
        sv.start_queue_runners(sess)
        sv.saver.restore(sess, last_checkpoint)

        for batch in range(num_batches):
          ops = [predicted_labels, gt_labels, seq_name]
          if FLAGS.save_embeddings:
            ops.append(all_embeddings)
          tf.logging.info('Visualizing batch %d / %d', batch + 1, num_batches)
          res = sess.run(ops)
          tf.logging.info('Forwarding done')
          pred_labels_val, gt_labels_val, seq_name_val = res[:3]
          if FLAGS.save_embeddings:
            all_embeddings_val = res[3]
          else:
            all_embeddings_val = None
          seq_ious = _process_seq_data(segmentation_dir, embeddings_dir,
                                       seq_name_val, pred_labels_val,
                                       gt_labels_val, all_embeddings_val)
          all_ious.append(seq_ious)
      all_ious = np.concatenate(all_ious, axis=0)
      tf.logging.info('n_seqs %s, mIoU %f', all_ious.shape, all_ious.mean())
      tf.logging.info(
          'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                       time.gmtime()))
      result_dir = FLAGS.vis_logdir + '/results/'
      tf.gfile.MakeDirs(result_dir)
      with tf.gfile.GFile(result_dir + seq_name_val + '.txt', 'w') as f:
        f.write(str(all_ious))
      if FLAGS.first_frame_finetuning or FLAGS.eval_once_and_quit:
        break
      time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('vis_logdir')
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
