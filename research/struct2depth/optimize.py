
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

"""Applies online refinement while running inference.

Instructions: Run static inference first before calling this script. Make sure
to point output_dir to the same folder where static inference results were
saved previously.

For example use, please refer to README.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import random
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

import model
import nets
import reader
import util

gfile = tf.gfile
SAVE_EVERY = 1  # Defines the interval that predictions should be saved at.
SAVE_PREVIEWS = True  # If set, while save image previews of depth predictions.
FIXED_SEED = 8964  # Fixed seed for repeatability.

flags.DEFINE_string('output_dir', None, 'Directory to store predictions. '
                    'Assumes that regular inference has been executed before '
                    'and results were stored in this folder.')
flags.DEFINE_string('data_dir', None, 'Folder pointing to preprocessed '
                    'triplets to fine-tune on.')
flags.DEFINE_string('triplet_list_file', None, 'Text file containing paths to '
                    'image files to process. Paths should be relative with '
                    'respect to the list file location. Every line should be '
                    'of the form [input_folder_name] [input_frame_num] '
                    '[output_path], where [output_path] is optional to specify '
                    'a different path to store the prediction.')
flags.DEFINE_string('triplet_list_file_remains', None, 'Optional text file '
                    'containing relative paths to image files which should not '
                    'be fine-tuned, e.g. because of missing adjacent frames. '
                    'For all files listed, the static prediction will be '
                    'copied instead. File can be empty. If not, every line '
                    'should be of the form [input_folder_name] '
                    '[input_frame_num] [output_path], where [output_path] is '
                    'optional to specify a different path to take and store '
                    'the unrefined prediction from/to.')
flags.DEFINE_string('model_ckpt', None, 'Model checkpoint to optimize.')
flags.DEFINE_string('ft_name', '', 'Optional prefix for temporary files.')
flags.DEFINE_string('file_extension', 'png', 'Image data file extension.')
flags.DEFINE_float('learning_rate', 0.0001, 'Adam learning rate.')
flags.DEFINE_float('beta1', 0.9, 'Adam momentum.')
flags.DEFINE_float('reconstr_weight', 0.85, 'Frame reconstruction loss weight.')
flags.DEFINE_float('ssim_weight', 0.15, 'SSIM loss weight.')
flags.DEFINE_float('smooth_weight', 0.01, 'Smoothness loss weight.')
flags.DEFINE_float('icp_weight', 0.0, 'ICP loss weight.')
flags.DEFINE_float('size_constraint_weight', 0.0005, 'Weight of the object '
                   'size constraint loss. Use only with motion handling.')
flags.DEFINE_integer('batch_size', 1, 'The size of a sample batch')
flags.DEFINE_integer('img_height', 128, 'Input frame height.')
flags.DEFINE_integer('img_width', 416, 'Input frame width.')
flags.DEFINE_integer('seq_length', 3, 'Number of frames in sequence.')
flags.DEFINE_enum('architecture', nets.RESNET, nets.ARCHITECTURES,
                  'Defines the architecture to use for the depth prediction '
                  'network. Defaults to ResNet-based encoder and accompanying '
                  'decoder.')
flags.DEFINE_boolean('imagenet_norm', True, 'Whether to normalize the input '
                     'images channel-wise so that they match the distribution '
                     'most ImageNet-models were trained on.')
flags.DEFINE_float('weight_reg', 0.05, 'The amount of weight regularization to '
                   'apply. This has no effect on the ResNet-based encoder '
                   'architecture.')
flags.DEFINE_boolean('exhaustive_mode', False, 'Whether to exhaustively warp '
                     'from any frame to any other instead of just considering '
                     'adjacent frames. Where necessary, multiple egomotion '
                     'estimates will be applied. Does not have an effect if '
                     'compute_minimum_loss is enabled.')
flags.DEFINE_boolean('random_scale_crop', False, 'Whether to apply random '
                     'image scaling and center cropping during training.')
flags.DEFINE_bool('depth_upsampling', True, 'Whether to apply depth '
                  'upsampling of lower-scale representations before warping to '
                  'compute reconstruction loss on full-resolution image.')
flags.DEFINE_bool('depth_normalization', True, 'Whether to apply depth '
                  'normalization, that is, normalizing inverse depth '
                  'prediction maps by their mean to avoid degeneration towards '
                  'small values.')
flags.DEFINE_bool('compute_minimum_loss', True, 'Whether to take the '
                  'element-wise minimum of the reconstruction/SSIM error in '
                  'order to avoid overly penalizing dis-occlusion effects.')
flags.DEFINE_bool('use_skip', True, 'Whether to use skip connections in the '
                  'encoder-decoder architecture.')
flags.DEFINE_bool('joint_encoder', False, 'Whether to share parameters '
                  'between the depth and egomotion networks by using a joint '
                  'encoder architecture. The egomotion network is then '
                  'operating only on the hidden representation provided by the '
                  'joint encoder.')
flags.DEFINE_float('egomotion_threshold', 0.01, 'Minimum egomotion magnitude '
                   'to apply finetuning. If lower, just forwards the ordinary '
                   'prediction.')
flags.DEFINE_integer('num_steps', 20, 'Number of optimization steps to run.')
flags.DEFINE_boolean('handle_motion', True, 'Whether the checkpoint was '
                     'trained with motion handling.')
flags.DEFINE_bool('flip', False, 'Whether images should be flipped as well as '
                  'resulting predictions (for test-time augmentation). This '
                  'currently applies to the depth network only.')

FLAGS = flags.FLAGS
flags.mark_flag_as_required('output_dir')
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('model_ckpt')
flags.mark_flag_as_required('triplet_list_file')


def main(_):
  """Runs fine-tuning and inference.

  There are three categories of images.
  1) Images where we have previous and next frame, and that are not filtered
     out by the heuristic. For them, we will use the fine-tuned predictions.
  2) Images where we have previous and next frame, but that were filtered out
     by our heuristic. For them, we will use the ordinary prediction instead.
  3) Images where we have at least one missing adjacent frame. For them, we will
     use the ordinary prediction as indicated by triplet_list_file_remains (if
     provided). They will also not be part of the generated inference list in
     the first place.

  Raises:
     ValueError: Invalid parameters have been passed.
  """

  if FLAGS.handle_motion and FLAGS.joint_encoder:
    raise ValueError('Using a joint encoder is currently not supported when '
                     'modeling object motion.')
  if FLAGS.handle_motion and FLAGS.seq_length != 3:
    raise ValueError('The current motion model implementation only supports '
                     'using a sequence length of three.')
  if FLAGS.handle_motion and not FLAGS.compute_minimum_loss:
    raise ValueError('Computing the minimum photometric loss is required when '
                     'enabling object motion handling.')
  if FLAGS.size_constraint_weight > 0 and not FLAGS.handle_motion:
    raise ValueError('To enforce object size constraints, enable motion '
                     'handling.')
  if FLAGS.icp_weight > 0.0:
    raise ValueError('ICP is currently not supported.')
  if FLAGS.compute_minimum_loss and FLAGS.seq_length % 2 != 1:
    raise ValueError('Compute minimum loss requires using an odd number of '
                     'images in a sequence.')
  if FLAGS.compute_minimum_loss and FLAGS.exhaustive_mode:
    raise ValueError('Exhaustive mode has no effect when compute_minimum_loss '
                     'is enabled.')
  if FLAGS.img_width % (2 ** 5) != 0 or FLAGS.img_height % (2 ** 5) != 0:
    logging.warn('Image size is not divisible by 2^5. For the architecture '
                 'employed, this could cause artefacts caused by resizing in '
                 'lower dimensions.')

  if FLAGS.output_dir.endswith('/'):
    FLAGS.output_dir = FLAGS.output_dir[:-1]

  # Create file lists to prepare fine-tuning, save it to unique_file.
  unique_file_name = (str(datetime.datetime.now().date()) + '_' +
                      str(datetime.datetime.now().time()).replace(':', '_'))
  unique_file = os.path.join(FLAGS.data_dir, unique_file_name + '.txt')
  with gfile.FastGFile(FLAGS.triplet_list_file, 'r') as f:
    files_to_process = f.readlines()
    files_to_process = [line.rstrip() for line in files_to_process]
    files_to_process = [line for line in files_to_process if len(line)]
  logging.info('Creating unique file list %s with %s entries.', unique_file,
               len(files_to_process))
  with gfile.FastGFile(unique_file, 'w') as f_out:
    fetches_network = FLAGS.num_steps * FLAGS.batch_size
    fetches_saves = FLAGS.batch_size * int(np.floor(FLAGS.num_steps/SAVE_EVERY))
    repetitions = fetches_network + 3 * fetches_saves
    for i in range(len(files_to_process)):
      for _ in range(repetitions):
        f_out.write(files_to_process[i] + '\n')

  # Read remaining files.
  remaining = []
  if gfile.Exists(FLAGS.triplet_list_file_remains):
    with gfile.FastGFile(FLAGS.triplet_list_file_remains, 'r') as f:
      remaining = f.readlines()
      remaining = [line.rstrip() for line in remaining]
      remaining = [line for line in remaining if len(line)]
  logging.info('Running fine-tuning on %s files, %s files are remaining.',
               len(files_to_process), len(remaining))

  # Run fine-tuning process and save predictions in id-folders.
  tf.set_random_seed(FIXED_SEED)
  np.random.seed(FIXED_SEED)
  random.seed(FIXED_SEED)
  flipping_mode = reader.FLIP_ALWAYS if FLAGS.flip else reader.FLIP_NONE
  train_model = model.Model(data_dir=FLAGS.data_dir,
                            file_extension=FLAGS.file_extension,
                            is_training=True,
                            learning_rate=FLAGS.learning_rate,
                            beta1=FLAGS.beta1,
                            reconstr_weight=FLAGS.reconstr_weight,
                            smooth_weight=FLAGS.smooth_weight,
                            ssim_weight=FLAGS.ssim_weight,
                            icp_weight=FLAGS.icp_weight,
                            batch_size=FLAGS.batch_size,
                            img_height=FLAGS.img_height,
                            img_width=FLAGS.img_width,
                            seq_length=FLAGS.seq_length,
                            architecture=FLAGS.architecture,
                            imagenet_norm=FLAGS.imagenet_norm,
                            weight_reg=FLAGS.weight_reg,
                            exhaustive_mode=FLAGS.exhaustive_mode,
                            random_scale_crop=FLAGS.random_scale_crop,
                            flipping_mode=flipping_mode,
                            random_color=False,
                            depth_upsampling=FLAGS.depth_upsampling,
                            depth_normalization=FLAGS.depth_normalization,
                            compute_minimum_loss=FLAGS.compute_minimum_loss,
                            use_skip=FLAGS.use_skip,
                            joint_encoder=FLAGS.joint_encoder,
                            build_sum=False,
                            shuffle=False,
                            input_file=unique_file_name,
                            handle_motion=FLAGS.handle_motion,
                            size_constraint_weight=FLAGS.size_constraint_weight,
                            train_global_scale_var=False)

  failed_heuristic_ids = finetune_inference(train_model, FLAGS.model_ckpt,
                                            FLAGS.output_dir + '_ft')
  logging.info('Fine-tuning completed, %s files were filtered out by '
               'heuristic.', len(failed_heuristic_ids))
  for failed_id in failed_heuristic_ids:
    failed_entry = files_to_process[failed_id]
    remaining.append(failed_entry)
  logging.info('In total, %s images were fine-tuned, while %s were not.',
               len(files_to_process)-len(failed_heuristic_ids), len(remaining))

  # Copy all results to have the same structural output as running ordinary
  # inference.
  for i in range(len(files_to_process)):
    if files_to_process[i] not in remaining:  # Use fine-tuned result.
      elements = files_to_process[i].split(' ')
      source_file = os.path.join(FLAGS.output_dir + '_ft', FLAGS.ft_name +
                                 'id_' + str(i),
                                 str(FLAGS.num_steps).zfill(10) +
                                 ('_flip' if FLAGS.flip else ''))
      if len(elements) == 2:  # No differing mapping defined.
        target_dir = os.path.join(FLAGS.output_dir + '_ft', elements[0])
        target_file = os.path.join(
            target_dir, elements[1] + ('_flip' if FLAGS.flip else ''))
      else:  # Other mapping for file defined, copy to this location instead.
        target_dir = os.path.join(
            FLAGS.output_dir + '_ft', os.path.dirname(elements[2]))
        target_file = os.path.join(
            target_dir,
            os.path.basename(elements[2]) + ('_flip' if FLAGS.flip else ''))
      if not gfile.Exists(target_dir):
        gfile.MakeDirs(target_dir)
      logging.info('Copy refined result %s to %s.', source_file, target_file)
      gfile.Copy(source_file + '.npy', target_file + '.npy', overwrite=True)
      gfile.Copy(source_file + '.txt', target_file + '.txt', overwrite=True)
      gfile.Copy(source_file + '.%s' % FLAGS.file_extension,
                 target_file + '.%s' % FLAGS.file_extension, overwrite=True)
  for j in range(len(remaining)):
    elements = remaining[j].split(' ')
    if len(elements) == 2:  # No differing mapping defined.
      target_dir = os.path.join(FLAGS.output_dir + '_ft', elements[0])
      target_file = os.path.join(
          target_dir, elements[1] + ('_flip' if FLAGS.flip else ''))
    else:  # Other mapping for file defined, copy to this location instead.
      target_dir = os.path.join(
          FLAGS.output_dir + '_ft', os.path.dirname(elements[2]))
      target_file = os.path.join(
          target_dir,
          os.path.basename(elements[2]) + ('_flip' if FLAGS.flip else ''))
    if not gfile.Exists(target_dir):
      gfile.MakeDirs(target_dir)
    source_file = target_file.replace('_ft', '')
    logging.info('Copy unrefined result %s to %s.', source_file, target_file)
    gfile.Copy(source_file + '.npy', target_file + '.npy', overwrite=True)
    gfile.Copy(source_file + '.%s' % FLAGS.file_extension,
               target_file + '.%s' % FLAGS.file_extension, overwrite=True)
  logging.info('Done, predictions saved in %s.', FLAGS.output_dir + '_ft')


def finetune_inference(train_model, model_ckpt, output_dir):
  """Train model."""
  vars_to_restore = None
  if model_ckpt is not None:
    vars_to_restore = util.get_vars_to_save_and_restore(model_ckpt)
    ckpt_path = model_ckpt
  pretrain_restorer = tf.train.Saver(vars_to_restore)
  sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None,
                           summary_op=None)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  img_nr = 0
  failed_heuristic = []
  with sv.managed_session(config=config) as sess:
    # TODO(casser): Caching the weights would be better to avoid I/O bottleneck.
    while True:  # Loop terminates when all examples have been processed.
      if model_ckpt is not None:
        logging.info('Restored weights from %s', ckpt_path)
        pretrain_restorer.restore(sess, ckpt_path)
      logging.info('Running fine-tuning, image %s...', img_nr)
      img_pred_folder = os.path.join(
          output_dir, FLAGS.ft_name + 'id_' + str(img_nr))
      if not gfile.Exists(img_pred_folder):
        gfile.MakeDirs(img_pred_folder)
      step = 1

      # Run fine-tuning.
      while step <= FLAGS.num_steps:
        logging.info('Running step %s of %s.', step, FLAGS.num_steps)
        fetches = {
            'train': train_model.train_op,
            'global_step': train_model.global_step,
            'incr_global_step': train_model.incr_global_step
        }
        _ = sess.run(fetches)
        if step % SAVE_EVERY == 0:
          # Get latest prediction for middle frame, highest scale.
          pred = train_model.depth[1][0].eval(session=sess)
          if FLAGS.flip:
            pred = np.flip(pred, axis=2)
          input_img = train_model.image_stack.eval(session=sess)
          input_img_prev = input_img[0, :, :, 0:3]
          input_img_center = input_img[0, :, :, 3:6]
          input_img_next = input_img[0, :, :, 6:]
          img_pred_file = os.path.join(
              img_pred_folder,
              str(step).zfill(10) + ('_flip' if FLAGS.flip else '') + '.npy')
          motion = np.squeeze(train_model.egomotion.eval(session=sess))
          # motion of shape (seq_length - 1, 6).
          motion = np.mean(motion, axis=0)  # Average egomotion across frames.

          if SAVE_PREVIEWS or step == FLAGS.num_steps:
            # Also save preview of depth map.
            color_map = util.normalize_depth_for_display(
                np.squeeze(pred[0, :, :]))
            visualization = np.concatenate(
                (input_img_prev, input_img_center, input_img_next, color_map))
            motion_s = [str(m) for m in motion]
            s_rep = ','.join(motion_s)
            with gfile.Open(img_pred_file.replace('.npy', '.txt'), 'w') as f:
              f.write(s_rep)
            util.save_image(
                img_pred_file.replace('.npy', '.%s' % FLAGS.file_extension),
                visualization, FLAGS.file_extension)

          with gfile.Open(img_pred_file, 'wb') as f:
            np.save(f, pred)

        # Apply heuristic to not finetune if egomotion magnitude is too low.
        ego_magnitude = np.linalg.norm(motion[:3], ord=2)
        heuristic = ego_magnitude >= FLAGS.egomotion_threshold
        if not heuristic and step == FLAGS.num_steps:
          failed_heuristic.append(img_nr)

        step += 1
      img_nr += 1
  return failed_heuristic


if __name__ == '__main__':
  app.run(main)
