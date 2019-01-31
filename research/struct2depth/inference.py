
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

"""Runs struct2depth at inference. Produces depth estimates, ego-motion and object motion."""

# Example usage:
#
# python inference.py \
#    --input_dir ~/struct2depth/kitti-raw-uncompressed/ \
#    --output_dir ~/struct2depth/output \
#    --model_ckpt ~/struct2depth/model/model-199160
#    --file_extension png \
#    --depth \
#    --egomotion true \



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging
#import matplotlib.pyplot as plt
import model
import numpy as np
import fnmatch
import tensorflow as tf
import nets
import util

gfile = tf.gfile

# CMAP = 'plasma'

INFERENCE_MODE_SINGLE = 'single'  # Take plain single-frame input.
INFERENCE_MODE_TRIPLETS = 'triplets'  # Take image triplets as input.
# For KITTI, we just resize input images and do not perform cropping. For
# Cityscapes, the car hood and more image content has been cropped in order
# to fit aspect ratio, and remove static content from the images. This has to be
# kept at inference time.
INFERENCE_CROP_NONE = 'none'
INFERENCE_CROP_CITYSCAPES = 'cityscapes'


flags.DEFINE_string('output_dir', None, 'Directory to store predictions.')
flags.DEFINE_string('file_extension', 'png', 'Image data file extension of '
                    'files provided with input_dir. Also determines the output '
                    'file format of depth prediction images.')
flags.DEFINE_bool('depth', True, 'Determines if the depth prediction network '
                  'should be executed and its predictions be saved.')
flags.DEFINE_bool('egomotion', False, 'Determines if the egomotion prediction '
                  'network should be executed and its predictions be saved. If '
                  'inference is run in single inference mode, it is assumed '
                  'that files in the same directory belong in the same '
                  'sequence, and sorting them alphabetically establishes the '
                  'right temporal order.')
flags.DEFINE_string('model_ckpt', None, 'Model checkpoint to evaluate.')
flags.DEFINE_string('input_dir', None, 'Directory containing image files to '
                    'evaluate. This crawls recursively for images in the '
                    'directory, mirroring relative subdirectory structures '
                    'into the output directory.')
flags.DEFINE_string('input_list_file', None, 'Text file containing paths to '
                    'image files to process. Paths should be relative with '
                    'respect to the list file location. Relative path '
                    'structures will be mirrored in the output directory.')
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
flags.DEFINE_bool('use_skip', True, 'Whether to use skip connections in the '
                  'encoder-decoder architecture.')
flags.DEFINE_bool('joint_encoder', False, 'Whether to share parameters '
                  'between the depth and egomotion networks by using a joint '
                  'encoder architecture. The egomotion network is then '
                  'operating only on the hidden representation provided by the '
                  'joint encoder.')
flags.DEFINE_bool('shuffle', False, 'Whether to shuffle the order in which '
                  'images are processed.')
flags.DEFINE_bool('flip', False, 'Whether images should be flipped as well as '
                  'resulting predictions (for test-time augmentation). This '
                  'currently applies to the depth network only.')
flags.DEFINE_enum('inference_mode', INFERENCE_MODE_SINGLE,
                  [INFERENCE_MODE_SINGLE,
                   INFERENCE_MODE_TRIPLETS],
                  'Whether to use triplet mode for inference, which accepts '
                  'triplets instead of single frames.')
flags.DEFINE_enum('inference_crop', INFERENCE_CROP_NONE,
                  [INFERENCE_CROP_NONE,
                   INFERENCE_CROP_CITYSCAPES],
                  'Whether to apply a Cityscapes-specific crop on the input '
                  'images first before running inference.')
flags.DEFINE_bool('use_masks', False, 'Whether to mask out potentially '
                  'moving objects when feeding image input to the egomotion '
                  'network. This might improve odometry results when using '
                  'a motion model. For this, pre-computed segmentation '
                  'masks have to be available for every image, with the '
                  'background being zero.')

FLAGS = flags.FLAGS

flags.mark_flag_as_required('output_dir')
flags.mark_flag_as_required('model_ckpt')


def _run_inference(output_dir=None,
                   file_extension='png',
                   depth=True,
                   egomotion=False,
                   model_ckpt=None,
                   input_dir=None,
                   input_list_file=None,
                   batch_size=1,
                   img_height=128,
                   img_width=416,
                   seq_length=3,
                   architecture=nets.RESNET,
                   imagenet_norm=True,
                   use_skip=True,
                   joint_encoder=True,
                   shuffle=False,
                   flip_for_depth=False,
                   inference_mode=INFERENCE_MODE_SINGLE,
                   inference_crop=INFERENCE_CROP_NONE,
                   use_masks=False):
  """Runs inference. Refer to flags in inference.py for details."""
  inference_model = model.Model(is_training=False,
                                batch_size=batch_size,
                                img_height=img_height,
                                img_width=img_width,
                                seq_length=seq_length,
                                architecture=architecture,
                                imagenet_norm=imagenet_norm,
                                use_skip=use_skip,
                                joint_encoder=joint_encoder)
  vars_to_restore = util.get_vars_to_save_and_restore(model_ckpt)
  saver = tf.train.Saver(vars_to_restore)
  sv = tf.train.Supervisor(logdir='/tmp/', saver=None)
  with sv.managed_session() as sess:
    saver.restore(sess, model_ckpt)
    if not gfile.Exists(output_dir):
      gfile.MakeDirs(output_dir)
    logging.info('Predictions will be saved in %s.', output_dir)

    # Collect all images to run inference on.
    im_files, basepath_in = collect_input_images(input_dir, input_list_file,
                                                 file_extension)
    if shuffle:
      logging.info('Shuffling data...')
      np.random.shuffle(im_files)
    logging.info('Running inference on %d files.', len(im_files))

    # Create missing output folders and pre-compute target directories.
    output_dirs = create_output_dirs(im_files, basepath_in, output_dir)

    # Run depth prediction network.
    if depth:
      im_batch = []
      for i in range(len(im_files)):
        if i % 100 == 0:
          logging.info('%s of %s files processed.', i, len(im_files))

        # Read image and run inference.
        if inference_mode == INFERENCE_MODE_SINGLE:
          if inference_crop == INFERENCE_CROP_NONE:
            im = util.load_image(im_files[i], resize=(img_width, img_height))
          elif inference_crop == INFERENCE_CROP_CITYSCAPES:
            im = util.crop_cityscapes(util.load_image(im_files[i]),
                                      resize=(img_width, img_height))
        elif inference_mode == INFERENCE_MODE_TRIPLETS:
          im = util.load_image(im_files[i], resize=(img_width * 3, img_height))
          im = im[:, img_width:img_width*2]
        if flip_for_depth:
          im = np.flip(im, axis=1)
        im_batch.append(im)

        if len(im_batch) == batch_size or i == len(im_files) - 1:
          # Call inference on batch.
          for _ in range(batch_size - len(im_batch)):  # Fill up batch.
            im_batch.append(np.zeros(shape=(img_height, img_width, 3),
                                     dtype=np.float32))
          im_batch = np.stack(im_batch, axis=0)
          est_depth = inference_model.inference_depth(im_batch, sess)
          if flip_for_depth:
            est_depth = np.flip(est_depth, axis=2)
            im_batch = np.flip(im_batch, axis=2)

          for j in range(len(im_batch)):
            color_map = util.normalize_depth_for_display(
                np.squeeze(est_depth[j]))
            visualization = np.concatenate((im_batch[j], color_map), axis=0)
            # Save raw prediction and color visualization. Extract filename
            # without extension from full path: e.g. path/to/input_dir/folder1/
            # file1.png -> file1
            k = i - len(im_batch) + 1 + j
            filename_root = os.path.splitext(os.path.basename(im_files[k]))[0]
            pref = '_flip' if flip_for_depth else ''
            output_raw = os.path.join(
                output_dirs[k], filename_root + pref + '.npy')
            output_vis = os.path.join(
                output_dirs[k], filename_root + pref + '.png')
            with gfile.Open(output_raw, 'wb') as f:
              np.save(f, est_depth[j])
            util.save_image(output_vis, visualization, file_extension)
          im_batch = []

    # Run egomotion network.
    if egomotion:
      if inference_mode == INFERENCE_MODE_SINGLE:
        # Run regular egomotion inference loop.
        input_image_seq = []
        input_seg_seq = []
        current_sequence_dir = None
        current_output_handle = None
        for i in range(len(im_files)):
          sequence_dir = os.path.dirname(im_files[i])
          if sequence_dir != current_sequence_dir:
            # Assume start of a new sequence, since this image lies in a
            # different directory than the previous ones.
            # Clear egomotion input buffer.
            output_filepath = os.path.join(output_dirs[i], 'egomotion.txt')
            if current_output_handle is not None:
              current_output_handle.close()
            current_sequence_dir = sequence_dir
            logging.info('Writing egomotion sequence to %s.', output_filepath)
            current_output_handle = gfile.Open(output_filepath, 'w')
            input_image_seq = []
          im = util.load_image(im_files[i], resize=(img_width, img_height))
          input_image_seq.append(im)
          if use_masks:
            im_seg_path = im_files[i].replace('.%s' % file_extension,
                                              '-seg.%s' % file_extension)
            if not gfile.Exists(im_seg_path):
              raise ValueError('No segmentation mask %s has been found for '
                               'image %s. If none are available, disable '
                               'use_masks.' % (im_seg_path, im_files[i]))
            input_seg_seq.append(util.load_image(im_seg_path,
                                                 resize=(img_width, img_height),
                                                 interpolation='nn'))

          if len(input_image_seq) < seq_length:  # Buffer not filled yet.
            continue
          if len(input_image_seq) > seq_length:  # Remove oldest entry.
            del input_image_seq[0]
            if use_masks:
              del input_seg_seq[0]

          input_image_stack = np.concatenate(input_image_seq, axis=2)
          input_image_stack = np.expand_dims(input_image_stack, axis=0)
          if use_masks:
            input_image_stack = mask_image_stack(input_image_stack,
                                                 input_seg_seq)
          est_egomotion = np.squeeze(inference_model.inference_egomotion(
              input_image_stack, sess))
          egomotion_str = []
          for j in range(seq_length - 1):
            egomotion_str.append(','.join([str(d) for d in est_egomotion[j]]))
          current_output_handle.write(
              str(i) + ' ' + ' '.join(egomotion_str) + '\n')
        if current_output_handle is not None:
          current_output_handle.close()
      elif inference_mode == INFERENCE_MODE_TRIPLETS:
        written_before = []
        for i in range(len(im_files)):
          im = util.load_image(im_files[i], resize=(img_width * 3, img_height))
          input_image_stack = np.concatenate(
              [im[:, :img_width], im[:, img_width:img_width*2],
               im[:, img_width*2:]], axis=2)
          input_image_stack = np.expand_dims(input_image_stack, axis=0)
          if use_masks:
            im_seg_path = im_files[i].replace('.%s' % file_extension,
                                              '-seg.%s' % file_extension)
            if not gfile.Exists(im_seg_path):
              raise ValueError('No segmentation mask %s has been found for '
                               'image %s. If none are available, disable '
                               'use_masks.' % (im_seg_path, im_files[i]))
            seg = util.load_image(im_seg_path,
                                  resize=(img_width * 3, img_height),
                                  interpolation='nn')
            input_seg_seq = [seg[:, :img_width], seg[:, img_width:img_width*2],
                             seg[:, img_width*2:]]
            input_image_stack = mask_image_stack(input_image_stack,
                                                 input_seg_seq)
          est_egomotion = inference_model.inference_egomotion(
              input_image_stack, sess)
          est_egomotion = np.squeeze(est_egomotion)
          egomotion_1_2 = ','.join([str(d) for d in est_egomotion[0]])
          egomotion_2_3 = ','.join([str(d) for d in est_egomotion[1]])

          output_filepath = os.path.join(output_dirs[i], 'egomotion.txt')
          file_mode = 'w' if output_filepath not in written_before else 'a'
          with gfile.Open(output_filepath, file_mode) as current_output_handle:
            current_output_handle.write(str(i) + ' ' + egomotion_1_2 + ' ' +
                                        egomotion_2_3 + '\n')
          written_before.append(output_filepath)
      logging.info('Done.')


def mask_image_stack(input_image_stack, input_seg_seq):
  """Masks out moving image contents by using the segmentation masks provided.

  This can lead to better odometry accuracy for motion models, but is optional
  to use. Is only called if use_masks is enabled.
  Args:
    input_image_stack: The input image stack of shape (1, H, W, seq_length).
    input_seg_seq: List of segmentation masks with seq_length elements of shape
                   (H, W, C) for some number of channels C.

  Returns:
    Input image stack with detections provided by segmentation mask removed.
  """
  background = [mask == 0 for mask in input_seg_seq]
  background = reduce(lambda m1, m2: m1 & m2, background)
  # If masks are RGB, assume all channels to be the same. Reduce to the first.
  if background.ndim == 3 and background.shape[2] > 1:
    background = np.expand_dims(background[:, :, 0], axis=2)
  elif background.ndim == 2:  # Expand.
    background = np.expand_dism(background, axis=2)
  # background is now of shape (H, W, 1).
  background_stack = np.tile(background, [1, 1, input_image_stack.shape[3]])
  return np.multiply(input_image_stack, background_stack)


def collect_input_images(input_dir, input_list_file, file_extension):
  """Collects all input images that are to be processed."""
  if input_dir is not None:
    im_files = _recursive_glob(input_dir, '*.' + file_extension)
    basepath_in = os.path.normpath(input_dir)
  elif input_list_file is not None:
    im_files = util.read_text_lines(input_list_file)
    basepath_in = os.path.dirname(input_list_file)
    im_files = [os.path.join(basepath_in, f) for f in im_files]
  im_files = [f for f in im_files if 'disp' not in f and '-seg' not in f and
              '-fseg' not in f and '-flip' not in f]
  return sorted(im_files), basepath_in


def create_output_dirs(im_files, basepath_in, output_dir):
  """Creates required directories, and returns output dir for each file."""
  output_dirs = []
  for i in range(len(im_files)):
    relative_folder_in = os.path.relpath(
        os.path.dirname(im_files[i]), basepath_in)
    absolute_folder_out = os.path.join(output_dir, relative_folder_in)
    if not gfile.IsDirectory(absolute_folder_out):
      gfile.MakeDirs(absolute_folder_out)
    output_dirs.append(absolute_folder_out)
  return output_dirs


def _recursive_glob(treeroot, pattern):
  results = []
  for base, _, files in os.walk(treeroot):
    files = fnmatch.filter(files, pattern)
    results.extend(os.path.join(base, f) for f in files)
  return results


def main(_):
  #if (flags.input_dir is None) == (flags.input_list_file is None):
  #  raise ValueError('Exactly one of either input_dir or input_list_file has '
  #                   'to be provided.')
  #if not flags.depth and not flags.egomotion:
  #  raise ValueError('At least one of the depth and egomotion network has to '
  #                   'be called for inference.')
  #if (flags.inference_mode == inference_lib.INFERENCE_MODE_TRIPLETS and
  #    flags.seq_length != 3):
  #  raise ValueError('For sequence lengths other than three, single inference '
  #                   'mode has to be used.')

  _run_inference(output_dir=FLAGS.output_dir,
                 file_extension=FLAGS.file_extension,
                 depth=FLAGS.depth,
                 egomotion=FLAGS.egomotion,
                 model_ckpt=FLAGS.model_ckpt,
                 input_dir=FLAGS.input_dir,
                 input_list_file=FLAGS.input_list_file,
                 batch_size=FLAGS.batch_size,
                 img_height=FLAGS.img_height,
                 img_width=FLAGS.img_width,
                 seq_length=FLAGS.seq_length,
                 architecture=FLAGS.architecture,
                 imagenet_norm=FLAGS.imagenet_norm,
                 use_skip=FLAGS.use_skip,
                 joint_encoder=FLAGS.joint_encoder,
                 shuffle=FLAGS.shuffle,
                 flip_for_depth=FLAGS.flip,
                 inference_mode=FLAGS.inference_mode,
                 inference_crop=FLAGS.inference_crop,
                 use_masks=FLAGS.use_masks)


if __name__ == '__main__':
  app.run(main)
