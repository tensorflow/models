
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

"""Contains common utilities and functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import locale
import os
import re
from absl import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
gfile = tf.gfile


CMAP_DEFAULT = 'plasma'
# Defines the cropping that is applied to the Cityscapes dataset with respect to
# the original raw input resolution.
CITYSCAPES_CROP = [256, 768, 192, 1856]


def crop_cityscapes(im, resize=None):
  ymin, ymax, xmin, xmax = CITYSCAPES_CROP
  im = im[ymin:ymax, xmin:xmax]
  if resize is not None:
    im = cv2.resize(im, resize)
  return im


def gray2rgb(im, cmap=CMAP_DEFAULT):
  cmap = plt.get_cmap(cmap)
  result_img = cmap(im.astype(np.float32))
  if result_img.shape[2] > 3:
    result_img = np.delete(result_img, 3, 2)
  return result_img


def load_image(img_file, resize=None, interpolation='linear'):
  """Load image from disk. Output value range: [0,1]."""
  im_data = np.fromstring(gfile.Open(img_file).read(), np.uint8)
  im = cv2.imdecode(im_data, cv2.IMREAD_COLOR)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  if resize and resize != im.shape[:2]:
    ip = cv2.INTER_LINEAR if interpolation == 'linear' else cv2.INTER_NEAREST
    im = cv2.resize(im, resize, interpolation=ip)
  return np.array(im, dtype=np.float32) / 255.0


def save_image(img_file, im, file_extension):
  """Save image from disk. Expected input value range: [0,1]."""
  im = (im * 255.0).astype(np.uint8)
  with gfile.Open(img_file, 'w') as f:
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    _, im_data = cv2.imencode('.%s' % file_extension, im)
    f.write(im_data.tostring())


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None,
                                cmap=CMAP_DEFAULT):
  """Converts a depth map to an RGB image."""
  # Convert to disparity.

  disp = 1.0 / (depth + 1e-6)
  if normalizer is not None:
    disp /= normalizer
  else:
    disp /= (np.percentile(disp, pc) + 1e-6)
  disp = np.clip(disp, 0, 1)
  disp = gray2rgb(disp, cmap=cmap)
  keep_h = int(disp.shape[0] * (1 - crop_percent))
  disp = disp[:keep_h]
  return disp


def get_seq_start_end(target_index, seq_length, sample_every=1):
  """Returns absolute seq start and end indices for a given target frame."""
  half_offset = int((seq_length - 1) / 2) * sample_every
  end_index = target_index + half_offset
  start_index = end_index - (seq_length - 1) * sample_every
  return start_index, end_index


def get_seq_middle(seq_length):
  """Returns relative index for the middle frame in sequence."""
  half_offset = int((seq_length - 1) / 2)
  return seq_length - 1 - half_offset


def info(obj):
  """Return info on shape and dtype of a numpy array or TensorFlow tensor."""
  if obj is None:
    return 'None.'
  elif isinstance(obj, list):
    if obj:
      return 'List of %d... %s' % (len(obj), info(obj[0]))
    else:
      return 'Empty list.'
  elif isinstance(obj, tuple):
    if obj:
      return 'Tuple of %d... %s' % (len(obj), info(obj[0]))
    else:
      return 'Empty tuple.'
  else:
    if is_a_numpy_array(obj):
      return 'Array with shape: %s, dtype: %s' % (obj.shape, obj.dtype)
    else:
      return str(obj)


def is_a_numpy_array(obj):
  """Returns true if obj is a numpy array."""
  return type(obj).__module__ == np.__name__


def count_parameters(also_print=True):
  """Cound the number of parameters in the model.

  Args:
    also_print: Boolean.  If True also print the numbers.

  Returns:
    The total number of parameters.
  """
  total = 0
  if also_print:
    logging.info('Model Parameters:')
  for (_, v) in get_vars_to_save_and_restore().items():
    shape = v.get_shape()
    if also_print:
      logging.info('%s %s: %s', v.op.name, shape,
                   format_number(shape.num_elements()))
    total += shape.num_elements()
  if also_print:
    logging.info('Total: %s', format_number(total))
  return total


def get_vars_to_save_and_restore(ckpt=None):
  """Returns list of variables that should be saved/restored.

  Args:
    ckpt: Path to existing checkpoint.  If present, returns only the subset of
        variables that exist in given checkpoint.

  Returns:
    List of all variables that need to be saved/restored.
  """
  model_vars = tf.trainable_variables()
  # Add batchnorm variables.
  bn_vars = [v for v in tf.global_variables()
             if 'moving_mean' in v.op.name or 'moving_variance' in v.op.name or
             'mu' in v.op.name or 'sigma' in v.op.name or
             'global_scale_var' in v.op.name]
  model_vars.extend(bn_vars)
  model_vars = sorted(model_vars, key=lambda x: x.op.name)
  mapping = {}
  if ckpt is not None:
    ckpt_var = tf.contrib.framework.list_variables(ckpt)
    ckpt_var_names = [name for (name, unused_shape) in ckpt_var]
    ckpt_var_shapes = [shape for (unused_name, shape) in ckpt_var]
    not_loaded = list(ckpt_var_names)
    for v in model_vars:
      if v.op.name not in ckpt_var_names:
        # For backward compatibility, try additional matching.
        v_additional_name = v.op.name.replace('egomotion_prediction/', '')
        if v_additional_name in ckpt_var_names:
          # Check if shapes match.
          ind = ckpt_var_names.index(v_additional_name)
          if ckpt_var_shapes[ind] == v.get_shape():
            mapping[v_additional_name] = v
            not_loaded.remove(v_additional_name)
            continue
          else:
            logging.warn('Shape mismatch, will not restore %s.', v.op.name)
        logging.warn('Did not find var %s in checkpoint: %s', v.op.name,
                     os.path.basename(ckpt))
      else:
        # Check if shapes match.
        ind = ckpt_var_names.index(v.op.name)
        if ckpt_var_shapes[ind] == v.get_shape():
          mapping[v.op.name] = v
          not_loaded.remove(v.op.name)
        else:
          logging.warn('Shape mismatch, will not restore %s.', v.op.name)
    if not_loaded:
      logging.warn('The following variables in the checkpoint were not loaded:')
      for varname_not_loaded in not_loaded:
        logging.info('%s', varname_not_loaded)
  else:  # just get model vars.
    for v in model_vars:
      mapping[v.op.name] = v
  return mapping


def get_imagenet_vars_to_restore(imagenet_ckpt):
  """Returns dict of variables to restore from ImageNet-checkpoint."""
  vars_to_restore_imagenet = {}
  ckpt_var_names = tf.contrib.framework.list_variables(imagenet_ckpt)
  ckpt_var_names = [name for (name, unused_shape) in ckpt_var_names]
  model_vars = tf.global_variables()
  for v in model_vars:
    if 'global_step' in v.op.name: continue
    mvname_noprefix = v.op.name.replace('depth_prediction/', '')
    mvname_noprefix = mvname_noprefix.replace('moving_mean', 'mu')
    mvname_noprefix = mvname_noprefix.replace('moving_variance', 'sigma')
    if mvname_noprefix in ckpt_var_names:
      vars_to_restore_imagenet[mvname_noprefix] = v
    else:
      logging.info('The following variable will not be restored from '
                   'pretrained ImageNet-checkpoint: %s', mvname_noprefix)
  return vars_to_restore_imagenet


def format_number(n):
  """Formats number with thousands commas."""
  locale.setlocale(locale.LC_ALL, 'en_US')
  return locale.format('%d', n, grouping=True)


def atoi(text):
  return int(text) if text.isdigit() else text


def natural_keys(text):
  return [atoi(c) for c in re.split(r'(\d+)', text)]


def read_text_lines(filepath):
  with tf.gfile.Open(filepath, 'r') as f:
    lines = f.readlines()
  lines = [l.rstrip() for l in lines]
  return lines
