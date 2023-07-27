# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""COCO data loader for Pix2Seq."""

from typing import Tuple
import tensorflow as tf

from official.vision.dataloaders import decoder
from official.vision.dataloaders import parser
from official.vision.ops import preprocess_ops
from official.projects.rngdet.dataloaders import sampler as rngdet_sampler


class Decoder(decoder.Decoder):
  """A tf.Example decoder for RNGDet."""

  def __init__(self):
    
    self._keys_to_features = {
    "image/encoded": tf.io.FixedLenFeature((), tf.string),
    "image/intersection": tf.io.FixedLenFeature((), tf.string),
    "image/segment": tf.io.FixedLenFeature((), tf.string),
    "edges/id": tf.io.VarLenFeature(tf.int64),
    "edges/src": tf.io.VarLenFeature(tf.int64),
    "edges/dst": tf.io.VarLenFeature(tf.int64),
    "edges/vertices": tf.io.VarLenFeature(tf.int64),
    "edges/orientation": tf.io.VarLenFeature(tf.int64),
    "edges/length": tf.io.VarLenFeature(tf.int64),
    "vertices/id": tf.io.VarLenFeature(tf.int64),
    "vertices/x": tf.io.VarLenFeature(tf.int64),
    "vertices/y": tf.io.VarLenFeature(tf.int64),
    }

  def _decode_image(self, parsed_tensors):
    """Decodes the image and set its static shape."""
    image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
    image.set_shape([None, None, 3])
    
    intsec = tf.io.decode_image(parsed_tensors['image/intersection'], channels=3)
    intsec.set_shape([None, None, 3])
    
    segment = tf.io.decode_image(parsed_tensors['image/segment'], channels=3)
    segment.set_shape([None, None, 3])
    
    return image, intsec, segment

  def decode(self, serialized_example):
    parsed_tensors = tf.io.parse_single_example(
        serialized=serialized_example, features=self._keys_to_features)
    for k in parsed_tensors:
      if isinstance(parsed_tensors[k], tf.SparseTensor):
        if parsed_tensors[k].dtype == tf.string:
          parsed_tensors[k] = tf.sparse.to_dense(
              parsed_tensors[k], default_value='')
        else:
          parsed_tensors[k] = tf.sparse.to_dense(
              parsed_tensors[k], default_value=0)
    image, intsec, segment = self._decode_image(parsed_tensors)
  
    decoded_tensors = {
        'image': image,
        'intersection': intsec,
        'segment': segment,
        "edges/id": parsed_tensors['edges/id'],
        "edges/src": parsed_tensors['edges/src'],
        "edges/dst": parsed_tensors['edges/dst'],
        "edges/vertices": parsed_tensors['edges/vertices'],
        "edges/orientation": parsed_tensors['edges/orientation'],
        "edges/length": parsed_tensors['edges/length'],
        "vertices/id": parsed_tensors['vertices/id'],
        "vertices/x": parsed_tensors['vertices/x'],
        "vertices/y": parsed_tensors['vertices/y'],
    }
    
    return decoded_tensors


class Parser(parser.Parser):
  """Parse an image and its annotations into a dictionary of tensors."""

  def __init__(
      self,
      eos_token_weight: float = 0.1,
      output_size: Tuple[int, int] = (1333, 1333)
  ):
    self._eos_token_weight = eos_token_weight
    self._output_size = output_size
    
  def parse_fn(self, is_training):
    """Returns a parse fn that reads and parses raw tensors from the decoder.

    Args:
      is_training: a `bool` to indicate whether it is in training mode.

    Returns:
      parse: a `callable` that takes the serialized example and generate the
        images, labels tuple where labels is a dict of Tensors that contains
        labels.
    """
    def parse(decoded_tensors):
      """Parses the serialized example data."""
      if is_training:
        return self._parse_train_data(decoded_tensors)
      else:
        return self._parse_eval_data(decoded_tensors)

    return parse

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    edges = {
      'id': data['edges/id'].numpy(),
      'src': data['edges/src'].numpy(),
      'dst': data['edges/dst'].numpy(),
      'vertices': data['edges/vertices'],
      'orientation': data['edges/orientation'],
      'length': data['edges/length'].numpy()
    }
    vertices = {
      'id': data['vertices/id'].numpy(),
      'x': data['vertices/x'].numpy(),
      'y': data['vertices/y'].numpy()
    }
    sampler = rngdet_sampler.Sampler(
        data['image'],
        data['intersection'],
        data['segment'],
        edges, vertices)
    while 1:
      if sampler.finish_current_image:
        break
        # crop
      v_current = sampler.current_coord.copy()
      sat_ROI, label_masks_ROI ,historical_ROI = sampler.crop_ROI(sampler.current_coord)
      print(v_current)
      # vertices in the next step
      v_nexts, ahead_segments = sampler.step_expert_BC_sampler()
      # save training sample
      gt_probs, gt_coords, list_len = sampler.calcualte_label(v_current,v_nexts)
    exit()

    # Normalizes image with mean and std pixel values.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=self._output_size,
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)
    
    out = tf.stack([image, image], 0)

    return out
  
  def _parse_eval_data(self, data):
    """Parses data for training and evaluation."""

    # Gets original image.
    image = data['image']

    # Normalizes image with mean and std pixel values.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=self._output_size,
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)

    return image