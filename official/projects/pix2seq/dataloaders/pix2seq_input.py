# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
import tensorflow as tf, tf_keras

from official.projects.pix2seq import utils
from official.projects.pix2seq.configs import pix2seq as pix2seq_cfg
from official.projects.simclr.dataloaders import preprocess_ops as simclr_preprocess_ops
from official.vision.dataloaders import parser
from official.vision.ops import box_ops
from official.vision.ops import preprocess_ops

RESIZE_SCALES = (480, 512, 544, 576, 608, 640)


class Parser(parser.Parser):
  """Parse an image and its annotations into a dictionary of tensors."""

  def __init__(
      self,
      eos_token_weight: float = 0.1,
      output_size: Tuple[int, int] = (1333, 1333),
      max_num_boxes: int = 100,
      aug_rand_hflip=True,
      aug_scale_min=0.3,
      aug_scale_max=2.0,
      aug_color_jitter_strength: float = 0.5,
      aug_color_jitter_impl='simclrv2',
      coord_vocab_shift=1000,
      quantization_bins=1000,
      skip_crowd_during_training=True,
      label_shift: int = 0,
  ):
    self._eos_token_weight = eos_token_weight
    self._output_size = output_size
    self._max_num_boxes = max_num_boxes
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max
    self._aug_color_jitter_strength = aug_color_jitter_strength
    self._aug_color_jitter_impl = aug_color_jitter_impl
    self._coord_vocab_shift = coord_vocab_shift
    self._quantization_bins = quantization_bins
    self._skip_crowd_during_training = skip_crowd_during_training
    self._label_shift = label_shift

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    classes = data['groundtruth_classes'] + self._label_shift
    boxes = data['groundtruth_boxes']

    is_crowds = data['groundtruth_is_crowd']
    # Skips annotations with `is_crowd` = True.
    if self._skip_crowd_during_training:
      num_groundtruths = tf.shape(classes)[0]
      with tf.control_dependencies([num_groundtruths, is_crowds]):
        indices = tf.cond(
            tf.greater(tf.size(is_crowds), 0),
            lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
            lambda: tf.cast(tf.range(num_groundtruths), tf.int64),
        )
      classes = tf.gather(classes, indices)
      boxes = tf.gather(boxes, indices)

    # Gets original image.
    image = data['image']

    # Normalizes image with mean and std pixel values.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Color jitter.
    image = simclr_preprocess_ops.random_color_jitter(
        image=image,
        color_jitter_strength=self._aug_color_jitter_strength,
        impl=self._aug_color_jitter_impl,
    )
    image = tf.clip_by_value(image, 0.0, 1.0)
    image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)

    image_shape = tf.shape(image)[:2]
    boxes = box_ops.denormalize_boxes(boxes, image_shape)

    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=self._output_size,
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)

    boxes = preprocess_ops.resize_and_crop_boxes(
        boxes, image_info[2, :], image_info[1, :], image_info[3, :]
    )
    boxes = box_ops.normalize_boxes(boxes, image_info[1, :])

    # Filters out ground truth boxes that are all zeros.
    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)

    boxes, classes = utils.reorder_object_instances(boxes, classes, 'random')
    boxes, classes = utils.inject_noise_bbox(
        boxes, classes, self._max_num_boxes
    )

    boxes = utils.clip_or_pad_to_max_len(boxes, self._max_num_boxes, 0)
    classes = utils.clip_or_pad_to_max_len(classes, self._max_num_boxes, 0)

    outputs = self.build_response_seq_from_bbox(
        boxes, classes, self._coord_vocab_shift, self._quantization_bins
    )
    response_seq, response_seq_class_m, token_weights = outputs
    prompt_seq = utils.build_prompt_seq_from_task_id(
        pix2seq_cfg.OD_ID, response_seq
    )  # (1)
    input_seq = tf.concat([prompt_seq, response_seq_class_m], -1)
    target_seq = tf.concat([prompt_seq, response_seq], -1)

    backgrnd_val = 0.3
    image = backgrnd_val + tf.image.pad_to_bounding_box(
        image - backgrnd_val, 0, 0, self._output_size[0], self._output_size[1]
    )

    input_seq = utils.clip_or_pad_to_max_len(
        input_seq, self._max_num_boxes * 5 + 1, -1)
    target_seq = utils.clip_or_pad_to_max_len(
        target_seq, self._max_num_boxes * 5 + 1, -1
    )

    input_seq, target_seq = input_seq[..., :-1], target_seq[..., 1:]
    token_weights = utils.clip_or_pad_to_max_len(
        token_weights, self._max_num_boxes * 5, -1
    )

    # Assign lower weights for ending/padding tokens.
    token_weights = tf.where(
        target_seq == pix2seq_cfg.PADDING_TOKEN,
        tf.zeros_like(token_weights) + self._eos_token_weight,
        token_weights,
    )

    labels = {
        'targets': target_seq,
        'weights': token_weights,
        'inputs': input_seq,
    }

    return image, labels

  def build_response_seq_from_bbox(
      self,
      bbox,
      label,
      coord_vocab_shift,
      quantization_bins,
      noise_bbox_weight=1.0,
      class_label_corruption='rand_n_fake_cls',
  ):
    """Build target seq from bounding bboxes for object detection.

    Objects are serialized using the format of yxyxc.

    Args:
      bbox: `float` bounding box of shape (n, 4).
      label: `int` label of shape (n).
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.
      quantization_bins: `int`.
      noise_bbox_weight: `float` on the token weights for noise bboxes.
      class_label_corruption: `string` specifying how labels are corrupted for
        the input_seq.

    Returns:
      discrete sequences with shape (seqlen).
    """
    # Bbox and label quantization.
    is_padding = tf.expand_dims(tf.equal(label, 0), -1)
    quantized_bbox = utils.quantize(bbox, quantization_bins)
    quantized_bbox = quantized_bbox + coord_vocab_shift
    quantized_bbox = tf.where(
        is_padding, tf.zeros_like(quantized_bbox), quantized_bbox
    )
    new_label = tf.expand_dims(label + pix2seq_cfg.BASE_VOCAB_SHIFT, -1)
    new_label = tf.where(is_padding, tf.zeros_like(new_label), new_label)
    lb_shape = tf.shape(new_label)

    # Bbox and label serialization.
    response_seq = tf.concat([quantized_bbox, new_label], axis=-1)

    response_seq = tf.reshape(response_seq, [-1])
    rand_cls = pix2seq_cfg.BASE_VOCAB_SHIFT + tf.random.uniform(
        lb_shape,
        0,
        coord_vocab_shift - pix2seq_cfg.BASE_VOCAB_SHIFT,
        dtype=new_label.dtype,
    )
    fake_cls = pix2seq_cfg.FAKE_CLASS_TOKEN + tf.zeros_like(new_label)
    rand_n_fake_cls = tf.where(
        tf.random.uniform(lb_shape) > 0.5, rand_cls, fake_cls
    )
    real_n_fake_cls = tf.where(
        tf.random.uniform(lb_shape) > 0.5, new_label, fake_cls
    )
    real_n_rand_n_fake_cls = tf.where(
        tf.random.uniform(lb_shape) > 0.5, new_label, rand_n_fake_cls
    )
    label_mapping = {
        'none': new_label,
        'rand_cls': rand_cls,
        'real_n_fake_cls': real_n_fake_cls,
        'rand_n_fake_cls': rand_n_fake_cls,
        'real_n_rand_n_fake_cls': real_n_rand_n_fake_cls,
    }
    new_label_m = label_mapping[class_label_corruption]
    new_label_m = tf.where(is_padding, tf.zeros_like(new_label_m), new_label_m)

    response_seq_class_m = tf.concat([quantized_bbox, new_label_m], axis=-1)
    response_seq_class_m = tf.reshape(response_seq_class_m, [-1])

    # Get token weights.
    is_real = tf.cast(
        tf.not_equal(new_label, pix2seq_cfg.FAKE_CLASS_TOKEN), tf.float32
    )
    bbox_weight = tf.tile(is_real, [1, 4])
    label_weight = is_real + (1.0 - is_real) * noise_bbox_weight
    token_weights = tf.concat([bbox_weight, label_weight], -1)
    token_weights = tf.reshape(token_weights, [-1])

    return response_seq, response_seq_class_m, token_weights

  def _parse_eval_data(self, data):
    """Parses data for training and evaluation."""
    classes = data['groundtruth_classes'] + self._label_shift
    boxes = data['groundtruth_boxes']
    is_crowd = data['groundtruth_is_crowd']

    # Gets original image and its size.
    image = data['image']
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image_shape = tf.shape(image)[:2]
    boxes = box_ops.denormalize_boxes(boxes, image_shape)
    gt_boxes = boxes
    image, image_info = preprocess_ops.resize_image(
        image, min(self._output_size), max(self._output_size)
    )
    boxes = preprocess_ops.resize_and_crop_boxes(
        boxes, image_info[2, :], image_info[1, :], image_info[3, :]
    )
    scale = tf.cast(
        tf.concat([self._output_size, self._output_size], -1), boxes.dtype
    )
    boxes = boxes / scale

    # Filters out ground truth boxes that are all zeros.
    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    is_crowd = tf.gather(is_crowd, indices)

    prompt_seq = tf.constant([pix2seq_cfg.OD_ID], dtype=tf.int64)
    backgrnd_val = 0.3
    image = backgrnd_val + tf.image.pad_to_bounding_box(
        image - backgrnd_val, 0, 0, self._output_size[0], self._output_size[1]
    )

    labels = {
        'prompt': prompt_seq,
        'classes': preprocess_ops.clip_or_pad_to_fixed_size(
            classes, self._max_num_boxes
        ),
        'boxes': preprocess_ops.clip_or_pad_to_fixed_size(
            boxes, self._max_num_boxes
        ),
    }
    labels.update({
        'id': int(data['source_id']),
        'image_info': image_info,
        'is_crowd': preprocess_ops.clip_or_pad_to_fixed_size(
            is_crowd, self._max_num_boxes
        ),
        'gt_boxes': preprocess_ops.clip_or_pad_to_fixed_size(
            gt_boxes, self._max_num_boxes
        ),
    })

    return image, labels
