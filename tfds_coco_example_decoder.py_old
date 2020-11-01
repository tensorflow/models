import tensorflow_datasets as tfds 
import tensorflow as tf
from official.vision.beta.dataloaders import decoder

import matplotlib.pyplot as plt
import cv2


class TfdsExampleDecoder(decoder.Decoder):
  """Tensorflow Dataset Example proto decoder."""
  def __init__(self,
               include_mask=False,
               regenerate_source_id=False):
    self._include_mask = include_mask
    self._regenerate_source_id = regenerate_source_id

  def decode(self, serialized_example):
    """Decode the serialized example.
    Args:
      serialized_example: a single serialized tf.Example string.
    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - source_id: a string scalar tensor.
        - image: a uint8 tensor of shape [None, None, 3].
        - height: an integer scalar tensor.
        - width: an integer scalar tensor.
        - groundtruth_classes: a int64 tensor of shape [None].
        - groundtruth_is_crowd: a bool tensor of shape [None].
        - groundtruth_area: a float32 tensor of shape [None].
        - groundtruth_boxes: a float32 tensor of shape [None, 4].
        - groundtruth_instance_masks: a float32 tensor of shape
            [None, None, None].
        - groundtruth_instance_masks_png: a string tensor of shape [None].
    """
    decoded_tensors = {
        'source_id': serialized_example['image/id'],
        'image': serialized_example['image'],
        'height': tf.shape(serialized_example['image'])[0],
        'width':  tf.shape(serialized_example['image'])[1],
        'groundtruth_classes': serialized_example['objects']['label'],
        'groundtruth_is_crowd': serialized_example['objects']['is_crowd'],
        'groundtruth_area': serialized_example['objects']['area'],
        'groundtruth_boxes': serialized_example['objects']['bbox'],
    }
    return decoded_tensors

