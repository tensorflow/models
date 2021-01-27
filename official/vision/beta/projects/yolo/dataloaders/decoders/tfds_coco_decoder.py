import tensorflow as tf
from official.vision.beta.dataloaders import decoder


class MSCOCODecoder(decoder.Decoder):
  """Tensorflow Example proto decoder."""

  def __init__(self, include_mask=False, regenerate_source_id=False):
    self._include_mask = include_mask
    self._regenerate_source_id = regenerate_source_id

  def decode(self, sample):
    """Decode the serialized example
    Args:
      sample: a dictonary example produced by tfds.

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
    """
    decoded_tensors = {
        'source_id': sample['image/id'],
        'image': sample['image'],
        'height': tf.shape(sample['image'])[0],
        'width': tf.shape(sample['image'])[1],
        'groundtruth_classes': sample['objects']['label'],
        'groundtruth_is_crowd': sample['objects']['is_crowd'],
        'groundtruth_area': sample['objects']['area'],
        'groundtruth_boxes': sample['objects']['bbox'],
    }
    return decoded_tensors
