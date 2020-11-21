import tensorflow as tf
from official.vision.beta.dataloaders import decoder

class MSCOCODecoder(decoder.Decoder):
  """Tensorflow Example proto decoder."""
  def __init__(self,
               include_mask=False,
               regenerate_source_id=False):
    self._include_mask = include_mask
    self._regenerate_source_id = regenerate_source_id
  def decode(self, sample):
    """Decode the serialized example"""
    decoded_tensors = {
        'source_id': sample['image/id'],
        'image': sample['image'],
        'height': tf.shape(sample['image'])[0],
        'width':  tf.shape(sample['image'])[1],
        'groundtruth_classes': sample['objects']['label'],
        'groundtruth_is_crowd': sample['objects']['is_crowd'],
        'groundtruth_area': sample['objects']['area'],
        'groundtruth_boxes': sample['objects']['bbox'],
    }
    return decoded_tensors