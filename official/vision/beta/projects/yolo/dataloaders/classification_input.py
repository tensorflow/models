"""Classification parser."""

# Import libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.ops import augment


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               aug_policy,
               scale=[128, 448],
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      num_classes: `float`, number of classes.
      aug_policy: An optional Augmentation object to choose from AutoAugment and
        RandAugment.
      scale: A `List[int]`, minimum and maximum image shape range.
      dtype: `str`, cast output image in dtype. It can be 'float32', 'float16',
        or 'bfloat16'.
    """
    self._output_size = output_size
    if aug_policy:
      if aug_policy == 'autoaug':
        self._augmenter = augment.AutoAugment()
      elif aug_policy == 'randaug':
        self._augmenter = augment.RandAugment(num_layers=2, magnitude=20)
      else:
        raise ValueError(
            'Augmentation policy {} not supported.'.format(aug_policy))
    else:
      self._augmenter = None

    self._scale = scale
    if dtype == 'float32':
      self._dtype = tf.float32
    elif dtype == 'float16':
      self._dtype = tf.float16
    elif dtype == 'bfloat16':
      self._dtype = tf.bfloat16
    else:
      raise ValueError('dtype {!r} is not supported!'.format(dtype))

  def _parse_train_data(self, decoded_tensors):
    """Generates images and labels that are usable for model training.
     Args:
       decoded_tensors: a dict of Tensors produced by the decoder.
     Returns:
       images: the image tensor.
       labels: a dict of Tensors that contains labels.
    """
    image = tf.io.decode_image(decoded_tensors['image/encoded'])
    image.set_shape((None, None, 3))

    image = tf.image.resize_with_pad(
        image,
        target_width=self._output_size[0],
        target_height=self._output_size[1])

    scale = tf.random.uniform([],
                              minval=self._scale[0],
                              maxval=self._scale[1],
                              dtype=tf.int32)
    if scale > self._output_size[0]:
      image = tf.image.resize_with_crop_or_pad(
          image, target_height=scale, target_width=scale)
    else:
      image = tf.image.random_crop(image, (scale, scale, 3))

    if self._augmenter is not None:
      image = self._augmenter.distort(image)

    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255
    image = tf.image.resize(image, (self._output_size[0], self._output_size[1]))

    label = decoded_tensors['image/class/label']
    return image, label

  def _parse_eval_data(self, decoded_tensors):
    """Generates images and labels that are usable for model evaluation.
    Args:
      decoded_tensors: a dict of Tensors produced by the decoder.
    Returns:
      images: the image tensor.
      labels: a dict of Tensors that contains labels.
    """
    image = tf.io.decode_image(decoded_tensors['image/encoded'])
    image.set_shape((None, None, 3))
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(
        image,
        target_width=self._output_size[0],
        target_height=self._output_size[1])  # Final Output Shape
    image = image / 255.  # Normalize
    #label = tf.one_hot(decoded_tensors['image/class/label'], self._num_classes)
    label = decoded_tensors['image/class/label']
    return image, label
