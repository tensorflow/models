"""Priming parser."""

# Import libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from official.vision.beta.dataloaders import parser

class Parser(parser.Parser):
    """Parser to parse an image and its annotations into a dictionary of tensors."""
    def __init__(self,
                 output_size,
                 num_classes,
                 aug_rand_zoom=True,
                 scale=[128, 448],
                 seed=10,
                 dtype='float32'):
        """Initializes parameters for parsing annotations in the dataset.
        Args:
            output_size: `Tensor` or `list` for [height, width] of output image. The
                output_size should be divided by the largest feature stride 2^max_level.
            num_classes: `float`, number of classes.
            aug_rand_zoom: `bool`, if True, augment training with random
                zoom.
            scale: 'list', `Tensor` or `list` for [low, high] of the bounds of the random
                scale.
            seed: an `int` for the seed used by tf.random
        """
        self._output_size = output_size
        self._aug_rand_zoom = aug_rand_zoom
        self._num_classes = num_classes
        self._scale = scale
        self._seed = seed
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
        image = tf.cast(image, tf.float32)
        w = tf.cast(tf.shape(image)[0], tf.float32)
        h = tf.cast(tf.shape(image)[1], tf.int32)

        if self._aug_rand_zoom:
            scale = tf.random.uniform([],
                                      minval=self._scale[0],
                                      maxval=self._scale[1],
                                      seed=self._seed,
                                      dtype=tf.int32)
            image = tf.image.resize_with_crop_or_pad(image,
                                                     target_height=scale,
                                                     target_width=scale)

        image = tf.image.resize_with_pad(image,
                                         target_width=self._output_size[0],
                                         target_height=self._output_size[1])

        image = tf.image.convert_image_dtype(image / 255, self._dtype)

        label = tf.one_hot(decoded_tensors['image/class/label'],
                           self._num_classes)
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
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_pad(
            image,
            target_width=self._output_size[0],
            target_height=self._output_size[1])  # Final Output Shape
        image = image / 255.  # Normalize

        label = tf.one_hot(decoded_tensors['image/class/label'],
                           self._num_classes)
        return image, label
