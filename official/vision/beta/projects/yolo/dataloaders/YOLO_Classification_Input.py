"""Classification parser."""

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
                 aug_rand_saturation=True,
                 aug_rand_brightness=True,
                 aug_rand_zoom=True,
                 aug_rand_rotate=True,
                 aug_rand_hue=True,
                 aug_rand_aspect=True,
                 scale=[128, 448],
                 seed=10,
                 dtype='float32'):
        """Initializes parameters for parsing annotations in the dataset.
        Args:
            output_size: `Tensor` or `list` for [height, width] of output image. The
                output_size should be divided by the largest feature stride 2^max_level.
            num_classes: `float`, number of classes.
            aug_rand_saturation: `bool`, if True, augment training with random
                saturation.
            aug_rand_brightness: `bool`, if True, augment training with random
                brightness.
            aug_rand_zoom: `bool`, if True, augment training with random
                zoom.
            aug_rand_rotate: `bool`, if True, augment training with random
                rotate.
            aug_rand_hue: `bool`, if True, augment training with random
                hue.
            aug_rand_aspect: `bool`, if True, augment training with random
                aspect.
            scale: 'list', `Tensor` or `list` for [low, high] of the bounds of the random
                scale.
            seed: an `int` for the seed used by tf.random
        """
        self._output_size = output_size
        self._aug_rand_saturation = aug_rand_saturation
        self._aug_rand_brightness = aug_rand_brightness
        self._aug_rand_zoom = aug_rand_zoom
        self._aug_rand_rotate = aug_rand_rotate
        self._aug_rand_hue = aug_rand_hue
        self._num_classes = num_classes
        self._aug_rand_aspect = aug_rand_aspect
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
        image.set_shape((None, None, 3))
        image = tf.cast(image, tf.float32)
        w = tf.cast(tf.shape(image)[0], tf.float32)
        h = tf.cast(tf.shape(image)[1], tf.int32)

        if self._aug_rand_aspect:
            aspect = tf.random.uniform(
                [], minval=3, maxval=5, seed=self._seed, dtype=tf.float32) / 4.
            nh = tf.cast(w / aspect, dtype=tf.int32)
            nw = tf.cast(w, dtype=tf.int32)
            image = tf.image.resize(image, size=(nw, nh))

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

        if self._aug_rand_rotate:
            deg = tf.random.uniform([],
                                    minval=-30,
                                    maxval=30,
                                    seed=self._seed,
                                    dtype=tf.float32)
            deg = deg * 3.14 / 180.
            deg.set_shape(())
            image = tfa.image.rotate(image, deg, interpolation="NEAREST")

        if self._aug_rand_brightness:
            image = tf.image.random_brightness(image=image, max_delta=.75)

        if self._aug_rand_saturation:
            image = tf.image.random_saturation(image=image,
                                               lower=0.75,
                                               upper=1.25)

        if self._aug_rand_hue:
            image = tf.image.random_hue(image=image, max_delta=.1)

        image = image / 255
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, self._dtype)

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
        image.set_shape((None, None, 3))
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_pad(
            image,
            target_width=self._output_size[0],
            target_height=self._output_size[1])  # Final Output Shape
        image = image / 255.  # Normalize
        label = tf.one_hot(decoded_tensors['image/class/label'],
                           self._num_classes)

        return image, label
