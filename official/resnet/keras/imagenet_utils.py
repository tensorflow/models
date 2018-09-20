"""Utilities for ImageNet data preprocessing & prediction decoding.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import warnings
import numpy as np


CLASS_INDEX = None
CLASS_INDEX_PATH = ('https://s3.amazonaws.com/deep-learning-models/'
                    'image-models/imagenet_class_index.json')

# Global tensor of imagenet mean for preprocessing symbolic inputs
_IMAGENET_MEAN = None

def get_submodules_from_kwargs(kwargs):
  backend = kwargs.get('backend', _KERAS_BACKEND)
  layers = kwargs.get('layers', _KERAS_LAYERS)
  models = kwargs.get('models', _KERAS_MODELS)
  utils = kwargs.get('utils', _KERAS_UTILS)
  for key in kwargs.keys():
    if key not in ['backend', 'layers', 'models', 'utils']:
      raise TypeError('Invalid keyword argument: %s', key)
  return backend, layers, models, utils

def _preprocess_numpy_input(x, data_format, mode, **kwargs):
  """Preprocesses a Numpy array encoding a batch of images.
  # Arguments
      x: Input array, 3D or 4D.
      data_format: Data format of the image array.
      mode: One of "caffe", "tf" or "torch".
          - caffe: will convert the images from RGB to BGR,
              then will zero-center each color channel with
              respect to the ImageNet dataset,
              without scaling.
          - tf: will scale pixels between -1 and 1,
              sample-wise.
          - torch: will scale pixels between 0 and 1 and then
              will normalize each channel with respect to the
              ImageNet dataset.
  # Returns
      Preprocessed Numpy array.
  """
  backend, _, _, _ = get_submodules_from_kwargs(kwargs)
  if not issubclass(x.dtype.type, np.floating):
    x = x.astype(backend.floatx(), copy=False)

  if mode == 'tf':
    x /= 127.5
    x -= 1.
    return x

  if mode == 'torch':
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  else:
    if data_format == 'channels_first':
      # 'RGB'->'BGR'
      if x.ndim == 3:
        x = x[::-1, ...]
      else:
        x = x[:, ::-1, ...]
    else:
      # 'RGB'->'BGR'
      x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None

  # Zero-center by mean pixel
  if data_format == 'channels_first':
    if x.ndim == 3:
      x[0, :, :] -= mean[0]
      x[1, :, :] -= mean[1]
      x[2, :, :] -= mean[2]
      if std is not None:
        x[0, :, :] /= std[0]
        x[1, :, :] /= std[1]
        x[2, :, :] /= std[2]
    else:
      x[:, 0, :, :] -= mean[0]
      x[:, 1, :, :] -= mean[1]
      x[:, 2, :, :] -= mean[2]
      if std is not None:
        x[:, 0, :, :] /= std[0]
        x[:, 1, :, :] /= std[1]
        x[:, 2, :, :] /= std[2]
  else:
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
      x[..., 0] /= std[0]
      x[..., 1] /= std[1]
      x[..., 2] /= std[2]
  return x


def _preprocess_symbolic_input(x, data_format, mode, **kwargs):
  """Preprocesses a tensor encoding a batch of images.
  # Arguments
      x: Input tensor, 3D or 4D.
      data_format: Data format of the image tensor.
      mode: One of "caffe", "tf" or "torch".
          - caffe: will convert the images from RGB to BGR,
              then will zero-center each color channel with
              respect to the ImageNet dataset,
              without scaling.
          - tf: will scale pixels between -1 and 1,
              sample-wise.
          - torch: will scale pixels between 0 and 1 and then
              will normalize each channel with respect to the
              ImageNet dataset.
  # Returns
      Preprocessed tensor.
  """
  global _IMAGENET_MEAN

  backend, _, _, _ = get_submodules_from_kwargs(kwargs)

  if mode == 'tf':
    x /= 127.5
    x -= 1.
    return x

  if mode == 'torch':
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  else:
    if data_format == 'channels_first':
      # 'RGB'->'BGR'
      if backend.ndim(x) == 3:
        x = x[::-1, ...]
      else:
        x = x[:, ::-1, ...]
    else:
      # 'RGB'->'BGR'
      x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None

  if _IMAGENET_MEAN is None:
    _IMAGENET_MEAN = backend.constant(-np.array(mean))

  # Zero-center by mean pixel
  if backend.dtype(x) != backend.dtype(_IMAGENET_MEAN):
    x = backend.bias_add(
        x, backend.cast(_IMAGENET_MEAN, backend.dtype(x)),
        data_format=data_format)
  else:
    x = backend.bias_add(x, _IMAGENET_MEAN, data_format)
  if std is not None:
    x /= std
  return x


def preprocess_input(x, data_format=None, mode='caffe', **kwargs):
  """Preprocesses a tensor or Numpy array encoding a batch of images.
  # Arguments
      x: Input Numpy or symbolic tensor, 3D or 4D.
          The preprocessed data is written over the input data
          if the data types are compatible. To avoid this
          behaviour, `numpy.copy(x)` can be used.
      data_format: Data format of the image tensor/array.
      mode: One of "caffe", "tf" or "torch".
          - caffe: will convert the images from RGB to BGR,
              then will zero-center each color channel with
              respect to the ImageNet dataset,
              without scaling.
          - tf: will scale pixels between -1 and 1,
              sample-wise.
          - torch: will scale pixels between 0 and 1 and then
              will normalize each channel with respect to the
              ImageNet dataset.
  # Returns
      Preprocessed tensor or Numpy array.
  # Raises
      ValueError: In case of unknown `data_format` argument.
  """
  backend, _, _, _ = get_submodules_from_kwargs(kwargs)

  if data_format is None:
    data_format = backend.image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))

  if isinstance(x, np.ndarray):
    return _preprocess_numpy_input(x, data_format=data_format,
                                   mode=mode, **kwargs)
  else:
    return _preprocess_symbolic_input(x, data_format=data_format,
                                      mode=mode, **kwargs)


def decode_predictions(preds, top=5, **kwargs):
  """Decodes the prediction of an ImageNet model.
  # Arguments
      preds: Numpy tensor encoding a batch of predictions.
      top: Integer, how many top-guesses to return.
  # Returns
      A list of lists of top class prediction tuples
      `(class_name, class_description, score)`.
      One list of tuples per sample in batch input.
  # Raises
      ValueError: In case of invalid shape of the `pred` array
          (must be 2D).
  """
  global CLASS_INDEX

  backend, _, _, keras_utils = get_submodules_from_kwargs(kwargs)

  if len(preds.shape) != 2 or preds.shape[1] != 1000:
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
  if CLASS_INDEX is None:
    fpath = keras_utils.get_file(
        'imagenet_class_index.json',
        CLASS_INDEX_PATH,
        cache_subdir='models',
        file_hash='c2c37ea517e94d9795004a39431a14cb')
    with open(fpath) as f:
      CLASS_INDEX = json.load(f)
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results


def _obtain_input_shape(input_shape,
    default_size,
    min_size,
    data_format,
    require_flatten,
    weights=None):
  """Internal utility to compute/validate a model's input shape.
  # Arguments
      input_shape: Either None (will return the default network input shape),
          or a user-provided shape to be validated.
      default_size: Default input width/height for the model.
      min_size: Minimum input width/height accepted by the model.
      data_format: Image data format to use.
      require_flatten: Whether the model is expected to
          be linked to a classifier via a Flatten layer.
      weights: One of `None` (random initialization)
          or 'imagenet' (pre-training on ImageNet).
          If weights='imagenet' input channels must be equal to 3.
  # Returns
      An integer shape tuple (may include None entries).
  # Raises
      ValueError: In case of invalid argument values.
  """
  if weights != 'imagenet' and input_shape and len(input_shape) == 3:
    if data_format == 'channels_first':
      if input_shape[0] not in {1, 3}:
        warnings.warn(
            'This model usually expects 1 or 3 input channels. '
            'However, it was passed an input_shape with ' +
            str(input_shape[0]) + ' input channels.')
      default_shape = (input_shape[0], default_size, default_size)
    else:
      if input_shape[-1] not in {1, 3}:
        warnings.warn(
            'This model usually expects 1 or 3 input channels. '
            'However, it was passed an input_shape with ' +
            str(input_shape[-1]) + ' input channels.')
      default_shape = (default_size, default_size, input_shape[-1])
  else:
    if data_format == 'channels_first':
      default_shape = (3, default_size, default_size)
    else:
      default_shape = (default_size, default_size, 3)
  if weights == 'imagenet' and require_flatten:
    if input_shape is not None:
      if input_shape != default_shape:
        raise ValueError('When setting`include_top=True` '
                         'and loading `imagenet` weights, '
                         '`input_shape` should be ' +
                         str(default_shape) + '.')
    return default_shape
  if input_shape:
    if data_format == 'channels_first':
      if input_shape is not None:
        if len(input_shape) != 3:
          raise ValueError(
              '`input_shape` must be a tuple of three integers.')
        if input_shape[0] != 3 and weights == 'imagenet':
          raise ValueError('The input must have 3 channels; got '
                           '`input_shape=' + str(input_shape) + '`')
        if ((input_shape[1] is not None and input_shape[1] < min_size) or
            (input_shape[2] is not None and input_shape[2] < min_size)):
          raise ValueError('Input size must be at least ' +
                           str(min_size) + 'x' + str(min_size) +
                           '; got `input_shape=' +
                           str(input_shape) + '`')
    else:
      if input_shape is not None:
        if len(input_shape) != 3:
          raise ValueError(
              '`input_shape` must be a tuple of three integers.')
        if input_shape[-1] != 3 and weights == 'imagenet':
          raise ValueError('The input must have 3 channels; got '
                           '`input_shape=' + str(input_shape) + '`')
        if ((input_shape[0] is not None and input_shape[0] < min_size) or
            (input_shape[1] is not None and input_shape[1] < min_size)):
          raise ValueError('Input size must be at least ' +
                           str(min_size) + 'x' + str(min_size) +
                           '; got `input_shape=' +
                           str(input_shape) + '`')
  else:
    if require_flatten:
      input_shape = default_shape
    else:
      if data_format == 'channels_first':
        input_shape = (3, None, None)
      else:
        input_shape = (None, None, 3)
  if require_flatten:
    if None in input_shape:
      raise ValueError('If `include_top` is True, '
                       'you should specify a static `input_shape`. '
                       'Got `input_shape=' + str(input_shape) + '`')
  return input_shape
