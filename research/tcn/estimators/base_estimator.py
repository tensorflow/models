# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Base estimator defining TCN training, test, and inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os
import numpy as np
import numpy as np
import data_providers
import preprocessing
from utils import util
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.training import session_run_hook

tf.app.flags.DEFINE_integer(
    'tf_random_seed', 0, 'Random seed.')
FLAGS = tf.app.flags.FLAGS


class InitFromPretrainedCheckpointHook(session_run_hook.SessionRunHook):
  """Hook that can init graph from a pretrained checkpoint."""

  def __init__(self, pretrained_checkpoint_dir):
    """Initializes a `InitFromPretrainedCheckpointHook`.

    Args:
      pretrained_checkpoint_dir: The dir of pretrained checkpoint.

    Raises:
      ValueError: If pretrained_checkpoint_dir is invalid.
    """
    if pretrained_checkpoint_dir is None:
      raise ValueError('pretrained_checkpoint_dir must be specified.')
    self._pretrained_checkpoint_dir = pretrained_checkpoint_dir

  def begin(self):
    checkpoint_reader = tf.contrib.framework.load_checkpoint(
        self._pretrained_checkpoint_dir)
    variable_shape_map = checkpoint_reader.get_variable_to_shape_map()

    exclude_scopes = 'logits/,final_layer/,aux_'
    # Skip restoring global_step as to run fine tuning from step=0.
    exclusions = ['global_step']
    if exclude_scopes:
      exclusions.extend([scope.strip() for scope in exclude_scopes.split(',')])

    variable_to_restore = tf.contrib.framework.get_model_variables()

    # Variable filtering by given exclude_scopes.
    filtered_variables_to_restore = {}
    for v in variable_to_restore:
      excluded = False
      for exclusion in exclusions:
        if v.name.startswith(exclusion):
          excluded = True
          break
      if not excluded:
        var_name = v.name.split(':')[0]
        filtered_variables_to_restore[var_name] = v

    # Final filter by checking shape matching and skipping variables that
    # are not in the checkpoint.
    final_variables_to_restore = {}
    for var_name, var_tensor in filtered_variables_to_restore.iteritems():
      if var_name not in variable_shape_map:
        # Try moving average version of variable.
        var_name = os.path.join(var_name, 'ExponentialMovingAverage')
        if var_name not in variable_shape_map:
          tf.logging.info(
              'Skip init [%s] because it is not in ckpt.', var_name)
          # Skip variables not in the checkpoint.
          continue

      if not var_tensor.get_shape().is_compatible_with(
          variable_shape_map[var_name]):
        # Skip init variable from ckpt if shape dismatch.
        tf.logging.info(
            'Skip init [%s] from [%s] in ckpt because shape dismatch: %s vs %s',
            var_tensor.name, var_name,
            var_tensor.get_shape(), variable_shape_map[var_name])
        continue

      tf.logging.info('Init %s from %s in ckpt' % (var_tensor, var_name))
      final_variables_to_restore[var_name] = var_tensor

    self._init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
        self._pretrained_checkpoint_dir,
        final_variables_to_restore)

  def after_create_session(self, session, coord):
    tf.logging.info('Restoring InceptionV3 weights.')
    self._init_fn(session)
    tf.logging.info('Done restoring InceptionV3 weights.')


class BaseEstimator(object):
  """Abstract TCN base estimator class."""
  __metaclass__ = ABCMeta

  def __init__(self, config, logdir):
    """Constructor.

    Args:
      config: A Luatable-like T object holding training config.
      logdir: String, a directory where checkpoints and summaries are written.
    """
    self._config = config
    self._logdir = logdir

  @abstractmethod
  def construct_input_fn(self, records, is_training):
    """Builds an estimator input_fn.

    The input_fn is used to pass feature and target data to the train,
    evaluate, and predict methods of the Estimator.

    Method to be overridden by implementations.

    Args:
      records: A list of Strings, paths to TFRecords with image data.
      is_training: Boolean, whether or not we're training.

    Returns:
      Function, that has signature of ()->(dict of features, target).
        features is a dict mapping feature names to `Tensors`
        containing the corresponding feature data (typically, just a single
        key/value pair 'raw_data' -> image `Tensor` for TCN.
        labels is a 1-D int32 `Tensor` holding labels.
    """
    pass

  def preprocess_data(self, images, is_training):
    """Preprocesses raw images for either training or inference.

    Args:
      images: A 4-D float32 `Tensor` holding images to preprocess.
      is_training: Boolean, whether or not we're in training.

    Returns:
      data_preprocessed: data after the preprocessor.
    """
    config = self._config
    height = config.data.height
    width = config.data.width
    min_scale = config.data.augmentation.minscale
    max_scale = config.data.augmentation.maxscale
    p_scale_up = config.data.augmentation.proportion_scaled_up
    aug_color = config.data.augmentation.color
    fast_mode = config.data.augmentation.fast_mode
    crop_strategy = config.data.preprocessing.eval_cropping
    preprocessed_images = preprocessing.preprocess_images(
        images, is_training, height, width,
        min_scale, max_scale, p_scale_up,
        aug_color=aug_color, fast_mode=fast_mode,
        crop_strategy=crop_strategy)
    return preprocessed_images

  @abstractmethod
  def forward(self, images, is_training, reuse=False):
    """Defines the forward pass that converts batch images to embeddings.

    Method to be overridden by implementations.

    Args:
      images: A 4-D float32 `Tensor` holding images to be embedded.
      is_training: Boolean, whether or not we're in training mode.
      reuse: Boolean, whether or not to reuse embedder.
    Returns:
      embeddings: A 2-D float32 `Tensor` holding embedded images.
    """
    pass

  @abstractmethod
  def define_loss(self, embeddings, labels, is_training):
    """Defines the loss function on the embedding vectors.

    Method to be overridden by implementations.

    Args:
      embeddings: A 2-D float32 `Tensor` holding embedded images.
      labels: A 1-D int32 `Tensor` holding problem labels.
      is_training: Boolean, whether or not we're in training mode.

    Returns:
      loss: tf.float32 scalar.
    """
    pass

  @abstractmethod
  def define_eval_metric_ops(self):
    """Defines the dictionary of eval metric tensors.

    Method to be overridden by implementations.

    Returns:
      eval_metric_ops:  A dict of name/value pairs specifying the
        metrics that will be calculated when the model runs in EVAL mode.
    """
    pass

  def get_train_op(self, loss):
    """Creates a training op.

    Args:
      loss: A float32 `Tensor` representing the total training loss.
    Returns:
      train_op: A slim.learning.create_train_op train_op.
    Raises:
      ValueError: If specified optimizer isn't supported.
    """
    # Get variables to train (defined in subclass).
    assert self.variables_to_train

    # Define a learning rate schedule.
    decay_steps = self._config.learning.decay_steps
    decay_factor = self._config.learning.decay_factor
    learning_rate = float(self._config.learning.learning_rate)

    # Define a learning rate schedule.
    global_step = slim.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        learning_rate,
        global_step,
        decay_steps,
        decay_factor,
        staircase=True)

    # Create an optimizer.
    opt_type = self._config.learning.optimizer
    if opt_type == 'adam':
      opt = tf.train.AdamOptimizer(learning_rate)
    elif opt_type == 'momentum':
      opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif opt_type == 'rmsprop':
      opt = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9,
                                      epsilon=1.0, decay=0.9)
    else:
      raise ValueError('Unsupported optimizer %s' % opt_type)

    if self._config.use_tpu:
      opt = tpu_optimizer.CrossShardOptimizer(opt)

    # Create a training op.
    # train_op = opt.minimize(loss, var_list=self.variables_to_train)
    # Create a training op.
    train_op = slim.learning.create_train_op(
        loss,
        optimizer=opt,
        variables_to_train=self.variables_to_train,
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    return train_op

  def _get_model_fn(self):
    """Defines behavior for training, evaluation, and inference (prediction).

    Returns:
      `model_fn` for `Estimator`.
    """
    # pylint: disable=unused-argument
    def model_fn(features, labels, mode, params):
      """Build the model based on features, labels, and mode.

      Args:
        features: Dict, strings to `Tensor` input data, returned by the
          input_fn.
        labels: The labels Tensor returned by the input_fn.
        mode: A string indicating the mode. This will be either
          tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.PREDICT,
          or tf.estimator.ModeKeys.EVAL.
        params: A dict holding training parameters, passed in during TPU
          training.

      Returns:
        A tf.estimator.EstimatorSpec specifying train/test/inference behavior.
      """
      is_training = mode == tf.estimator.ModeKeys.TRAIN

      # Get preprocessed images from the features dict.
      batch_preprocessed = features['batch_preprocessed']

      # Do a forward pass to embed data.
      batch_encoded = self.forward(batch_preprocessed, is_training)

      # Optionally set the pretrained initialization function.
      initializer_fn = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        initializer_fn = self.pretrained_init_fn

      # If we're training or evaluating, define total loss.
      total_loss = None
      if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss = self.define_loss(batch_encoded, labels, is_training)
        tf.losses.add_loss(loss)
        total_loss = tf.losses.get_total_loss()

      # If we're training, define a train op.
      train_op = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = self.get_train_op(total_loss)

      # If we're doing inference, set the output to be the embedded images.
      predictions_dict = None
      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions_dict = {'embeddings': batch_encoded}
        # Pass through additional metadata stored in features.
        for k, v in features.iteritems():
          predictions_dict[k] = v

      # If we're evaluating, define some eval metrics.
      eval_metric_ops = None
      if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = self.define_eval_metric_ops()

      # Define training scaffold to load pretrained weights.
      num_checkpoint_to_keep = self._config.logging.checkpoint.num_to_keep
      saver = tf.train.Saver(
          max_to_keep=num_checkpoint_to_keep)

      if is_training and self._config.use_tpu:
        # TPU doesn't have a scaffold option at the moment, so initialize
        # pretrained weights using a custom train_hook instead.
        return tpu_estimator.TPUEstimatorSpec(
            mode,
            loss=total_loss,
            eval_metrics=None,
            train_op=train_op,
            predictions=predictions_dict)
      else:
        # Build a scaffold to initialize pretrained weights.
        scaffold = tf.train.Scaffold(
            init_fn=initializer_fn,
            saver=saver,
            summary_op=None)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions_dict,
            loss=total_loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            scaffold=scaffold)
    return model_fn

  def train(self):
    """Runs training."""
    # Get a list of training tfrecords.
    config = self._config
    training_dir = config.data.training
    training_records = util.GetFilesRecursively(training_dir)

    # Define batch size.
    self._batch_size = config.data.batch_size

    # Create a subclass-defined training input function.
    train_input_fn = self.construct_input_fn(
        training_records, is_training=True)

    # Create the estimator.
    estimator = self._build_estimator(is_training=True)

    train_hooks = None
    if config.use_tpu:
      # TPU training initializes pretrained weights using a custom train hook.
      train_hooks = []
      if tf.train.latest_checkpoint(self._logdir) is None:
        train_hooks.append(
            InitFromPretrainedCheckpointHook(
                config[config.embedder_strategy].pretrained_checkpoint))

    # Run training.
    estimator.train(input_fn=train_input_fn, hooks=train_hooks,
                    steps=config.learning.max_step)

  def _build_estimator(self, is_training):
    """Returns an Estimator object.

    Args:
      is_training: Boolean, whether or not we're in training mode.

    Returns:
      A tf.estimator.Estimator.
    """
    config = self._config
    save_checkpoints_steps = config.logging.checkpoint.save_checkpoints_steps
    keep_checkpoint_max = self._config.logging.checkpoint.num_to_keep
    if is_training and config.use_tpu:
      iterations = config.tpu.iterations
      num_shards = config.tpu.num_shards
      run_config = tpu_config.RunConfig(
          save_checkpoints_secs=None,
          save_checkpoints_steps=save_checkpoints_steps,
          keep_checkpoint_max=keep_checkpoint_max,
          master=FLAGS.master,
          evaluation_master=FLAGS.master,
          model_dir=self._logdir,
          tpu_config=tpu_config.TPUConfig(
              iterations_per_loop=iterations,
              num_shards=num_shards,
              per_host_input_for_training=num_shards <= 8),
          tf_random_seed=FLAGS.tf_random_seed)

      batch_size = config.data.batch_size
      return tpu_estimator.TPUEstimator(
          model_fn=self._get_model_fn(),
          config=run_config,
          use_tpu=True,
          train_batch_size=batch_size,
          eval_batch_size=batch_size)
    else:
      run_config = tf.estimator.RunConfig().replace(
          model_dir=self._logdir,
          save_checkpoints_steps=save_checkpoints_steps,
          keep_checkpoint_max=keep_checkpoint_max,
          tf_random_seed=FLAGS.tf_random_seed)
      return tf.estimator.Estimator(
          model_fn=self._get_model_fn(),
          config=run_config)

  def evaluate(self):
    """Runs `Estimator` validation.
    """
    config = self._config

    # Get a list of validation tfrecords.
    validation_dir = config.data.validation
    validation_records = util.GetFilesRecursively(validation_dir)

    # Define batch size.
    self._batch_size = config.data.batch_size

    # Create a subclass-defined training input function.
    validation_input_fn = self.construct_input_fn(
        validation_records, False)

    # Create the estimator.
    estimator = self._build_estimator(is_training=False)

    # Run validation.
    eval_batch_size = config.data.batch_size
    num_eval_samples = config.val.num_eval_samples
    num_eval_batches = int(num_eval_samples / eval_batch_size)
    estimator.evaluate(input_fn=validation_input_fn, steps=num_eval_batches)

  def inference(
      self, inference_input, checkpoint_path, batch_size=None, **kwargs):
    """Defines 3 of modes of inference.

    Inputs:
    * Mode 1: Input is an input_fn.
    * Mode 2: Input is a TFRecord (or list of TFRecords).
    * Mode 3: Input is a numpy array holding an image (or array of images).

    Outputs:
    * Mode 1: this returns an iterator over embeddings and additional
      metadata. See
      https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#predict
      for details.
    * Mode 2: Returns an iterator over tuples of
      (embeddings, raw_image_strings, sequence_name), where embeddings is a
      2-D float32 numpy array holding [sequence_size, embedding_size] image
      embeddings, raw_image_strings is a 1-D string numpy array holding
      [sequence_size] jpeg-encoded image strings, and sequence_name is a
      string holding the name of the embedded sequence.
    * Mode 3: Returns a tuple of (embeddings, raw_image_strings), where
      embeddings is a 2-D float32 numpy array holding
      [batch_size, embedding_size] image embeddings, raw_image_strings is a
      1-D string numpy array holding [batch_size] jpeg-encoded image strings.

    Args:
      inference_input: This can be a tf.Estimator input_fn, a TFRecord path,
        a list of TFRecord paths, a numpy image, or an array of numpy images.
      checkpoint_path: String, path to the checkpoint to restore for inference.
      batch_size: Int, the size of the batch to use for inference.
      **kwargs: Additional keyword arguments, depending on the mode.
        See _input_fn_inference, _tfrecord_inference, and _np_inference.
    Returns:
      inference_output: Inference output depending on mode, see above for
        details.
    Raises:
      ValueError: If inference_input isn't a tf.Estimator input_fn,
        a TFRecord path, a list of TFRecord paths, or a numpy array,
    """
    # Mode 1: input is a callable tf.Estimator input_fn.
    if callable(inference_input):
      return self._input_fn_inference(
          input_fn=inference_input, checkpoint_path=checkpoint_path, **kwargs)
    # Mode 2: Input is a TFRecord path (or list of TFRecord paths).
    elif util.is_tfrecord_input(inference_input):
      return self._tfrecord_inference(
          records=inference_input, checkpoint_path=checkpoint_path,
          batch_size=batch_size, **kwargs)
    # Mode 3: Input is a numpy array of raw images.
    elif util.is_np_array(inference_input):
      return self._np_inference(
          np_images=inference_input, checkpoint_path=checkpoint_path, **kwargs)
    else:
      raise ValueError(
          'inference input must be a tf.Estimator input_fn, a TFRecord path,'
          'a list of TFRecord paths, or a numpy array. Got: %s' % str(type(
              inference_input)))

  def _input_fn_inference(self, input_fn, checkpoint_path, predict_keys=None):
    """Mode 1: tf.Estimator inference.

    Args:
      input_fn: Function, that has signature of ()->(dict of features, None).
        This is a function called by the estimator to get input tensors (stored
        in the features dict) to do inference over.
      checkpoint_path: String, path to a specific checkpoint to restore.
      predict_keys: List of strings, the keys of the `Tensors` in the features
        dict (returned by the input_fn) to evaluate during inference.
    Returns:
      predictions: An Iterator, yielding evaluated values of `Tensors`
        specified in `predict_keys`.
    """
    # Create the estimator.
    estimator = self._build_estimator(is_training=False)

    # Create an iterator of predicted embeddings.
    predictions = estimator.predict(input_fn=input_fn,
                                    checkpoint_path=checkpoint_path,
                                    predict_keys=predict_keys)
    return predictions

  def _tfrecord_inference(self, records, checkpoint_path, batch_size,
                          num_sequences=-1, reuse=False):
    """Mode 2: TFRecord inference.

    Args:
      records: List of strings, paths to TFRecords.
      checkpoint_path: String, path to a specific checkpoint to restore.
      batch_size: Int, size of inference batch.
      num_sequences: Int, number of sequences to embed. If -1,
        embed everything.
      reuse: Boolean, whether or not to reuse embedder weights.
    Yields:
      (embeddings, raw_image_strings, sequence_name):
        embeddings is a 2-D float32 numpy array holding
        [sequence_size, embedding_size] image embeddings.
        raw_image_strings is a 1-D string numpy array holding
        [sequence_size] jpeg-encoded image strings.
        sequence_name is a string holding the name of the embedded sequence.
    """
    tf.reset_default_graph()
    if not isinstance(records, list):
      records = list(records)

    # Map the list of tfrecords to a dataset of preprocessed images.
    num_views = self._config.data.num_views
    (views, task, seq_len) = data_providers.full_sequence_provider(
        records, num_views)
    tensor_dict = {
        'raw_image_strings': views,
        'task': task,
        'seq_len': seq_len
    }

    # Create a preprocess function over raw image string placeholders.
    image_str_placeholder = tf.placeholder(tf.string, shape=[None])
    decoded = preprocessing.decode_images(image_str_placeholder)
    decoded.set_shape([batch_size, None, None, 3])
    preprocessed = self.preprocess_data(decoded, is_training=False)

    # Create an inference graph over preprocessed images.
    embeddings = self.forward(preprocessed, is_training=False, reuse=reuse)

    # Create a saver to restore model variables.
    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(tf.all_variables())

    # Create a session and restore model variables.
    with tf.train.MonitoredSession() as sess:
      saver.restore(sess, checkpoint_path)
      cnt = 0
      # If num_sequences is specified, embed that many sequences, else embed
      # everything.
      try:
        while cnt < num_sequences if num_sequences != -1 else True:
          # Get a preprocessed image sequence.
          np_data = sess.run(tensor_dict)
          np_raw_images = np_data['raw_image_strings']
          np_seq_len = np_data['seq_len']
          np_task = np_data['task']

          # Embed each view.
          embedding_size = self._config.embedding_size
          view_embeddings = [
              np.zeros((0, embedding_size)) for _ in range(num_views)]
          for view_index in range(num_views):
            view_raw = np_raw_images[view_index]
            # Embed the full sequence.
            t = 0
            while t < np_seq_len:
              # Decode and preprocess the batch of image strings.
              embeddings_np = sess.run(
                  embeddings, feed_dict={
                      image_str_placeholder: view_raw[t:t+batch_size]})
              view_embeddings[view_index] = np.append(
                  view_embeddings[view_index], embeddings_np, axis=0)
              tf.logging.info('Embedded %d images for task %s' % (t, np_task))
              t += batch_size

          # Done embedding for all views.
          view_raw_images = np_data['raw_image_strings']
          yield (view_embeddings, view_raw_images, np_task)
          cnt += 1
      except tf.errors.OutOfRangeError:
        tf.logging.info('Done embedding entire dataset.')

  def _np_inference(self, np_images, checkpoint_path):
    """Mode 3: Call this repeatedly to do inference over numpy images.

    This mode is for when we we want to do real-time inference over
    some stream of images (represented as numpy arrays).

    Args:
      np_images: A float32 numpy array holding images to embed.
      checkpoint_path: String, path to a specific checkpoint to restore.
    Returns:
      (embeddings, raw_image_strings):
        embeddings is a 2-D float32 numpy array holding
        [inferred batch_size, embedding_size] image embeddings.
        raw_image_strings is a 1-D string numpy array holding
        [inferred batch_size] jpeg-encoded image strings.
    """
    if isinstance(np_images, list):
      np_images = np.asarray(np_images)
    # Add a batch dimension if only 3-dimensional.
    if len(np_images.shape) == 3:
      np_images = np.expand_dims(np_images, axis=0)

    # If np_images are in the range [0,255], convert to [0,1].
    assert np.min(np_images) >= 0.
    if (np.min(np_images), np.max(np_images)) == (0, 255):
      np_images = np_images.astype(np.float32) / 255.
      assert (np.min(np_images), np.max(np_images)) == (0., 1.)

    # If this is the first pass, set up inference graph.
    if not hasattr(self, '_np_inf_tensor_dict'):
      self._setup_np_inference(np_images, checkpoint_path)

    # Convert np_images to embeddings.
    np_tensor_dict = self._sess.run(self._np_inf_tensor_dict, feed_dict={
        self._image_placeholder: np_images
    })
    return np_tensor_dict['embeddings'], np_tensor_dict['raw_image_strings']

  def _setup_np_inference(self, np_images, checkpoint_path):
    """Sets up and restores inference graph, creates and caches a Session."""
    tf.logging.info('Restoring model weights.')

    # Define inference over an image placeholder.
    _, height, width, _ = np.shape(np_images)
    image_placeholder = tf.placeholder(
        tf.float32, shape=(None, height, width, 3))

    # Preprocess batch.
    preprocessed = self.preprocess_data(image_placeholder, is_training=False)

    # Unscale and jpeg encode preprocessed images for display purposes.
    im_strings = preprocessing.unscale_jpeg_encode(preprocessed)

    # Do forward pass to get embeddings.
    embeddings = self.forward(preprocessed, is_training=False)

    # Create a saver to restore model variables.
    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(tf.all_variables())

    self._image_placeholder = image_placeholder
    self._batch_encoded = embeddings

    self._np_inf_tensor_dict = {
        'embeddings': embeddings,
        'raw_image_strings': im_strings,
    }

    # Create a session and restore model variables.
    self._sess = tf.Session()
    saver.restore(self._sess, checkpoint_path)
