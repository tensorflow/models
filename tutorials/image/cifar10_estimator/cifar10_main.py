# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""ResNet model for classifying images from CIFAR-10 dataset.

Support single-host training with one or multiple devices.

ResNet as proposed in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. arXiv:1512.03385

CIFAR-10 as in:
http://www.cs.toronto.edu/~kriz/cifar.html


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cifar10
import cifar10_model

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', '',
                       'The directory where the CIFAR-10 input data is stored.')

tf.flags.DEFINE_string('model_dir', '',
                       'The directory where the model will be stored.')

tf.flags.DEFINE_boolean('is_cpu_ps', False,
                        'If using CPU as the parameter server.')

tf.flags.DEFINE_integer('num_gpus', 1,
                        'The number of gpus used. Uses only CPU if set to 0.')

tf.flags.DEFINE_integer('num_layers', 44, 'The number of layers of the model.')

tf.flags.DEFINE_integer('train_steps', 10000,
                        'The number of steps to use for training.')

tf.flags.DEFINE_integer('train_batch_size', 128, 'Batch size for training.')

tf.flags.DEFINE_integer('eval_batch_size', 100, 'Batch size for validation.')

tf.flags.DEFINE_float('momentum', 0.9, 'Momentum for MomentumOptimizer.')

tf.flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay for convolutions.')

tf.flags.DEFINE_boolean('use_distortion_for_training', True,
                        'If doing image distortion for training.')

# Perf flags
tf.flags.DEFINE_integer('num_intra_threads', 1,
                        """Number of threads to use for intra-op parallelism.
                        If set to 0, the system will pick an appropriate number.
                        The default is 1 since in this example CPU only handles
                        the input pipeline and gradient aggregation (when
                        --is_cpu_ps). Ops that could potentially benefit
                        from intra-op parallelism are scheduled to run on GPUs.
                        """)

tf.flags.DEFINE_integer('num_inter_threads', 0,
                        """Number of threads to use for inter-op
                       parallelism. If set to 0, the system will pick
                       an appropriate number.""")

tf.flags.DEFINE_boolean('force_gpu_compatible', False,
                        """whether to enable force_gpu_compatible in
                        GPU_Options. Check
                        tensorflow/core/protobuf/config.proto#L69
                        for details.""")

# Debugging flags
tf.flags.DEFINE_boolean('log_device_placement', False,
                        'Whether to log device placement.')


# TODO(jamesqin): Replace with fix in b/62239022
class ParamServerDeviceSetter(object):
  """Helper class to assign variables on the least loaded ps-device."""

  def __init__(self, worker_device, ps_devices):
    """Initializer for ParamServerDeviceSetter.

    Args:
      worker_device: the device to use for computer ops.
      ps_devices: a list of devices to use for Variable ops. Each variable is
      assigned to the least loaded device.
    """
    self.ps_devices = ps_devices
    self.worker_device = worker_device
    self.ps_sizes = [0] * len(self.ps_devices)

  def __call__(self, op):
    if op.device:
      return op.device
    if op.type not in ['Variable', 'VariableV2', 'VarHandleOp']:
      return self.worker_device

    device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
    device_name = self.ps_devices[device_index]
    var_size = op.outputs[0].get_shape().num_elements()
    self.ps_sizes[device_index] += var_size

    return device_name


def _create_device_setter(is_cpu_ps, worker):
  """Create device setter object."""
  if is_cpu_ps:
    return tf.train.replica_device_setter(
        worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
  else:
    gpus = ['/gpu:%d' % i for i in range(FLAGS.num_gpus)]
    return ParamServerDeviceSetter(worker, gpus)


def _resnet_model_fn(features, labels, mode):
  """Resnet model body.

  Support single host, one or more GPU training. Parameter distribution can be
  either one of the following scheme.
  1. CPU is the parameter server and manages gradient updates.
  2. Paramters are distributed evenly across all GPUs, and the first GPU
     manages gradient updates.

  Args:
    features: a list of tensors, one for each tower
    labels: a list of tensors, one for each tower
    mode: ModeKeys.TRAIN or EVAL
  Returns:
    A EstimatorSpec object.
  """
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  is_cpu_ps = FLAGS.is_cpu_ps
  num_gpus = FLAGS.num_gpus
  weight_decay = FLAGS.weight_decay
  momentum = FLAGS.momentum

  tower_features = features
  tower_labels = labels
  tower_losses = []
  tower_gradvars = []
  tower_preds = []

  if num_gpus != 0:
    for i in range(num_gpus):
      worker = '/gpu:%d' % i
      device_setter = _create_device_setter(is_cpu_ps, worker)
      with tf.variable_scope('resnet', reuse=bool(i != 0)):
        with tf.name_scope('tower_%d' % i) as name_scope:
          with tf.device(device_setter):
            _tower_fn(is_training, weight_decay, tower_features[i],
                      tower_labels[i], tower_losses, tower_gradvars,
                      tower_preds, False)
            if i == 0:
              # Only trigger batch_norm moving mean and variance update from the
              # 1st tower. Ideally, we should grab the updates from all towers
              # but these stats accumulate extremely fast so we can ignore the
              # other stats from the other towers without significant detriment.
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                             name_scope)
  else:
    with tf.variable_scope('resnet'), tf.device('/cpu:0'):
      with tf.name_scope('tower_cpu') as name_scope:
        _tower_fn(is_training, weight_decay, tower_features[0], tower_labels[0],
                  tower_losses, tower_gradvars, tower_preds, True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

  # Now compute global loss and gradients.
  gradvars = []
  # parameter server here isn't necessarily one server storing the model params.
  # (For gpu-as-ps case, model params are distributed evenly across all gpus.)
  # It's the server that runs the ops to apply global gradient updates.
  ps_device = '/cpu:0' if is_cpu_ps else '/gpu:0'
  with tf.device(ps_device):
    with tf.name_scope('gradient_averaging'):
      loss = tf.reduce_mean(tower_losses)
      for zipped_gradvars in zip(*tower_gradvars):
        # Averaging one var's gradients computed from multiple towers
        var = zipped_gradvars[0][1]
        grads = [gv[0] for gv in zipped_gradvars]
        with tf.device(var.device):
          if len(grads) == 1:
            avg_grad = grads[0]
          else:
            avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
        gradvars.append((avg_grad, var))

    # Suggested learning rate scheduling from
    # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py#L155
    # users could apply other scheduling.
    num_batches_per_epoch = cifar10.Cifar10DataSet.num_examples_per_epoch(
        'train') // FLAGS.train_batch_size
    boundaries = [
        num_batches_per_epoch * x
        for x in np.array([82, 123, 300], dtype=np.int64)
    ]
    staged_lr = [0.1, 0.01, 0.001, 0.0002]
    learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
                                                boundaries, staged_lr)
    # Create a nicely-named tensor for logging
    learning_rate = tf.identity(learning_rate, name='learning_rate')

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=momentum)

    # Create single grouped train op
    train_op = [
        optimizer.apply_gradients(
            gradvars, global_step=tf.train.get_global_step())
    ]
    train_op.extend(update_ops)
    train_op = tf.group(*train_op)

    predictions = {
        'classes':
            tf.concat([p['classes'] for p in tower_preds], axis=0),
        'probabilities':
            tf.concat([p['probabilities'] for p in tower_preds], axis=0)
    }
    stacked_labels = tf.concat(labels, axis=0)
    metrics = {
        'accuracy': tf.metrics.accuracy(stacked_labels, predictions['classes'])
    }

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def _tower_fn(is_training, weight_decay, feature, label, tower_losses,
              tower_gradvars, tower_preds, is_cpu):
  """Build computation tower for each device (CPU or GPU).

  Args:
    is_training: true if is for training graph.
    weight_decay: weight regularization strength, a float.
    feature: a Tensor.
    label: a Tensor.
    tower_losses: a list to be appended with current tower's loss.
    tower_gradvars: a list to be appended with current tower's gradients.
    tower_preds: a list to be appended with current tower's predictions.
    is_cpu: true if build tower on CPU.
  """
  data_format = 'channels_last' if is_cpu else 'channels_first'
  model = cifar10_model.ResNetCifar10(
      FLAGS.num_layers, is_training=is_training, data_format=data_format)
  logits = model.forward_pass(feature, input_data_format='channels_last')
  tower_pred = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits)
  }
  tower_preds.append(tower_pred)

  tower_loss = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=label)
  tower_loss = tf.reduce_mean(tower_loss)
  tower_losses.append(tower_loss)

  model_params = tf.trainable_variables()
  tower_loss += weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in model_params])
  tower_losses.append(tower_loss)

  tower_grad = tf.gradients(tower_loss, model_params)
  tower_gradvars.append(zip(tower_grad, model_params))


def input_fn(subset, num_shards):
  """Create input graph for model.

  Args:
    subset: one of 'train', 'validate' and 'eval'.
    num_shards: num of towers participating in data-parallel training.
  Returns:
    two lists of tensors for features and labels, each of num_shards length.
  """
  dataset = cifar10.Cifar10DataSet(FLAGS.data_dir)
  is_training = (subset == 'train')
  if is_training:
    batch_size = FLAGS.train_batch_size
  else:
    batch_size = FLAGS.eval_batch_size
  with tf.device('/cpu:0'), tf.name_scope('batching'):
    # CPU loads all data from disk since there're only 60k 32*32 RGB images.
    all_images, all_labels = dataset.read_all_data(subset)
    dataset = tf.contrib.data.Dataset.from_tensor_slices(
        (all_images, all_labels))
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)),
        num_threads=2,
        output_buffer_size=batch_size)

    # Image preprocessing.
    def _preprocess(image, label):
      # If GPU is available, NHWC to NCHW transpose is done in ResNetCifar10
      # class, not included in preprocessing.
      return cifar10.Cifar10DataSet.preprocess(
          image, is_training, FLAGS.use_distortion_for_training), label
    dataset = dataset.map(
        _preprocess, num_threads=batch_size, output_buffer_size=2 * batch_size)
    # Repeat infinitely.
    dataset = dataset.repeat()
    if is_training:
      min_fraction_of_examples_in_queue = 0.4
      min_queue_examples = int(
          cifar10.Cifar10DataSet.num_examples_per_epoch(subset) *
          min_fraction_of_examples_in_queue)
      # Ensure that the capacity is sufficiently large to provide good random
      # shuffling
      dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    if num_shards <= 1:
      # No GPU available or only 1 GPU.
      return [image_batch], [label_batch]

    # Note that passing num=batch_size is safe here, even though
    # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
    # examples. This is because it does so only when repeating for a limited
    # number of epochs, but our dataset repeats forever.
    image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
    label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
    feature_shards = [[] for i in range(num_shards)]
    label_shards = [[] for i in range(num_shards)]
    for i in xrange(batch_size):
      idx = i % num_shards
      feature_shards[idx].append(image_batch[i])
      label_shards[idx].append(label_batch[i])
    feature_shards = [tf.parallel_stack(x) for x in feature_shards]
    label_shards = [tf.parallel_stack(x) for x in label_shards]
    return feature_shards, label_shards


def main(unused_argv):
  # The env variable is on deprecation path, default is set to off.
  os.environ['TF_SYNC_ON_FINISH'] = '0'

  if FLAGS.num_gpus < 0:
    raise ValueError(
        'Invalid GPU count: \"num_gpus\" must be 0 or a positive integer.')
  if FLAGS.num_gpus == 0 and not FLAGS.is_cpu_ps:
    raise ValueError(
        'No GPU available for use, must use CPU as parameter server.')
  if (FLAGS.num_layers - 2) % 6 != 0:
    raise ValueError('Invalid num_layers parameter.')
  if FLAGS.num_gpus != 0 and FLAGS.train_batch_size % FLAGS.num_gpus != 0:
    raise ValueError('train_batch_size must be multiple of num_gpus.')
  if FLAGS.num_gpus != 0 and FLAGS.eval_batch_size % FLAGS.num_gpus != 0:
    raise ValueError('eval_batch_size must be multiple of num_gpus.')

  num_eval_examples = cifar10.Cifar10DataSet.num_examples_per_epoch('eval')
  if num_eval_examples % FLAGS.eval_batch_size != 0:
    raise ValueError('validation set size must be multiple of eval_batch_size')

  config = tf.estimator.RunConfig()
  sess_config = tf.ConfigProto()
  sess_config.allow_soft_placement = True
  sess_config.log_device_placement = FLAGS.log_device_placement
  sess_config.intra_op_parallelism_threads = FLAGS.num_intra_threads
  sess_config.inter_op_parallelism_threads = FLAGS.num_inter_threads
  sess_config.gpu_options.force_gpu_compatible = FLAGS.force_gpu_compatible
  config = config.replace(session_config=sess_config)

  classifier = tf.estimator.Estimator(
      model_fn=_resnet_model_fn, model_dir=FLAGS.model_dir, config=config)

  tensors_to_log = {'learning_rate': 'learning_rate'}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)

  print('Starting to train...')
  classifier.train(
      input_fn=functools.partial(
          input_fn, subset='train', num_shards=FLAGS.num_gpus),
      steps=FLAGS.train_steps,
      hooks=[logging_hook])

  print('Starting to evaluate...')
  eval_results = classifier.evaluate(
      input_fn=functools.partial(
          input_fn, subset='eval', num_shards=FLAGS.num_gpus),
      steps=num_eval_examples // FLAGS.eval_batch_size)
  print(eval_results)


if __name__ == '__main__':
  tf.app.run()
