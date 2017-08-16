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
from __future__ import division
from __future__ import print_function

import argparse
import functools
import operator
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

import cifar10
import cifar10_model

tf.logging.set_verbosity(tf.logging.INFO)


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
  """Hook to print out examples per second.

    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.
  """

  def __init__(
      self,
      batch_size,
      every_n_steps=100,
      every_n_secs=None,):
    """Initializer for ExamplesPerSecondHook.

      Args:
      batch_size: Total batch size used to calculate examples/second from
      global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds.
    """
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
                       ' and every_n_secs should be provided.')
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    _ = run_context

    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        steps_per_sec = elapsed_steps / elapsed_time
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        current_examples_per_sec = steps_per_sec * self._batch_size
        # Average examples/sec followed by current examples/sec
        logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                     average_examples_per_sec, current_examples_per_sec,
                     self._total_steps)


class GpuParamServerDeviceSetter(object):
  """Used with tf.device() to place variables on the least loaded GPU.

    A common use for this class is to pass a list of GPU devices, e.g. ['gpu:0',
    'gpu:1','gpu:2'], as ps_devices.  When each variable is placed, it will be
    placed on the least loaded gpu. All other Ops, which will be the computation
    Ops, will be placed on the worker_device.
  """

  def __init__(self, worker_device, ps_devices):
    """Initializer for GpuParamServerDeviceSetter.

    Args:
      worker_device: the device to use for computation Ops.
      ps_devices: a list of devices to use for Variable Ops. Each variable is
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

    # Gets the least loaded ps_device
    device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
    device_name = self.ps_devices[device_index]
    var_size = op.outputs[0].get_shape().num_elements()
    self.ps_sizes[device_index] += var_size

    return device_name


def _create_device_setter(avg_on_gpu, worker, num_gpus):
  """Create device setter object."""
  if avg_on_gpu:
    gpus = ['/gpu:%d' % i for i in range(num_gpus)]
    return GpuParamServerDeviceSetter(worker, gpus)
  else:
    # tf.train.replica_device_setter supports placing variables on the CPU, all
    # on one GPU, or on ps_servers defined in a cluster_spec.
    return tf.train.replica_device_setter(
        worker_device=worker, ps_device='/cpu:0', ps_tasks=1)

def get_model_fn(num_gpus, avg_on_gpu, num_workers):
  def _resnet_model_fn(features, labels, mode, params):
    """Resnet model body.

    Support single host, one or more GPU training. Parameter distribution can be
    either one of the following scheme.
    1. CPU is the parameter server and manages gradient updates.
    2. Parameters are distributed evenly across all GPUs, and the first GPU
       manages gradient updates.

    Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Dictionary of Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    weight_decay = params['weight_decay']
    momentum = params['momentum']

    tower_features = features
    tower_labels = labels
    tower_losses = []
    tower_gradvars = []
    tower_preds = []

    if num_gpus != 0:
      for i in range(num_gpus):
        worker = '/gpu:%d' % i
        device_setter = _create_device_setter(avg_on_gpu, worker, num_gpus)
        with tf.variable_scope('resnet', reuse=bool(i != 0)):
          with tf.name_scope('tower_%d' % i) as name_scope:
            with tf.device(device_setter):
              loss, gradvars, preds = _tower_fn(
                  is_training,
                  weight_decay,
                  tower_features[i],
                  tower_labels[i],
                  False,
                  params['num_layers'],
                  params['batch_norm_decay'],
                  params['batch_norm_epsilon'])
              tower_losses.append(loss)
              tower_gradvars.append(gradvars)
              tower_preds.append(preds)
              if i == 0:
                # Only trigger batch_norm moving mean and variance update from
                # the 1st tower. Ideally, we should grab the updates from all
                # towers but these stats accumulate extremely fast so we can
                # ignore the other stats from the other towers without
                # significant detriment.
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                               name_scope)
    else:
      with tf.variable_scope('resnet'), tf.device('/cpu:0'):
        with tf.name_scope('tower_cpu') as name_scope:
          loss, gradvars, preds = _tower_fn(
              is_training,
              weight_decay,
              tower_features[0],
              tower_labels[0],
              True,
              params['num_layers'],
              params['batch_norm_decay'],
              params['batch_norm_epsilon'])
          tower_losses.append(loss)
          tower_gradvars.append(gradvars)
          tower_preds.append(preds)

          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

    # Now compute global loss and gradients.
    gradvars = []
    # Server that runs the ops to apply global gradient updates.
    avg_device = '/gpu:0' if avg_on_gpu else '/cpu:0'
    with tf.device(avg_device):
      with tf.name_scope('gradient_averaging'):
        loss = tf.reduce_mean(tower_losses, name='loss')
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
          'train') // (params['train_batch_size'] * num_workers)
      boundaries = [
          num_batches_per_epoch * x
          for x in np.array([82, 123, 300], dtype=np.int64)
      ]
      staged_lr = [params['learning_rate'] * x for x in [1, 0.1, 0.01, 0.002]]

      learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
                                                  boundaries, staged_lr)
      # Create a nicely-named tensor for logging
      learning_rate = tf.identity(learning_rate, name='learning_rate')

      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=momentum)

      chief_hooks = []
      if params['sync']:
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=num_workers)
        sync_replicas_hook = optimizer.make_session_run_hook(True)
        chief_hooks.append(sync_replicas_hook)

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
        training_chief_hooks=chief_hooks,
        eval_metric_ops=metrics)
  return _resnet_model_fn


def _tower_fn(is_training,
              weight_decay,
              feature,
              label,
              is_cpu,
              num_layers,
              batch_norm_decay,
              batch_norm_epsilon):
  """Build computation tower for each device (CPU or GPU).

  Args:
    is_training: true if is training graph.
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
      num_layers,
      batch_norm_decay=batch_norm_decay,
      batch_norm_epsilon=batch_norm_epsilon,
      is_training=is_training, data_format=data_format)
  logits = model.forward_pass(feature, input_data_format='channels_last')
  tower_pred = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits)
  }

  tower_loss = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=label)
  tower_loss = tf.reduce_mean(tower_loss)

  model_params = tf.trainable_variables()
  tower_loss += weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in model_params])

  tower_grad = tf.gradients(tower_loss, model_params)

  return tower_loss, tower_grad, tower_pred


def input_fn(data_dir, subset, num_shards, batch_size,
             use_distortion_for_training=True):
  """Create input graph for model.

  Args:
    subset: one of 'train', 'validate' and 'eval'.
    num_shards: num of towers participating in data-parallel training.
  Returns:
    two lists of tensors for features and labels, each of num_shards length.
  """
  with tf.device('/cpu:0'):
    use_distortion = subset == 'train' and use_distortion_for_training
    dataset = cifar10.Cifar10DataSet(data_dir, subset, use_distortion)
    image_batch, label_batch = dataset.make_batch(batch_size)
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


# create experiment
def get_experiment_fn(data_dir, num_gpus, is_gpu_ps,
                      use_distortion_for_training=True):
  """Returns an Experiment function.

  Experiments perform training on several workers in parallel,
  in other words experiments know how to invoke train and eval in a sensible
  fashion for distributed training. Arguments passed directly to this
  function are not tunable, all other arguments should be passed within
  tf.HParams, passed to the enclosed function.

  Args:
      data_dir: str. Location of the data for input_fns.
      num_gpus: int. Number of GPUs on each worker.
      is_gpu_ps: bool. If true, average gradients on GPUs.
      use_distortion_for_training: bool. See cifar10.Cifar10DataSet.
  Returns:
      A function (tf.estimator.RunConfig, tf.contrib.training.HParams) ->
      tf.contrib.learn.Experiment.

      Suitable for use by tf.contrib.learn.learn_runner, which will run various
      methods on Experiment (train, evaluate) based on information
      about the current runner in `run_config`.
  """
  def _experiment_fn(run_config, hparams):
    """Returns an Experiment."""
    # Create estimator.
    train_input_fn = functools.partial(
        input_fn,
        data_dir,
        subset='train',
        num_shards=num_gpus,
        batch_size=hparams.train_batch_size,
        use_distortion_for_training=use_distortion_for_training
    )

    eval_input_fn = functools.partial(
        input_fn,
        data_dir,
        subset='eval',
        batch_size=hparams.eval_batch_size,
        num_shards=num_gpus
    )

    num_eval_examples = cifar10.Cifar10DataSet.num_examples_per_epoch('eval')
    if num_eval_examples % hparams.eval_batch_size != 0:
      raise ValueError('validation set size must be multiple of eval_batch_size')

    train_steps = hparams.train_steps
    eval_steps = num_eval_examples // hparams.eval_batch_size
    examples_sec_hook = ExamplesPerSecondHook(
      hparams.train_batch_size, every_n_steps=10)

    tensors_to_log = {'learning_rate': 'learning_rate',
                      'loss': 'gradient_averaging/loss'}

    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)

    hooks = [logging_hook, examples_sec_hook]

    classifier = tf.estimator.Estimator(
        model_fn=get_model_fn(
            num_gpus, is_gpu_ps, run_config.num_worker_replicas),
        config=run_config,
        params=vars(hparams)
    )

    # Create experiment.
    experiment = tf.contrib.learn.Experiment(
        classifier,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=train_steps,
        eval_steps=eval_steps)
    # Adding hooks to be used by the estimator on training mode.
    experiment.extend_train_hooks(hooks)
    return experiment
  return _experiment_fn


def main(job_dir,
         data_dir,
         num_gpus,
         avg_on_gpu,
         use_distortion_for_training,
         log_device_placement,
         num_intra_threads,
         force_gpu_compatible,
         **hparams):
  # The env variable is on deprecation path, default is set to off.
  os.environ['TF_SYNC_ON_FINISH'] = '0'

  # Session configuration.
  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=log_device_placement,
      intra_op_parallelism_threads=num_intra_threads,
      gpu_options=tf.GPUOptions(
          force_gpu_compatible=force_gpu_compatible
      )
  )

  config = tf.contrib.learn.RunConfig(
      session_config=sess_config,
      model_dir=job_dir)
  tf.contrib.learn.learn_runner.run(
      get_experiment_fn(
          data_dir,
          num_gpus,
          avg_on_gpu,
          use_distortion_for_training
      ),
      run_config=config,
      hparams=tf.contrib.training.HParams(**hparams)
  )


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      required=True,
      help='The directory where the CIFAR-10 input data is stored.'
  )
  parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='The directory where the model will be stored.'
  )
  parser.add_argument(
      '--avg-on-gpu',
      action='store_true',
      default=False,
      help='If present, use GPU to average gradients.'
  )
  parser.add_argument(
      '--num-gpus',
      type=int,
      default=1,
      help='The number of gpus used. Uses only CPU if set to 0.'
  )
  parser.add_argument(
      '--num-layers',
      type=int,
      default=44,
      help='The number of layers of the model.'
  )
  parser.add_argument(
      '--train-steps',
      type=int,
      default=80000,
      help='The number of steps to use for training.'
  )
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=128,
      help='Batch size for training.'
  )
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=100,
      help='Batch size for validation.'
  )
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.9,
      help='Momentum for MomentumOptimizer.'
  )
  parser.add_argument(
      '--weight-decay',
      type=float,
      default=2e-4,
      help='Weight decay for convolutions.'
  )
  
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.1,
      help="""\
      This is the inital learning rate value. The learning rate will decrease
      during training. For more details check the model_fn implementation in
      this file.\
      """
  )
  parser.add_argument(
      '--use-distortion-for-training',
      type=bool,
      default=True,
      help='If doing image distortion for training.'
  )
  parser.add_argument(
      '--sync',
      action='store_true',
      default=False,
      help="""\
      If present when running in a distributed environment will run on sync mode.\
      """
  )
  parser.add_argument(
      '--num-intra-threads',
      type=int,
      default=1,
      help="""\
      Number of threads to use for intra-op parallelism. If set to 0, the
      system will pick an appropriate number. The default is 1 since in this
      example CPU only handles the input pipeline and gradient aggregation
      (when --is-cpu-ps). Ops that could potentially benefit from intra-op
      parallelism are scheduled to run on GPUs.\
      """
  )
  parser.add_argument(
      '--num-inter-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.\
      """
  )
  parser.add_argument(
      '--force-gpu-compatible',
      action='store_true',
      default=False,
      help="""\
      Whether to enable force_gpu_compatible in GPU_Options. Check
      tensorflow/core/protobuf/config.proto#L69 for details.\
      """
  )
  parser.add_argument(
      '--log-device-placement',
      action='store_true',
      default=False,
      help='Whether to log device placement.'
  )
  parser.add_argument(
      '--batch_norm_decay',
      type=float,
      default=0.997,
      help='Decay for batch norm.'
  )
  parser.add_argument(
      '--batch_norm_epsilon',
      type=float,
      default=1e-5,
      help='Epsilon for batch norm.'
  )
  args = parser.parse_args()

  if args.num_gpus < 0:
    raise ValueError(
        'Invalid GPU count: \"num_gpus\" must be 0 or a positive integer.')
  if args.num_gpus == 0 and args.avg_on_gpu:
    raise ValueError(
        'No GPU available for use, must use CPU to average gradients.')
  if (args.num_layers - 2) % 6 != 0:
    raise ValueError('Invalid num_layers parameter.')
  if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
    raise ValueError('train_batch_size must be multiple of num_gpus.')
  if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
    raise ValueError('eval_batch_size must be multiple of num_gpus.')

  main(**vars(args))
