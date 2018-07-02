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

"""Training and evaluation for Mask_RCNN.

  This module repeatedly runs 1 training epoch and then evaluation
  ##add explanation for all the options!!!!!!!
"""

import functools
import json
import os

from object_detection import evaluator
from object_detection import trainer
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                                           'replicas.')
flags.DEFINE_integer('parameter_server_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', '',
                    'Directory to save the checkpoints and training summaries.')

flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job.')

flags.DEFINE_string('eval_dir', '',
                    'Directory to write eval summaries to.')

flags.DEFINE_boolean('run_once', False, 'Option to only run a single pass of '
                                        'evaluation. Overrides the `max_evals`'
                                        ' parameter in the provided config.')
flags.DEFINE_float('box_min_ap', -1, 'Option to run until the box average'
                                    'precision reaches this number')
flags.DEFINE_float('mask_min_ap', -1, 'Option to run until the mask average'
                                     'precision reaches this number')
flags.DEFINE_integer('epochs_between_evals', 1, 'Number of training epochs to '
                                                 'run before running eval.')
FLAGS = flags.FLAGS


def stopping_criteria_met(eval_metrics, mask_min_ap, box_min_ap):
  """Returns true if both of the min precision criteria are met in the given
  evaluation metrics.

  Args:
    eval_metrics: dict of metrics names as keys and their corresponding values,
      containing "DetectionMasks_Precision/mAP", and
      "DetectionBoxes_Precision/mAP" fields.
    mask_min_ap: minimum desired mask average precision, will be ignored if -1
    box_min_ap: minimum desired box average precision, will be ignored if -1

  Returns:
    True if non -1 criteria are met, false o.w.
  """
  assert mask_min_ap == -1 or 0 < mask_min_ap < 1
  assert box_min_ap == -1 or 0 < box_min_ap < 1
  try:
    mask_mAP_reached = eval_metrics['DetectionMasks_Precision/mAP']
    box_mAP_reached = eval_metrics['DetectionBoxes_Precision/mAP']
  except KeyError as err:
    raise Exception('eval_metrics dict does not contain the mAP field') from err

  return (mask_min_ap == -1 or mask_mAP_reached > mask_min_ap) & \
         (box_min_ap == -1 or box_mAP_reached > box_min_ap) & \
         (mask_min_ap != -1 or box_min_ap != -1)


def main(_):
  assert FLAGS.train_dir, '`train_dir` is missing.'
  assert FLAGS.pipeline_config_path, '`pipeline_config_path` is missing'
  assert FLAGS.eval_dir, '`eval_dir` is missing.'

  configs = config_util.get_configs_from_pipeline_file(
      FLAGS.pipeline_config_path)
  if FLAGS.task == 0:
    tf.gfile.MakeDirs(FLAGS.train_dir)
    tf.gfile.Copy(FLAGS.pipeline_config_path,
                  os.path.join(FLAGS.train_dir, 'pipeline.config'),
                  overwrite=True)

  tf.gfile.MakeDirs(FLAGS.eval_dir)
  tf.gfile.Copy(FLAGS.pipeline_config_path,
                os.path.join(FLAGS.eval_dir, 'pipeline.config'),
                overwrite=True)

  model_config = configs['model']

  train_config = configs['train_config']
  train_input_config = configs['train_input_config']

  eval_config = configs['eval_config']
  if FLAGS.eval_training_data:
    eval_input_config = configs['train_input_config']
  else:
    eval_input_config = configs['eval_input_config']

  # setting to run evaluation after EPOCHS_BETWEEN_EVALS epochs of training.
  # total number of training is set to total_num_epochs provided in the config
  if train_config.num_steps:
    total_num_epochs = train_config.num_steps
    train_config.num_steps = FLAGS.epochs_between_evals
    total_training_cycle = total_num_epochs // train_config.num_steps
  else:
    # TODO(mehdi): make it run indef
    total_num_epochs = 20000000
    train_config.num_steps = FLAGS.epochs_between_evals
    total_training_cycle = total_num_epochs // train_config.num_steps

  train_model_fn = functools.partial(model_builder.build,
                                     model_config=model_config,
                                     is_training=True)
  eval_model_fn = functools.partial(model_builder.build,
                                    model_config=model_config,
                                    is_training=False)

  def get_next(config):
    return dataset_util.make_initializable_iterator(
        dataset_builder.build(config)).get_next()

  # functions to create a tensor input dictionary for both training & evaluation
  train_input_dict_fn = functools.partial(get_next, train_input_config)
  eval_input_dict_fn = functools.partial(get_next, eval_input_config)

  # If not explicitly specified in the constructor and the TF_CONFIG
  # environment variable is present, load cluster_spec from TF_CONFIG.
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  cluster_data = env.get('cluster', None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
  task_data = env.get('task', {'type': 'master', 'index': 0})
  task_info = type('TaskSpec', (object,), task_data)

  # Parameters for a single worker.
  parameter_server_tasks = 0
  worker_replicas = 1
  worker_job_name = 'lonely_worker'
  task = 0
  is_chief = True
  master = ''

  if cluster_data and 'worker' in cluster_data:
    # Number of total worker replicas include "worker"s and the "master".
    worker_replicas = len(cluster_data['worker']) + 1
  if cluster_data and 'ps' in cluster_data:
    parameter_server_tasks = len(cluster_data['ps'])

  if worker_replicas > 1 and parameter_server_tasks < 1:
    raise ValueError('At least 1 ps task is needed for distributed training.')

  if worker_replicas >= 1 and parameter_server_tasks > 0:
    # Set up distributed training.
    server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                             job_name=task_info.type,
                             task_index=task_info.index)
    if task_info.type == 'ps':
      server.join()
      return

    worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
    task = task_info.index
    is_chief = (task_info.type == 'master')
    master = server.target

  label_map = label_map_util.load_labelmap(eval_input_config.label_map_path)
  max_num_classes = max([item.id for item in label_map.item])
  categories = label_map_util.convert_label_map_to_categories(label_map,
                                                              max_num_classes)

  if FLAGS.run_once:
    eval_config.max_evals = 1

  train_graph_rewriter_fn = eval_graph_rewriter_fn = None
  if 'graph_rewriter_config' in configs:
    train_graph_rewriter_fn = graph_rewriter_builder.build(
        configs['graph_rewriter_config'], is_training=True)
    eval_graph_rewriter_fn = graph_rewriter_builder.build(
        configs['eval_rewriter_config'], is_training=False)

  def train():
    return trainer.train(create_tensor_dict_fn=train_input_dict_fn,
                         create_model_fn=train_model_fn,
                         train_config=train_config, master=master, task=task,
                         num_clones=FLAGS.num_clones,
                         worker_replicas=worker_replicas,
                         clone_on_cpu=FLAGS.clone_on_cpu,
                         ps_tasks=parameter_server_tasks,
                         worker_job_name=worker_job_name,
                         is_chief=is_chief, train_dir=FLAGS.train_dir,
                         graph_hook_fn=train_graph_rewriter_fn)

  def evaluate():
    return evaluator.evaluate(eval_input_dict_fn, eval_model_fn, eval_config,
                              categories, FLAGS.train_dir, FLAGS.eval_dir,
                              graph_hook_fn=eval_graph_rewriter_fn)

  for cycle_index in range(total_training_cycle):
    tf.logging.info('Starting a training cycle: %d/%d',
                    cycle_index, total_training_cycle)
    train()
    tf.logging.info('Starting to evaluate.')
    eval_metrics = evaluate()
    if stopping_criteria_met(eval_metrics, FLAGS.mask_min_ap, FLAGS.box_min_ap):
      tf.logging.info('Stopping criteria met. Training stopped')
      break


if __name__ == '__main__':
  tf.app.run()
