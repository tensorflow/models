# Copyright 2017 Google Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
import sys
import os

import numpy as np
import tensorflow as tf

import rebar
import datasets
import logger as L

try:
  xrange          # Python 2
except NameError:
  xrange = range  # Python 3

gfile = tf.gfile

tf.app.flags.DEFINE_string("working_dir", "/tmp/rebar",
                           """Directory where to save data, write logs, etc.""")
tf.app.flags.DEFINE_string('hparams', '',
                           '''Comma separated list of name=value pairs.''')
tf.app.flags.DEFINE_integer('eval_freq', 20,
                           '''How often to run the evaluation step.''')
FLAGS = tf.flags.FLAGS

def manual_scalar_summary(name, value):
  value = tf.Summary.Value(tag=name, simple_value=value)
  summary_str = tf.Summary(value=[value])
  return summary_str

def eval(sbn, eval_xs, n_samples=100, batch_size=5):
  n = eval_xs.shape[0]
  i = 0
  res = []
  while i < n:
    batch_xs = eval_xs[i:min(i+batch_size, n)]
    res.append(sbn.partial_eval(batch_xs, n_samples))
    i += batch_size
  res = np.mean(res, axis=0)
  return res

def train(sbn, train_xs, valid_xs, test_xs, training_steps, debug=False):
  hparams = sorted(sbn.hparams.values().items())
  hparams = (map(str, x) for x in hparams)
  hparams = ('_'.join(x) for x in hparams)
  hparams_str = '.'.join(hparams)

  logger = L.Logger()

  # Create the experiment name from the hparams
  experiment_name = ([str(sbn.hparams.n_hidden) for i in xrange(sbn.hparams.n_layer)] +
                     [str(sbn.hparams.n_input)])
  if sbn.hparams.nonlinear:
    experiment_name = '~'.join(experiment_name)
  else:
    experiment_name = '-'.join(experiment_name)
  experiment_name = 'SBN_%s' % experiment_name
  rowkey = {'experiment': experiment_name,
            'model': hparams_str}

  # Create summary writer
  summ_dir = os.path.join(FLAGS.working_dir, hparams_str)
  summary_writer = tf.summary.FileWriter(
      summ_dir, flush_secs=15, max_queue=100)

  sv = tf.train.Supervisor(logdir=os.path.join(
      FLAGS.working_dir, hparams_str),
                     save_summaries_secs=0,
                     save_model_secs=1200,
                     summary_op=None,
                     recovery_wait_secs=30,
                     global_step=sbn.global_step)
  with sv.managed_session() as sess:
    # Dump hparams to file
    with gfile.Open(os.path.join(FLAGS.working_dir,
                                 hparams_str,
                                 'hparams.json'),
                    'w') as out:
      json.dump(sbn.hparams.values(), out)

    sbn.initialize(sess)
    batch_size = sbn.hparams.batch_size
    scores = []
    n = train_xs.shape[0]
    index = range(n)

    while not sv.should_stop():
      lHats = []
      grad_variances = []
      temperatures = []
      random.shuffle(index)
      i = 0
      while i < n:
        batch_index = index[i:min(i+batch_size, n)]
        batch_xs = train_xs[batch_index, :]

        if sbn.hparams.dynamic_b:
          # Dynamically binarize the batch data
          batch_xs = (np.random.rand(*batch_xs.shape) < batch_xs).astype(float)

        lHat, grad_variance, step, temperature = sbn.partial_fit(batch_xs,
                                                    sbn.hparams.n_samples)
        if debug:
          print(i, lHat)
          if i > 100:
            return
        lHats.append(lHat)
        grad_variances.append(grad_variance)
        temperatures.append(temperature)
        i += batch_size

      grad_variances = np.log(np.mean(grad_variances, axis=0)).tolist()
      summary_strings = []
      if isinstance(grad_variances, list):
        grad_variances = dict(zip([k for (k, v) in sbn.losses], map(float, grad_variances)))
        rowkey['step'] = step
        logger.log(rowkey, {'step': step,
                             'train': np.mean(lHats, axis=0)[0],
                             'grad_variances': grad_variances,
                             'temperature': np.mean(temperatures), })
        grad_variances = '\n'.join(map(str, sorted(grad_variances.iteritems())))
      else:
        rowkey['step'] = step
        logger.log(rowkey, {'step': step,
                             'train': np.mean(lHats, axis=0)[0],
                             'grad_variance': grad_variances,
                             'temperature': np.mean(temperatures), })
        summary_strings.append(manual_scalar_summary("log grad variance", grad_variances))

      print('Step %d: %s\n%s' % (step, str(np.mean(lHats, axis=0)), str(grad_variances)))

      # Every few epochs compute test and validation scores
      epoch = int(step / (train_xs.shape[0] / sbn.hparams.batch_size))
      if epoch % FLAGS.eval_freq == 0:
        valid_res = eval(sbn, valid_xs)
        test_res= eval(sbn, test_xs)

        print('\nValid %d: %s' % (step, str(valid_res)))
        print('Test %d: %s\n' % (step, str(test_res)))
        logger.log(rowkey, {'step': step,
                             'valid': valid_res[0],
                             'test': test_res[0]})
        logger.flush()  # Flush infrequently

      # Create summaries
      summary_strings.extend([
        manual_scalar_summary("Train ELBO", np.mean(lHats, axis=0)[0]),
        manual_scalar_summary("Temperature", np.mean(temperatures)),
      ])
      for summ_str in summary_strings:
        summary_writer.add_summary(summ_str, global_step=step)
      summary_writer.flush()

      sys.stdout.flush()
      scores.append(np.mean(lHats, axis=0))

      if step > training_steps:
        break

    return scores


def main():
  # Parse hyperparams
  hparams = rebar.default_hparams
  hparams.parse(FLAGS.hparams)
  print(hparams.values())

  train_xs, valid_xs, test_xs = datasets.load_data(hparams)
  mean_xs = np.mean(train_xs, axis=0)  # Compute mean centering on training

  training_steps = 2000000
  model = getattr(rebar, hparams.model)
  sbn = model(hparams, mean_xs=mean_xs)

  scores = train(sbn, train_xs, valid_xs, test_xs,
                 training_steps=training_steps, debug=False)

if __name__ == '__main__':
  main()
