#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with our temperature
 prediction model.

The client loads our test set, queries the service with
such data points and calculates our mean absolute error.

Typical usage example:

    temperature_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import sys
import threading

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
import time

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import data_utils


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests',
                            -1,
                            'Number of tests to run against the server. If you '
                            'do not specify a number of tests to run, the '
                            'client will automatically test with the entire '
                            'test set.')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('data_path', '/tmp/merge_2006_2012.csv',
                           'Path to the data csv file')
tf.app.flags.DEFINE_integer('data_points_day',
                            144,
                            'Number of data points in a day (24 hrs) default: '
                            'sample rate = 1/10min = 144 data points/day')
tf.app.flags.DEFINE_integer('num_days',
                            10,
                            'Number of days involved in lookback')
tf.app.flags.DEFINE_integer('steps',
                            6,
                            'Sample from datastream at this number of steps. '
                            'Default: sample rate = 1/10min = 6 steps '
                            'to sample at 1/hr')
FLAGS = tf.app.flags.FLAGS

class _ResultMonitor(object):
  """Monitor for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._mae = 0

    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def update_mae(self, preds, targets):
    with self._condition:
      self._mae += np.mean(np.abs(preds - targets))

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_mae(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._mae / float(self._num_tests)

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1


def _create_rpc_callback(label, result_counter):
  """Creates RPC callback function.

  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      print("An error occured: {}".format(exception))
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      prediction = np.array(
          result_future.result().outputs['temperature'].float_val)
      result_counter.update_mae(prediction, label)
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback


def do_inference(hostport, concurrency, num_tests):
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """

  ds = data_utils.Dataset(FLAGS.data_path,
                          1,
                          FLAGS.data_points_day,
                          FLAGS.num_days,
                          FLAGS.steps,
                          'inference')
  test_data_set = ds.load_jena_data()[-1]
  if num_tests == -1:
    print("Using whole test set")
    num_tests = ds.num_test
  assert num_tests <= ds.num_test, "More tests than available in test set."

  test_iter = test_data_set.make_one_shot_iterator()
  test_next_el = test_iter.get_next()

  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  result_counter = _ResultMonitor(num_tests, concurrency)

  print("Beginning inference")
  start_t = time.time()
  with tf.Session() as sess:
    for _ in range(num_tests):
      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'temp_predict'
      request.model_spec.signature_name = 'serving_default'
      samples, targets = sess.run(test_next_el)

      request.inputs['weather_data'].CopyFrom(
          tf.contrib.util.make_tensor_proto(samples[0], shape=[1,
                                                               samples.shape[1],
                                                               samples.shape[2]]))
      result_counter.throttle()
      result_future = stub.Predict.future(request, 5.0)  # 5 seconds
      result_future.add_done_callback(
          _create_rpc_callback(targets[0], result_counter))
  print("Finished in {}s".format(time.time() - start_t))
  return result_counter.get_mae()


def main(_):
  if not FLAGS.server:
    print('please specify server host:port')
    return
  error_rate = do_inference(FLAGS.server,
                            FLAGS.concurrency, FLAGS.num_tests)
  print('\nInference mean absolute error: %s' % error_rate)


if __name__ == '__main__':
  tf.app.run()