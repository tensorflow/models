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
import argparse

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("server",
                    help="PredictionService host:port")
parser.add_argument("--concurrency",
                    type=int, default=1,
                    help="maximum number of concurrent inference requests")
parser.add_argument("--num_tests",
                    type=int, default=-1,
                    help="Number of tests to run against the server. If you do "
                         "not specify a number of tests to run, the client "
                         "will automatically test with the entire test set.")
parser.add_argument("--data_path",
                    type=str, default="./data/merge_2006_2012.csv",
                    help="Path to the data csv file")
parser.add_argument("--data_points_day",
                    type=int, default=144,
                    help="Number of data points in a day (24 hrs) default: "
                         "sample rate = 1/10min = 144 data points/day")
parser.add_argument("--num_days",
                    type=int, default=10,
                    help="Number of days involved in lookback")
parser.add_argument("--steps",
                    type=int, default=6,
                    help="Sample from datastream at this number of steps. "
                         "Default: sample rate = 1/10min = 6 steps "
                         "to sample at 1/hr")
args = parser.parse_args()



class _ResultMonitor(object):
  """Monitor for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._mae = 0

    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def get_done(self):
    return self._done

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
      if result_counter.get_done() % 100 == 0:
        sys.stdout.write(".")
        sys.stdout.flush()
      prediction = np.array(
          result_future.result().outputs["temperature"].float_val)
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

  ds = data_utils.Dataset(args.data_path,
                          1,
                          args.data_points_day,
                          args.num_days,
                          args.steps,
                          "inference")
  test_data_set = ds.get_jena_datasets()[-1]
  if num_tests == -1:
    print("Using whole test set")
    num_tests = ds.num_test
  if num_tests > ds.num_test:
    raise ValueError("Num Tests more than available "
                     "in test set. ({}, {})".format(num_tests, ds.num_test))

  host, port = hostport.split(":")
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  result_counter = _ResultMonitor(num_tests, concurrency)

  print("Beginning inference")
  start_t = time.time()
  for samples, targets in test_data_set:
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "temp_predict"
    request.model_spec.signature_name = "serving_default"

    request.inputs["weather_data"].CopyFrom(
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
  error_rate = do_inference(args.server,
                            args.concurrency, args.num_tests)
  print("\nInference mean absolute error: %s" % error_rate)


if __name__ == "__main__":
  tf.app.run()