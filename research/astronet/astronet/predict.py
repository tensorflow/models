# Copyright 2018 The TensorFlow Authors.
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

"""Generate predictions for a Threshold Crossing Event using a trained model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from astronet import models
from astronet.data import preprocess
from astronet.util import config_util
from astronet.util import configdict
from astronet.util import estimator_util

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", type=str, required=True, help="Name of the model class.")

parser.add_argument(
    "--config_name",
    type=str,
    help="Name of the model and training configuration. Exactly one of "
    "--config_name or --config_json is required.")

parser.add_argument(
    "--config_json",
    type=str,
    help="JSON string or JSON file containing the model and training "
    "configuration. Exactly one of --config_name or --config_json is required.")

parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Directory containing a model checkpoint.")

parser.add_argument(
    "--kepler_data_dir",
    type=str,
    required=True,
    help="Base folder containing Kepler data.")

parser.add_argument(
    "--kepler_id",
    type=int,
    required=True,
    help="Kepler ID of the target star.")

parser.add_argument(
    "--period", type=float, required=True, help="Period of the TCE, in days.")

parser.add_argument("--t0", type=float, required=True, help="Epoch of the TCE.")

parser.add_argument(
    "--duration",
    type=float,
    required=True,
    help="Duration of the TCE, in days.")

parser.add_argument(
    "--output_image_file",
    type=str,
    help="If specified, path to an output image file containing feature plots. "
    "Must end in a valid image extension, e.g. png.")


def _process_tce(feature_config):
  """Reads and process the input features of a Threshold Crossing Event.

  Args:
    feature_config: ConfigDict containing the feature configurations.

  Returns:
    A dictionary of processed light curve features.

  Raises:
    ValueError: If feature_config contains features other than 'global_view'
    and 'local_view'.
  """
  if not {"global_view", "local_view"}.issuperset(feature_config.keys()):
    raise ValueError(
        "Only 'global_view' and 'local_view' features are supported.")

  # Read and process the light curve.
  time, flux = preprocess.read_and_process_light_curve(FLAGS.kepler_id,
                                                       FLAGS.kepler_data_dir)
  time, flux = preprocess.phase_fold_and_sort_light_curve(
      time, flux, FLAGS.period, FLAGS.t0)

  # Generate the local and global views.
  features = {}

  if "global_view" in feature_config:
    global_view = preprocess.global_view(time, flux, FLAGS.period)
    # Add a batch dimension.
    features["global_view"] = np.expand_dims(global_view, 0)

  if "local_view" in feature_config:
    local_view = preprocess.local_view(time, flux, FLAGS.period, FLAGS.duration)
    # Add a batch dimension.
    features["local_view"] = np.expand_dims(local_view, 0)

  # Possibly save plots.
  if FLAGS.output_image_file:
    ncols = len(features)
    fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 5), squeeze=False)

    for i, name in enumerate(sorted(features)):
      ax = axes[0][i]
      ax.plot(features[name][0], ".")
      ax.set_title(name)
      ax.set_xlabel("Bucketized Time (days)")
      ax.set_ylabel("Normalized Flux")

    fig.tight_layout()
    fig.savefig(FLAGS.output_image_file, bbox_inches="tight")

  return features


def main(_):
  model_class = models.get_model_class(FLAGS.model)

  # Look up the model configuration.
  assert (FLAGS.config_name is None) != (FLAGS.config_json is None), (
      "Exactly one of --config_name or --config_json is required.")
  config = (
      models.get_model_config(FLAGS.model, FLAGS.config_name)
      if FLAGS.config_name else config_util.parse_json(FLAGS.config_json))
  config = configdict.ConfigDict(config)

  # Create the estimator.
  estimator = estimator_util.create_estimator(
      model_class, config.hparams, model_dir=FLAGS.model_dir)

  # Read and process the input features.
  features = _process_tce(config.inputs.features)

  # Create an input function.
  def input_fn():
    return {
        "time_series_features":
            tf.estimator.inputs.numpy_input_fn(
                features, batch_size=1, shuffle=False, queue_capacity=1)()
    }

  # Generate the predictions.
  for predictions in estimator.predict(input_fn):
    assert len(predictions) == 1
    print("Prediction:", predictions[0])


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
