# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Regression using the DNNRegressor Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

import automobile_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument('--price_norm_factor', default=1000., type=float,
                    help='price normalization factor')

def my_dnn_regression_fn(features, labels, mode, params):
  """A model function implementing DNN regression for a custom Estimator."""

  # Extract the input into a dense layer, according to the feature_columns.
  top = tf.feature_column.input_layer(features, params["feature_columns"])

  # Iterate over the "hidden_units" list of layer sizes, default is [20].
  for units in params.get("hidden_units", [20]):
    # Add a hidden layer, densely connected on top of the previous layer.
    top = tf.layers.dense(inputs=top, units=units, activation=tf.nn.relu)

  # Connect a linear output layer on top.
  output_layer = tf.layers.dense(inputs=top, units=1)

  # Reshape the output layer to a 1-dim Tensor to return predictions
  predictions = tf.squeeze(output_layer, 1)

  if mode == tf.estimator.ModeKeys.PREDICT:
    # In `PREDICT` mode we only need to return predictions.
    return tf.estimator.EstimatorSpec(
        mode=mode, predictions={"price": predictions})

  # Calculate loss using mean squared error
  average_loss = tf.losses.mean_squared_error(labels, predictions)

  # Pre-made estimators use the total_loss instead of the average,
  # so report total_loss for compatibility.
  batch_size = tf.shape(labels)[0]
  total_loss = tf.to_float(batch_size) * average_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = params.get("optimizer", tf.train.AdamOptimizer)
    optimizer = optimizer(params.get("learning_rate", None))
    train_op = optimizer.minimize(
        loss=average_loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op)

  # In evaluation mode we will calculate evaluation metrics.
  assert mode == tf.estimator.ModeKeys.EVAL

  # Calculate root mean squared error
  print(labels)
  print(predictions)

  # Fixed for #4083
  predictions = tf.cast(predictions, tf.float64)

  rmse = tf.metrics.root_mean_squared_error(labels, predictions)

  # Add the rmse to the collection of evaluation metrics.
  eval_metrics = {"rmse": rmse}

  return tf.estimator.EstimatorSpec(
      mode=mode,
      # Report sum of error for compatibility with pre-made estimators
      loss=total_loss,
      eval_metric_ops=eval_metrics)


def main(argv):
  """Builds, trains, and evaluates the model."""
  args = parser.parse_args(argv[1:])

  (train_x,train_y), (test_x, test_y) = automobile_data.load_data()

  train_y /= args.price_norm_factor
  test_y /= args.price_norm_factor

  # Provide the training input dataset.
  train_input_fn = automobile_data.make_dataset(args.batch_size, train_x, train_y, True, 1000)

  # Build the validation dataset.
  test_input_fn = automobile_data.make_dataset(args.batch_size, test_x, test_y)

  # The first way assigns a unique weight to each category. To do this you must
  # specify the category's vocabulary (values outside this specification will
  # receive a weight of zero). Here we specify the vocabulary using a list of
  # options. The vocabulary can also be specified with a vocabulary file (using
  # `categorical_column_with_vocabulary_file`). For features covering a
  # range of positive integers use `categorical_column_with_identity`.
  body_style_vocab = ["hardtop", "wagon", "sedan", "hatchback", "convertible"]
  body_style = tf.feature_column.categorical_column_with_vocabulary_list(
      key="body-style", vocabulary_list=body_style_vocab)
  make = tf.feature_column.categorical_column_with_hash_bucket(
      key="make", hash_bucket_size=50)

  feature_columns = [
      tf.feature_column.numeric_column(key="curb-weight"),
      tf.feature_column.numeric_column(key="highway-mpg"),
      # Since this is a DNN model, convert categorical columns from sparse
      # to dense.
      # Wrap them in an `indicator_column` to create a
      # one-hot vector from the input.
      tf.feature_column.indicator_column(body_style),
      # Or use an `embedding_column` to create a trainable vector for each
      # index.
      tf.feature_column.embedding_column(make, dimension=3),
  ]

  # Build a custom Estimator, using the model_fn.
  # `params` is passed through to the `model_fn`.
  model = tf.estimator.Estimator(
      model_fn=my_dnn_regression_fn,
      params={
          "feature_columns": feature_columns,
          "learning_rate": 0.001,
          "optimizer": tf.train.AdamOptimizer,
          "hidden_units": [20, 20]
      })

  # Train the model.
  model.train(input_fn=train_input_fn, steps=args.train_steps)

  # Evaluate how the model performs on data it has not yet seen.
  eval_result = model.evaluate(input_fn=test_input_fn)

  # Print the Root Mean Square Error (RMSE).
  print("\n" + 80 * "*")
  print("\nRMS error for the test set: ${:.0f}"
        .format(args.price_norm_factor * eval_result["rmse"]))

  print()


if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
