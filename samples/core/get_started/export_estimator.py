#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of DNNClassifier for Iris plant dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf

import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100,
                    type=int, help='batch size')
parser.add_argument('--train_steps', default=200,
                    type=int, help='number of training steps')
parser.add_argument('--export_dir', type=str, default="saved_model_exports",
                    help="Base directory to export the model to.")
parser.add_argument('--npy_path', type=str, default="inputs.npy",
                    help="The path to save the .npy file, for the "
                         "saved_model_cli")


def dataset_input_fn(ds):
    return lambda: ds.make_one_shot_iterator().get_next()


def my_receiver():
    # The placeholder is where the parent program will write its input.
    csv_input = tf.placeholder(tf.string, (None,))

    feature_keys = iris_data.COLUMNS[:-1]
    csv_defaults = [[0.0]]*(len(feature_keys))

    # Build the feature dictionary from the parsed csv string.
    parsed_fields = tf.decode_csv(csv_input, csv_defaults)
    my_features = {}
    for key, field in zip(feature_keys, parsed_fields):
        my_features[key] = field

    # return the two pipeline ends in a ServingInputReceiver
    return tf.estimator.export.ServingInputReceiver(
        my_features, csv_input)


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    train_ds, test_ds = iris_data.datasets()

    # Feature columns describe the input: all columns are numeric.
    feature_columns = [tf.feature_column.numeric_column(col_name)
                       for col_name in iris_data.COLUMNS[:-1]]

    # Build 3 layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=3)

    # Train the Model.
    train_input = (
        train_ds
        .repeat()
        .shuffle(1000)
        .batch(args.batch_size))

    classifier.train(input_fn=dataset_input_fn(train_input),
                     steps=args.train_steps)

    savedmodel_path = classifier.export_savedmodel(
        export_dir_base=args.export_dir,
        serving_input_receiver_fn=my_receiver)

    print("\nmodel exported to:\n    "+savedmodel_path)

    make_npy(args.npy_path)
    print("\nsaved_model_cli input file saved to:\n    " + args.npy_path)
    print()


def make_npy(npy_path):
    # Load the data as numpy arrays.
    ((train_x, train_y), (test_x, test_y)) = iris_data.load_data()

    # Save the test data in .npy files for the saved_model_cli.
    test_x = np.array(test_x)
    test_x_str = []
    for row in test_x:
        csv_row = ','.join(row.astype(str))
        test_x_str.append(csv_row)

    np.save(npy_path, np.array(test_x_str))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
