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
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=200, type=int,
                    help='number of training steps')

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

COLUMNS = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Sentosa', 'Versicolor', 'Virginica']


def load_data(train_fraction=0.8, seed=0, y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    train = pd.read_csv(train_path, names=COLUMNS, header=0)
    train_x, train_y = train, train.pop(y_name)

    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=COLUMNS, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def make_dataset(*inputs):
    return tf.data.Dataset.from_tensor_slices(inputs)


def from_dataset(ds):
    return lambda: ds.make_one_shot_iterator().get_next()


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = load_data()
    train_x = dict(train_x)
    test_x = dict(test_x)

    # Feature columns describe the input: all columns are numeric.
    feature_columns = [tf.feature_column.numeric_column(col_name)
                       for col_name in COLUMNS[:-1]]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3)

    # Train the Model.
    train = (
        make_dataset(train_x, train_y)
        .repeat()
        .shuffle(1000)
        .batch(args.batch_size))
    classifier.train(input_fn=from_dataset(train), steps=args.train_steps)

    # Evaluate the model.
    test = make_dataset(test_x, test_y).batch(args.batch_size)
    eval_result = classifier.evaluate(input_fn=from_dataset(test))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    predict_input = make_dataset({
        'SepalLength': [6.4, 5.8],
        'SepalWidth': [3.2, 3.1],
        'PetalLength': [4.5, 5.0],
        'PetalWidth': [1.5, 1.7],
    }).batch(args.batch_size)

    for p in classifier.predict(input_fn=from_dataset(predict_input)):
        template = ('Prediction is "{}" ({:.1f}%)')

        class_id = p['class_ids'][0]
        probability = p['probabilities'][class_id]
        print(template.format(SPECIES[class_id], 100 * probability))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
