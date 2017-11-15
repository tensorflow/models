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
import tensorflow as tf

import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def dataset_input_fn(ds):
    return lambda: ds.make_one_shot_iterator().get_next()

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

    # Evaluate the model.
    test_input = test_ds.batch(args.batch_size)
    eval_result = classifier.evaluate(input_fn=dataset_input_fn(test_input))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    predict_input = iris_data.make_dataset({
        'SepalLength': [6.4, 5.8],
        'SepalWidth': [3.2, 3.1],
        'PetalLength': [4.5, 5.0],
        'PetalWidth': [1.5, 1.7],
    }).batch(args.batch_size)


    for p in classifier.predict(input_fn=dataset_input_fn(predict_input)):
        template = ('\nPrediction is "{species}" ({percent:.1f}%)')

        class_id = p['class_ids'][0]
        probability = p['probabilities'][class_id]
        print(template.format(species=iris_data.SPECIES[class_id],
                              percent=100 * probability))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
