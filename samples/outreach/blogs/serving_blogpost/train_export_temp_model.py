"""This script will load in the data, train your model, and then export and save
it.

Typical usage example:

    python train_export_temp_model.py /tmp/temp_predict/ --model_version=1 --data_path='/tmp/merge_2006_2012.csv'
"""

import argparse
import data_utils
import models
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("output_path",
                    help="Path to the export location of our saved model")
parser.add_argument("--model_version",
                    type=int,
                    help="What model version are you storing")
parser.add_argument("--data_path",
                    help="Path to the data csv file",
                    default='/tmp/merge_2006_2012.csv')
parser.add_argument("--batch_size", type=int, help="batch size", default=128)
parser.add_argument("--data_points_day",
                    type=int,
                    help="Number of data points in a day (24 hrs) "
                         "default: sample rate = 1/10min = 144 data points/day",
                    default=144)
parser.add_argument("--num_days",
                    type=int,
                    help="Number of days involved in lookback",
                    default=10)
parser.add_argument("--steps",
                    type=int,
                    help="Sample from datastream at this number of steps. "
                         "Default: sample rate = 1/10min = 6 steps to sample at 1/hr",
                    default=6)
args = parser.parse_args()


def main(_):
  print(args)
  print("Loading data")
  # Load the data
  ds = data_utils.Dataset(args.data_path,
                          args.batch_size,
                          args.data_points_day,
                          args.num_days,
                          args.steps,
                          'train')
  train_ds, val_ds, test_ds = ds.load_jena_data()

  print("Building model")
  # Build our model
  model = models.GRU_stack(ds.x_train.shape[-1])
  model.summary()

  print("Training")
  save_model_path = '/tmp/jena_weights_{}.hdf5'.format(os.path.basename(args.data_path))
  cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path,
                                          monitor='val_loss',
                                          save_best_only=True,
                                          verbose=1)
  # Train our model
  history = model.fit(train_ds,
                      steps_per_epoch=int(np.ceil(ds.num_train / float(args.batch_size))),
                      epochs=10,
                      validation_data=val_ds,
                      validation_steps=int(np.ceil(ds.num_val / float(args.batch_size))),
                      callbacks=[cp])

  print("Loading best model")
  model = load_model(save_model_path)

  # Evaluate our model
  loss = model.evaluate(test_ds, steps=int(np.ceil(ds.num_test / float(args.batch_size))))
  print("Model loss: {}".format(loss))

  print("Starting export process")
  tensor_info_x = tf.saved_model.utils.build_tensor_info(model.input)
  tensor_info_y = tf.saved_model.utils.build_tensor_info(model.output)

  prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'weather_data': tensor_info_x},
      outputs={'temperature': tensor_info_y},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

  export_path = os.path.join(
    tf.compat.as_bytes(args.output_path),
    tf.compat.as_bytes(str(args.model_version)))

  print('Exporting trained model to', export_path)
  sess = K.get_session()

  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

  builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
        prediction_signature,
    },
    legacy_init_op=legacy_init_op)
  builder.save()
  print("Done exporting")


if __name__ == '__main__':
  tf.app.run()