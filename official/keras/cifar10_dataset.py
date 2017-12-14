import keras
from keras.datasets import cifar10
import numpy as np
from tensorflow.contrib.data import Dataset

def get_train_eval_dataset():
  num_classes = 10
  # Load the CIFAR10 data.
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  # Normalize data.
  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255

  # Subtracting pixel mean improves accuracy
  subtract_pixel_mean = True

  # If subtract pixel mean is enabled
  if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

  y_train = y_train.astype('float32')
  y_test = y_test.astype('float32')

  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  dataset_x_train = Dataset.from_tensor_slices(x_train)
  dataset_y_train = Dataset.from_tensor_slices(y_train)

  dataset_x_test = Dataset.from_tensor_slices(x_test)
  dataset_y_test = Dataset.from_tensor_slices(y_test)

  return x_train.shape[1:], dataset_x_train, dataset_x_test, dataset_y_train, dataset_y_test