import numpy as np
import tensorflow as tf
import functools
import matplotlib.pyplot as plt


class Dataset():
  def __init__(self,
               data_path='/tmp/merge_2006_2012.csv',
               batch_size=128,
               data_points_day=144,
               num_days=10,
               steps=6,
               mode='train'):
    self.num_days = num_days
    self.data_points_day = data_points_day
    self.data_path = data_path
    # Look back 10 days
    self.lookback = data_points_day * num_days
    # observations once every hour
    self.steps = steps
    # We will target one day in the future (24 hours)
    self.delay = data_points_day
    self.batch_size = batch_size
    if mode == 'inference':
      print('Setting batch size to 1 in inference mode')
      self.batch_size = 1

  def _preprocess_dataset_element(self, idx, data, lookback, steps, delay):
    """fn to return the actual data given a specific index"""
    # We map this function onto individual elements of the dataset
    sample = data[idx - lookback: idx: steps]
    target = tf.expand_dims(data[idx + delay][1], -1)
    return sample, target

  def get_dataset(self,
                  data,
                  shuffle=False):
    """
    Returns a tf.data.Dataset object containing the data associated with the
    jena temperature data

    Arguments:
      data:  raw normalized data
      shuffle:  boolean true false for shuffling the data or not

    Returns:
      samples of size (batch, lookback // steps, num features),
             targets of size (batch,)
    """
    num_x = len(data)
    idxs = np.arange(self.lookback, num_x - self.delay)
    dataset = tf.data.Dataset.from_tensor_slices((idxs))
    cfg = {
      'data': tf.constant(data, dtype=tf.float32),
      'lookback': self.lookback,
      'steps': self.steps,
      'delay': self.delay,
    }
    preproc_fn = functools.partial(self._preprocess_dataset_element, **cfg)

    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(preproc_fn, num_parallel_calls=5)
    if shuffle:
      dataset = dataset.shuffle(num_x)
    # It's necessary to repeat our data for all epochs
    dataset = dataset.repeat().batch(self.batch_size).prefetch(1)
    return dataset

  def get_raw(self):
    """
    Loads the raw csv and returns the data within the csv without the header
    """
    with open(self.data_path) as f:
      data = f.read().rstrip().split('\n')

    header = data[0].split(',')
    raw_data = data[1:]

    print(header)
    print(raw_data[0])
    print(len(raw_data))
    return header, raw_data

  def load_jena_data(self, plot=False):
    """fn to return the train, val, and test tf.data.Datasets """
    header, raw_data = self.get_raw()
    all_data = np.zeros((len(raw_data), len(header) - 1))
    for i, line in enumerate(raw_data):
      values = [float(x) for x in line.split(',')[1:]]
      all_data[i, :] = values

    if plot:
      temp_data = all_data[:, 1]
      plt.plot(range(len(temp_data)), temp_data)
      plt.show()

      plt.plot(range(self.data_points_day * self.num_days),
               temp_data[:self.data_points_day * self.num_days])
      plt.title("First 10 days of Temperature Data")
      plt.xlabel("Time (hours)")
      plt.ylabel("Temperature (Degrees Celsius)")
      plt.show()

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    np.testing.assert_almost_equal((train_ratio + val_ratio + test_ratio), 1.0)

    num_total = len(all_data)
    self.num_train = int(num_total * train_ratio)
    self.num_val = int(num_total * val_ratio)
    self.num_test = num_total - (self.num_train + self.num_val)

    print("Number of total examples {}".format(num_total))
    print("Number of training examples {}".format(self.num_train))
    print("Number of validation examples {}".format(self.num_val))
    print("Number of testing examples {}".format(self.num_test))

    # Normalization and other shit
    mean = all_data[:self.num_train].mean(axis=0)
    std = all_data[:self.num_train].std(axis=0)

    # Normalize all the data by the training mean and standard deviation
    all_data -= mean
    all_data /= std

    self.x_train = all_data[:self.num_train]
    self.x_val = all_data[self.num_train:self.num_train + self.num_val]
    self.x_test = all_data[self.num_train + self.num_val:]

    train_ds = self.get_dataset(self.x_train, shuffle=True)
    val_ds = self.get_dataset(self.x_val)
    test_ds = self.get_dataset(self.x_test)

    return train_ds, val_ds, test_ds

