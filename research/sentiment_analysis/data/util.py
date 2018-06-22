import numpy as np
import tensorflow as tf

START_CHAR = 1
END_CHAR = 2
OOV_CHAR = 3


def pad_sentence(sen, sentence_length):
  sen = sen[:sentence_length]
  if len(sen) < sentence_length:
    sen = np.pad(sen, (0, sentence_length - len(sen)), "constant",
                 constant_values=(START_CHAR, END_CHAR))
  return sen


def to_dataset(x, y, batch_size, repeat):
  dataset = tf.data.Dataset.from_tensor_slices((x, y))

  # Repeat and batch the dataset
  dataset = dataset.repeat(repeat)
  dataset = dataset.batch(batch_size)

  # Prefetch to improve speed of input pipeline.
  dataset = dataset.prefetch(10)
  return dataset
