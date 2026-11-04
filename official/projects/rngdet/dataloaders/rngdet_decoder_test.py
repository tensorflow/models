from absl import app  # pylint:disable=unused-import
import tensorflow as tf
from official.projects.rngdet.dataloaders import rngdet_input


def main(_):
  raw_dataset_train = tf.data.TFRecordDataset(
    '/home/mjyun/cityscale/tfrecord/train-noise-8-00010-of-00032.tfrecord')
  decoder = rngdet_input.Decoder()
  data_set_train = raw_dataset_train.map(decoder.decode)
  data_sub = data_set_train.take(1)
  for i in data_sub: 
    print(i['gt_probs'])


if __name__ == '__main__':
  app.run(main)