from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import tensorflow as tf

import models


tf.app.flags.DEFINE_integer("task_index", None, "Task index, should be >= 0.")
tf.app.flags.DEFINE_string("job_name", None, "job name: worker or ps")
tf.app.flags.DEFINE_string("ps_hosts", None, "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", None, "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("data_dir", None, "Data dir to store data for testing.")


def main(args):
  import sys
  print('System path', sys.path)
  print(os.listdir(FLAGS.data_dir))
  models.official.resnet.cifar10_download_and_extract(FLAGS.data_dir)
  models.official.resnet.cifar10_main.main()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]])
