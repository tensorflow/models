from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import gmtime, sleep, strftime

import tensorflow as tf

from models.official.resnet.imagenet_test import *

tf.app.flags.DEFINE_integer("task_index", None, "Task index, should be >= 0.")
tf.app.flags.DEFINE_string("job_name", None, "job name: worker or ps")
tf.app.flags.DEFINE_string("ps_hosts", None, "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", None, "Comma-separated list of hostname:port pairs")


def main():
  print(tf.test.main())

  while True:
    sleep(300)


if __name__ == '__main__':
  main()
