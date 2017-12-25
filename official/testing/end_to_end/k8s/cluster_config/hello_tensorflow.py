from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

#from models.official.resnet.imagenet_test import *

tf.app.flags.DEFINE_integer("task_index", None, "Task index, should be >= 0.")
tf.app.flags.DEFINE_string("job_name", None, "job name: worker or ps")
tf.app.flags.DEFINE_string("ps_hosts", None, "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", None, "Comma-separated list of hostname:port pairs")


def main():
  import sys
  print('System path', sys.path)
  print(os.listdir('/opt/')


if __name__ == '__main__':
  main()
