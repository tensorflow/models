from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

#import tensorflow as tf

#tf.app.flags.DEFINE_integer("task_index", None, "Task index, should be >= 0.")
#tf.app.flags.DEFINE_string("job_name", None, "job name: worker or ps")
#tf.app.flags.DEFINE_string("ps_hosts", None, "Comma-separated list of hostname:port pairs")
#tf.app.flags.DEFINE_string("worker_hosts", None, "Comma-separated list of hostname:port pairs")
#tf.app.flags.DEFINE_string("data_dir", None, "Data dir to store data for testing.")


def main():
  print('LD_LIBRARY_PATH', os.environ['LD_LIBRARY_PATH'])
  import tensorflow as tf


if __name__ == '__main__':
  main()
