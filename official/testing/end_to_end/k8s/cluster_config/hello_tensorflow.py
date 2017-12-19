from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


tf.app.flags.DEFINE_integer("task_index", None, "Task index, should be >= 0.")
tf.app.flags.DEFINE_string("job_name", None, "job name: worker or ps")
tf.app.flags.DEFINE_string("ps_hosts", None, "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", None, "Comma-separated list of hostname:port pairs")


def main():
  s = tf.constant("Hello, Tensorflow!")
  print(tf.Session().run(s))
  import os
  print('git repo', os.listdir('/opt/tf-models'))
  print('cuda', os.listdir('/usr/lib/cuda'))
  print('nvidia', os.listdir('/usr/lib/nvidia'))
  print('gnu', os.listdir('/usr/lib/x86_64-linux-gnu'))
  print('ld lib path', os.environ['LD_LIBRARY_PATH'])
  import time
  time.sleep(300)


if __name__ == '__main__':
  main()
