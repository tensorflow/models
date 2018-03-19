from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tests for results_lib."""

import contextlib
import os
import shutil
import tempfile
from six.moves import xrange
import tensorflow as tf

from single_task import results_lib  # brain coder


@contextlib.contextmanager
def temporary_directory(suffix='', prefix='tmp', base_path=None):
  """A context manager to create a temporary directory and clean up on exit.

  The parameters are the same ones expected by tempfile.mkdtemp.
  The directory will be securely and atomically created.
  Everything under it will be removed when exiting the context.

  Args:
    suffix: optional suffix.
    prefix: options prefix.
    base_path: the base path under which to create the temporary directory.
  Yields:
    The absolute path of the new temporary directory.
  """
  temp_dir_path = tempfile.mkdtemp(suffix, prefix, base_path)
  try:
    yield temp_dir_path
  finally:
    try:
      shutil.rmtree(temp_dir_path)
    except OSError as e:
      if e.message == 'Cannot call rmtree on a symbolic link':
        # Interesting synthetic exception made up by shutil.rmtree.
        # Means we received a symlink from mkdtemp.
        # Also means must clean up the symlink instead.
        os.unlink(temp_dir_path)
      else:
        raise


def freeze(dictionary):
  """Convert dict to hashable frozenset."""
  return frozenset(dictionary.iteritems())


class ResultsLibTest(tf.test.TestCase):

  def testResults(self):
    with temporary_directory() as logdir:
      results_obj = results_lib.Results(logdir)
      self.assertEqual(results_obj.read_this_shard(), [])
      results_obj.append(
          {'foo': 1.5, 'bar': 2.5, 'baz': 0})
      results_obj.append(
          {'foo': 5.5, 'bar': -1, 'baz': 2})
      self.assertEqual(
          results_obj.read_this_shard(),
          [{'foo': 1.5, 'bar': 2.5, 'baz': 0},
           {'foo': 5.5, 'bar': -1, 'baz': 2}])

  def testShardedResults(self):
    with temporary_directory() as logdir:
      n = 4  # Number of shards.
      results_objs = [
          results_lib.Results(logdir, shard_id=i) for i in xrange(n)]
      for i, robj in enumerate(results_objs):
        robj.append({'foo': i, 'bar': 1 + i * 2})
      results_list, _ = results_objs[0].read_all()

      # Check results. Order does not matter here.
      self.assertEqual(
          set(freeze(r) for r in results_list),
          set(freeze({'foo': i, 'bar': 1 + i * 2}) for i in xrange(n)))


if __name__ == '__main__':
  tf.test.main()
