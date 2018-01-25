from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tests for common.utils.
"""

from collections import Counter
import random
import tempfile
import numpy as np
import tensorflow as tf

from common import utils  # brain coder


class UtilsTest(tf.test.TestCase):

  def testStackPad(self):
    # 1D.
    tensors = [[1, 2, 3], [4, 5, 6, 7, 8], [9]]
    result = utils.stack_pad(tensors, pad_axes=0, pad_to_lengths=6)
    self.assertTrue(np.array_equal(
        result,
        np.asarray([[1, 2, 3, 0, 0, 0],
                    [4, 5, 6, 7, 8, 0],
                    [9, 0, 0, 0, 0, 0]], dtype=np.float32)))

    # 3D.
    tensors = [[[[1, 2, 3], [4, 5, 6]]],
               [[[7, 8, 9], [0, 1, 2]], [[3, 4, 5], [6, 7, 8]]],
               [[[0, 1, 2]], [[3, 4, 5]]]]
    result = utils.stack_pad(tensors, pad_axes=[0, 1], pad_to_lengths=[2, 2])
    self.assertTrue(np.array_equal(
        result,
        np.asarray([[[[1, 2, 3], [4, 5, 6]],
                     [[0, 0, 0], [0, 0, 0]]],
                    [[[7, 8, 9], [0, 1, 2]],
                     [[3, 4, 5], [6, 7, 8]]],
                    [[[0, 1, 2], [0, 0, 0]],
                     [[3, 4, 5], [0, 0, 0]]]], dtype=np.float32)))

  def testStackPadNoAxes(self):
    # 2D.
    tensors = [[[1, 2, 3], [4, 5, 6]],
               [[7, 8, 9], [1, 2, 3]],
               [[4, 5, 6], [7, 8, 9]]]
    result = utils.stack_pad(tensors)
    self.assertTrue(np.array_equal(
        result,
        np.asarray(tensors)))

  def testStackPadNoneLength(self):
    # 1D.
    tensors = [[1, 2, 3], [4, 5, 6, 7, 8], [9]]
    result = utils.stack_pad(tensors, pad_axes=0, pad_to_lengths=None)
    self.assertTrue(np.array_equal(
        result,
        np.asarray([[1, 2, 3, 0, 0],
                    [4, 5, 6, 7, 8],
                    [9, 0, 0, 0, 0]], dtype=np.float32)))

    # 3D.
    tensors = [[[[1, 2, 3], [4, 5, 6]]],
               [[[7, 8, 9], [0, 1, 2]], [[3, 4, 5], [6, 7, 8]]],
               [[[0, 1, 2]], [[3, 4, 5]]]]
    result = utils.stack_pad(tensors, pad_axes=[0, 1], pad_to_lengths=None)
    self.assertTrue(np.array_equal(
        result,
        np.asarray([[[[1, 2, 3], [4, 5, 6]],
                     [[0, 0, 0], [0, 0, 0]]],
                    [[[7, 8, 9], [0, 1, 2]],
                     [[3, 4, 5], [6, 7, 8]]],
                    [[[0, 1, 2], [0, 0, 0]],
                     [[3, 4, 5], [0, 0, 0]]]], dtype=np.float32)))

    # 3D with partial pad_to_lengths.
    tensors = [[[[1, 2, 3], [4, 5, 6]]],
               [[[7, 8, 9], [0, 1, 2]], [[3, 4, 5], [6, 7, 8]]],
               [[[0, 1, 2]], [[3, 4, 5]]]]
    result = utils.stack_pad(tensors, pad_axes=[0, 1], pad_to_lengths=[None, 3])
    self.assertTrue(np.array_equal(
        result,
        np.asarray([[[[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
                    [[[7, 8, 9], [0, 1, 2], [0, 0, 0]],
                     [[3, 4, 5], [6, 7, 8], [0, 0, 0]]],
                    [[[0, 1, 2], [0, 0, 0], [0, 0, 0]],
                     [[3, 4, 5], [0, 0, 0], [0, 0, 0]]]], dtype=np.float32)))

  def testStackPadValueError(self):
    # 3D.
    tensors = [[[[1, 2, 3], [4, 5, 6]]],
               [[[7, 8, 9], [0, 1, 2]], [[3, 4, 5], [6, 7, 8]]],
               [[[0, 1, 2]], [[3, 4, 5]]],
               [[[1, 2, 3, 4]]]]

    # Not all tensors have the same shape along axis 2.
    with self.assertRaises(ValueError):
      utils.stack_pad(tensors, pad_axes=[0, 1], pad_to_lengths=[2, 2])

  def testRecord(self):
    my_record = utils.make_record('my_record', ['a', 'b', 'c'], {'b': 55})
    inst = my_record(a=1, b=2, c=3)
    self.assertEqual(1, inst.a)
    self.assertEqual(2, inst.b)
    self.assertEqual(3, inst.c)
    self.assertEqual(1, inst[0])
    self.assertEqual(2, inst[1])
    self.assertEqual(3, inst[2])
    self.assertEqual([1, 2, 3], list(iter(inst)))
    self.assertEqual(3, len(inst))

    inst.b = 999
    self.assertEqual(999, inst.b)
    self.assertEqual(999, inst[1])

    inst2 = my_record(1, 999, 3)
    self.assertTrue(inst == inst2)
    inst2[1] = 3
    self.assertFalse(inst == inst2)

    inst3 = my_record(a=1, c=3)
    inst.b = 55
    self.assertEqual(inst, inst3)

  def testRecordUnique(self):
    record1 = utils.make_record('record1', ['a', 'b', 'c'])
    record2 = utils.make_record('record2', ['a', 'b', 'c'])
    self.assertNotEqual(record1(1, 2, 3), record2(1, 2, 3))
    self.assertEqual(record1(1, 2, 3), record1(1, 2, 3))

  def testTupleToRecord(self):
    my_record = utils.make_record('my_record', ['a', 'b', 'c'])
    inst = utils.tuple_to_record((5, 6, 7), my_record)
    self.assertEqual(my_record(5, 6, 7), inst)

  def testRecordErrors(self):
    my_record = utils.make_record('my_record', ['a', 'b', 'c'], {'b': 10})

    with self.assertRaises(ValueError):
      my_record(c=5)  # Did not provide required argument 'a'.
    with self.assertRaises(ValueError):
      my_record(1, 2, 3, 4)  # Too many arguments.

  def testRandomQueue(self):
    np.random.seed(567890)
    queue = utils.RandomQueue(5)
    queue.push(5)
    queue.push(6)
    queue.push(7)
    queue.push(8)
    queue.push(9)
    queue.push(10)
    self.assertTrue(5 not in queue)
    sample = queue.random_sample(1000)
    self.assertEqual(1000, len(sample))
    self.assertEqual([6, 7, 8, 9, 10], sorted(np.unique(sample).tolist()))

  def testMaxUniquePriorityQueue(self):
    queue = utils.MaxUniquePriorityQueue(5)
    queue.push(1.0, 'string 1')
    queue.push(-0.5, 'string 2')
    queue.push(0.5, 'string 3')
    self.assertEqual((-0.5, 'string 2', None), queue.pop())
    queue.push(0.1, 'string 4')
    queue.push(1.5, 'string 5')
    queue.push(0.0, 'string 6')
    queue.push(0.2, 'string 7')
    self.assertEqual((1.5, 'string 5', None), queue.get_max())
    self.assertEqual((0.1, 'string 4', None), queue.get_min())
    self.assertEqual(
        [('string 5', None), ('string 1', None), ('string 3', None),
         ('string 7', None), ('string 4', None)],
        list(queue.iter_in_order()))

  def testMaxUniquePriorityQueue_Duplicates(self):
    queue = utils.MaxUniquePriorityQueue(5)
    queue.push(0.0, 'string 1')
    queue.push(0.0, 'string 2')
    queue.push(0.0, 'string 3')
    self.assertEqual((0.0, 'string 1', None), queue.pop())
    self.assertEqual((0.0, 'string 2', None), queue.pop())
    self.assertEqual((0.0, 'string 3', None), queue.pop())
    self.assertEqual(0, len(queue))
    queue.push(0.1, 'string 4')
    queue.push(1.5, 'string 5')
    queue.push(0.3, 'string 6')
    queue.push(0.2, 'string 7')
    queue.push(0.0, 'string 8')
    queue.push(1.5, 'string 5')
    queue.push(1.5, 'string 5')
    self.assertEqual((1.5, 'string 5', None), queue.get_max())
    self.assertEqual((0.0, 'string 8', None), queue.get_min())
    self.assertEqual(
        [('string 5', None), ('string 6', None), ('string 7', None),
         ('string 4', None), ('string 8', None)],
        list(queue.iter_in_order()))

  def testMaxUniquePriorityQueue_ExtraData(self):
    queue = utils.MaxUniquePriorityQueue(5)
    queue.push(1.0, 'string 1', [1, 2, 3])
    queue.push(0.5, 'string 2', [4, 5, 6])
    queue.push(0.5, 'string 3', [7, 8, 9])
    queue.push(0.5, 'string 2', [10, 11, 12])
    self.assertEqual((0.5, 'string 2', [4, 5, 6]), queue.pop())
    self.assertEqual((0.5, 'string 3', [7, 8, 9]), queue.pop())
    self.assertEqual((1.0, 'string 1', [1, 2, 3]), queue.pop())
    self.assertEqual(0, len(queue))
    queue.push(0.5, 'string 2', [10, 11, 12])
    self.assertEqual((0.5, 'string 2', [10, 11, 12]), queue.pop())

  def testRouletteWheel(self):
    random.seed(12345678987654321)
    r = utils.RouletteWheel()
    self.assertTrue(r.is_empty())
    with self.assertRaises(RuntimeError):
      r.sample()  # Cannot sample when empty.
    self.assertEqual(0, r.total_weight)
    self.assertEqual(True, r.add('a', 0.1))
    self.assertFalse(r.is_empty())
    self.assertEqual(0.1, r.total_weight)
    self.assertEqual(True, r.add('b', 0.01))
    self.assertEqual(0.11, r.total_weight)
    self.assertEqual(True, r.add('c', 0.5))
    self.assertEqual(True, r.add('d', 0.1))
    self.assertEqual(True, r.add('e', 0.05))
    self.assertEqual(True, r.add('f', 0.03))
    self.assertEqual(True, r.add('g', 0.001))
    self.assertEqual(0.791, r.total_weight)
    self.assertFalse(r.is_empty())

    # Check that sampling is correct.
    obj, weight = r.sample()
    self.assertTrue(isinstance(weight, float), 'Type: %s' % type(weight))
    self.assertTrue((obj, weight) in r)
    for obj, weight in r.sample_many(100):
      self.assertTrue(isinstance(weight, float), 'Type: %s' % type(weight))
      self.assertTrue((obj, weight) in r)

    # Check that sampling distribution is correct.
    n = 1000000
    c = Counter(r.sample_many(n))
    for obj, w in r:
      estimated_w = c[(obj, w)] / float(n) * r.total_weight
      self.assertTrue(
          np.isclose(w, estimated_w, atol=1e-3),
          'Expected %s, got %s, for object %s' % (w, estimated_w, obj))

  def testRouletteWheel_AddMany(self):
    random.seed(12345678987654321)
    r = utils.RouletteWheel()
    self.assertTrue(r.is_empty())
    with self.assertRaises(RuntimeError):
      r.sample()  # Cannot sample when empty.
    self.assertEqual(0, r.total_weight)
    count = r.add_many(
        ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
        [0.1, 0.01, 0.5, 0.1, 0.05, 0.03, 0.001])
    self.assertEqual(7, count)
    self.assertFalse(r.is_empty())
    self.assertEqual(0.791, r.total_weight)

    # Adding no items is allowed.
    count = r.add_many([], [])
    self.assertEqual(0, count)
    self.assertFalse(r.is_empty())
    self.assertEqual(0.791, r.total_weight)

    # Check that sampling is correct.
    obj, weight = r.sample()
    self.assertTrue(isinstance(weight, float), 'Type: %s' % type(weight))
    self.assertTrue((obj, weight) in r)
    for obj, weight in r.sample_many(100):
      self.assertTrue(isinstance(weight, float), 'Type: %s' % type(weight))
      self.assertTrue((obj, weight) in r)

    # Check that sampling distribution is correct.
    n = 1000000
    c = Counter(r.sample_many(n))
    for obj, w in r:
      estimated_w = c[(obj, w)] / float(n) * r.total_weight
      self.assertTrue(
          np.isclose(w, estimated_w, atol=1e-3),
          'Expected %s, got %s, for object %s' % (w, estimated_w, obj))

  def testRouletteWheel_AddZeroWeights(self):
    r = utils.RouletteWheel()
    self.assertEqual(True, r.add('a', 0))
    self.assertFalse(r.is_empty())
    self.assertEqual(4, r.add_many(['b', 'c', 'd', 'e'], [0, 0.1, 0, 0]))
    self.assertEqual(
        [('a', 0.0), ('b', 0.0), ('c', 0.1), ('d', 0.0), ('e', 0.0)],
        list(r))

  def testRouletteWheel_UniqueMode(self):
    random.seed(12345678987654321)
    r = utils.RouletteWheel(unique_mode=True)
    self.assertEqual(True, r.add([1, 2, 3], 1, 'a'))
    self.assertEqual(True, r.add([4, 5], 0.5, 'b'))
    self.assertEqual(False, r.add([1, 2, 3], 1.5, 'a'))
    self.assertEqual(
        [([1, 2, 3], 1.0), ([4, 5], 0.5)],
        list(r))
    self.assertEqual(1.5, r.total_weight)
    self.assertEqual(
        2,
        r.add_many(
            [[5, 6, 2, 3], [1, 2, 3], [8], [1, 2, 3]],
            [0.1, 0.2, 0.1, 2.0],
            ['c', 'a', 'd', 'a']))
    self.assertEqual(
        [([1, 2, 3], 1.0), ([4, 5], 0.5), ([5, 6, 2, 3], 0.1), ([8], 0.1)],
        list(r))
    self.assertTrue(np.isclose(1.7, r.total_weight))
    self.assertEqual(0, r.add_many([], [], []))  # Adding no items is allowed.
    with self.assertRaises(ValueError):
      # Key not given.
      r.add([7, 8, 9], 2.0)
    with self.assertRaises(ValueError):
      # Keys not given.
      r.add_many([[7, 8, 9], [10]], [2.0, 2.0])
    self.assertEqual(True, r.has_key('a'))
    self.assertEqual(True, r.has_key('b'))
    self.assertEqual(False, r.has_key('z'))
    self.assertEqual(1.0, r.get_weight('a'))
    self.assertEqual(0.5, r.get_weight('b'))

    r = utils.RouletteWheel(unique_mode=False)
    self.assertEqual(True, r.add([1, 2, 3], 1))
    self.assertEqual(True, r.add([4, 5], 0.5))
    self.assertEqual(True, r.add([1, 2, 3], 1.5))
    self.assertEqual(
        [([1, 2, 3], 1.0), ([4, 5], 0.5), ([1, 2, 3], 1.5)],
        list(r))
    self.assertEqual(3, r.total_weight)
    self.assertEqual(
        4,
        r.add_many(
            [[5, 6, 2, 3], [1, 2, 3], [8], [1, 2, 3]],
            [0.1, 0.2, 0.1, 0.2]))
    self.assertEqual(
        [([1, 2, 3], 1.0), ([4, 5], 0.5), ([1, 2, 3], 1.5),
         ([5, 6, 2, 3], 0.1), ([1, 2, 3], 0.2), ([8], 0.1), ([1, 2, 3], 0.2)],
        list(r))
    self.assertTrue(np.isclose(3.6, r.total_weight))
    with self.assertRaises(ValueError):
      # Key is given.
      r.add([7, 8, 9], 2.0, 'a')
    with self.assertRaises(ValueError):
      # Keys are given.
      r.add_many([[7, 8, 9], [10]], [2.0, 2.0], ['a', 'b'])

  def testRouletteWheel_IncrementalSave(self):
    f = tempfile.NamedTemporaryFile()
    r = utils.RouletteWheel(unique_mode=True, save_file=f.name)
    entries = [
        ([1, 2, 3], 0.1, 'a'),
        ([4, 5], 0.2, 'b'),
        ([6], 0.3, 'c'),
        ([7, 8, 9, 10], 0.25, 'd'),
        ([-1, -2], 0.15, 'e'),
        ([-3, -4, -5], 0.5, 'f')]

    self.assertTrue(r.is_empty())
    for i in range(0, len(entries), 2):
      r.add(*entries[i])
      r.add(*entries[i + 1])
      r.incremental_save()

      r2 = utils.RouletteWheel(unique_mode=True, save_file=f.name)
      self.assertEqual(i + 2, len(r2))
      count = 0
      for j, (obj, weight) in enumerate(r2):
        self.assertEqual(entries[j][0], obj)
        self.assertEqual(entries[j][1], weight)
        self.assertEqual(weight, r2.get_weight(entries[j][2]))
        count += 1
      self.assertEqual(i + 2, count)

if __name__ == '__main__':
  tf.test.main()
