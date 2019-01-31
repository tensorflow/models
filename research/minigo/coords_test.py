# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for coords."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

import coords
import numpy
import utils_test

tf.logging.set_verbosity(tf.logging.ERROR)


class TestCoords(utils_test.MiniGoUnitTest):

  def test_upperleft(self):
    self.assertEqual(coords.from_sgf('aa'), (0, 0))
    self.assertEqual(coords.from_flat(utils_test.BOARD_SIZE, 0), (0, 0))
    self.assertEqual(coords.from_kgs(utils_test.BOARD_SIZE, 'A9'), (0, 0))
    self.assertEqual(coords.from_pygtp(utils_test.BOARD_SIZE, (1, 9)), (0, 0))

    self.assertEqual(coords.to_sgf((0, 0)), 'aa')
    self.assertEqual(coords.to_flat(utils_test.BOARD_SIZE, (0, 0)), 0)
    self.assertEqual(coords.to_kgs(utils_test.BOARD_SIZE, (0, 0)), 'A9')
    self.assertEqual(coords.to_pygtp(utils_test.BOARD_SIZE, (0, 0)), (1, 9))

  def test_topleft(self):
    self.assertEqual(coords.from_sgf('ia'), (0, 8))
    self.assertEqual(coords.from_flat(utils_test.BOARD_SIZE, 8), (0, 8))
    self.assertEqual(coords.from_kgs(utils_test.BOARD_SIZE, 'J9'), (0, 8))
    self.assertEqual(coords.from_pygtp(utils_test.BOARD_SIZE, (9, 9)), (0, 8))

    self.assertEqual(coords.to_sgf((0, 8)), 'ia')
    self.assertEqual(coords.to_flat(utils_test.BOARD_SIZE, (0, 8)), 8)
    self.assertEqual(coords.to_kgs(utils_test.BOARD_SIZE, (0, 8)), 'J9')
    self.assertEqual(coords.to_pygtp(utils_test.BOARD_SIZE, (0, 8)), (9, 9))

  def test_pass(self):
    self.assertEqual(coords.from_sgf(''), None)
    self.assertEqual(coords.from_flat(utils_test.BOARD_SIZE, 81), None)
    self.assertEqual(coords.from_kgs(utils_test.BOARD_SIZE, 'pass'), None)
    self.assertEqual(coords.from_pygtp(utils_test.BOARD_SIZE, (0, 0)), None)

    self.assertEqual(coords.to_sgf(None), '')
    self.assertEqual(coords.to_flat(utils_test.BOARD_SIZE, None), 81)
    self.assertEqual(coords.to_kgs(utils_test.BOARD_SIZE, None), 'pass')
    self.assertEqual(coords.to_pygtp(utils_test.BOARD_SIZE, None), (0, 0))

  def test_parsing_9x9(self):
    self.assertEqual(coords.from_sgf('aa'), (0, 0))
    self.assertEqual(coords.from_sgf('ac'), (2, 0))
    self.assertEqual(coords.from_sgf('ca'), (0, 2))
    self.assertEqual(coords.from_sgf(''), None)
    self.assertEqual(coords.to_sgf(None), '')
    self.assertEqual('aa', coords.to_sgf(coords.from_sgf('aa')))
    self.assertEqual('sa', coords.to_sgf(coords.from_sgf('sa')))
    self.assertEqual((1, 17), coords.from_sgf(coords.to_sgf((1, 17))))
    self.assertEqual(coords.from_kgs(utils_test.BOARD_SIZE, 'A1'), (8, 0))
    self.assertEqual(coords.from_kgs(utils_test.BOARD_SIZE, 'A9'), (0, 0))
    self.assertEqual(coords.from_kgs(utils_test.BOARD_SIZE, 'C2'), (7, 2))
    self.assertEqual(coords.from_kgs(utils_test.BOARD_SIZE, 'J2'), (7, 8))
    self.assertEqual(coords.from_pygtp(utils_test.BOARD_SIZE, (1, 1)), (8, 0))
    self.assertEqual(coords.from_pygtp(utils_test.BOARD_SIZE, (1, 9)), (0, 0))
    self.assertEqual(coords.from_pygtp(utils_test.BOARD_SIZE, (3, 2)), (7, 2))
    self.assertEqual(coords.to_pygtp(utils_test.BOARD_SIZE, (8, 0)), (1, 1))
    self.assertEqual(coords.to_pygtp(utils_test.BOARD_SIZE, (0, 0)), (1, 9))
    self.assertEqual(coords.to_pygtp(utils_test.BOARD_SIZE, (7, 2)), (3, 2))

    self.assertEqual(coords.to_kgs(utils_test.BOARD_SIZE, (0, 8)), 'J9')
    self.assertEqual(coords.to_kgs(utils_test.BOARD_SIZE, (8, 0)), 'A1')

  def test_flatten(self):
    self.assertEqual(coords.to_flat(utils_test.BOARD_SIZE, (0, 0)), 0)
    self.assertEqual(coords.to_flat(utils_test.BOARD_SIZE, (0, 3)), 3)
    self.assertEqual(coords.to_flat(utils_test.BOARD_SIZE, (3, 0)), 27)
    self.assertEqual(coords.from_flat(utils_test.BOARD_SIZE, 27), (3, 0))
    self.assertEqual(coords.from_flat(utils_test.BOARD_SIZE, 10), (1, 1))
    self.assertEqual(coords.from_flat(utils_test.BOARD_SIZE, 80), (8, 8))
    self.assertEqual(coords.to_flat(
        utils_test.BOARD_SIZE, coords.from_flat(utils_test.BOARD_SIZE, 10)), 10)
    self.assertEqual(coords.from_flat(
        utils_test.BOARD_SIZE, coords.to_flat(
            utils_test.BOARD_SIZE, (5, 4))), (5, 4))

  def test_from_flat_ndindex_equivalence(self):
    ndindices = list(numpy.ndindex(
        utils_test.BOARD_SIZE, utils_test.BOARD_SIZE))
    flat_coords = list(range(
        utils_test.BOARD_SIZE * utils_test.BOARD_SIZE))
    def _from_flat(flat_coords):
      return coords.from_flat(utils_test.BOARD_SIZE, flat_coords)
    self.assertEqual(
        list(map(_from_flat, flat_coords)), ndindices)


if __name__ == '__main__':
  tf.test.main()
