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
"""Tests for utils, and base class for other unit tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import re
import tempfile
import time

import tensorflow as tf  # pylint: disable=g-bad-import-order

import go
import numpy as np
import utils

tf.logging.set_verbosity(tf.logging.ERROR)

BOARD_SIZE = 9
EMPTY_BOARD = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.int8)
ALL_COORDS = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]


def _check_bounds(c):
  return c[0] % BOARD_SIZE == c[0] and c[1] % BOARD_SIZE == c[1]

NEIGHBORS = {(x, y): list(filter(_check_bounds, [
    (x+1, y), (x-1, y), (x, y+1), (x, y-1)])) for x, y in ALL_COORDS}


def load_board(string):
  reverse_map = {
      'X': go.BLACK,
      'O': go.WHITE,
      '.': go.EMPTY,
      '#': go.FILL,
      '*': go.KO,
      '?': go.UNKNOWN
  }
  string = re.sub(r'[^XO\.#]+', '', string)
  if len(string) != BOARD_SIZE ** 2:
    raise ValueError("Board to load didn't have right dimensions")
  board = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.int8)
  for ii, char in enumerate(string):
    np.ravel(board)[ii] = reverse_map[char]
  return board


class TestUtils(tf.test.TestCase):

  def test_bootstrap_name(self):
    name = utils.generate_model_name(0)
    self.assertIn('bootstrap', name)

  def test_generate_model_name(self):
    name = utils.generate_model_name(17)
    self.assertIn('000017', name)

  def test_detect_name(self):
    string = '000017-model.index'
    detected_name = utils.detect_model_name(string)
    self.assertEqual(detected_name, '000017-model')

  def test_detect_num(self):
    string = '000017-model.index'
    detected_name = utils.detect_model_num(string)
    self.assertEqual(detected_name, 17)

  def test_get_models(self):
    with tempfile.TemporaryDirectory() as models_dir:
      model1 = '000013-model.meta'
      model2 = '000017-model.meta'
      f1 = open(os.path.join(models_dir, model1), 'w')
      f1.close()
      f2 = open(os.path.join(models_dir, model2), 'w')
      f2.close()
      model_nums_names = utils.get_models(models_dir)
      self.assertEqual(len(model_nums_names), 2)
      self.assertEqual(model_nums_names[0], (13, '000013-model'))
      self.assertEqual(model_nums_names[1], (17, '000017-model'))

  def test_get_latest_model(self):
    with tempfile.TemporaryDirectory() as models_dir:
      model1 = '000013-model.meta'
      model2 = '000017-model.meta'
      f1 = open(os.path.join(models_dir, model1), 'w')
      f1.close()
      f2 = open(os.path.join(models_dir, model2), 'w')
      f2.close()
      latest_model = utils.get_latest_model(models_dir)
      self.assertEqual(latest_model, (17, '000017-model'))

  def test_round_power_of_two(self):
    self.assertEqual(utils.round_power_of_two(84), 64)
    self.assertEqual(utils.round_power_of_two(120), 128)

  def test_shuffler(self):
    random.seed(1)
    dataset = (i for i in range(10))
    shuffled = list(utils.shuffler(
        dataset, pool_size=5, refill_threshold=0.8))
    self.assertEqual(len(shuffled), 10)
    self.assertNotEqual(shuffled, list(range(10)))

  def test_parse_game_result(self):
    self.assertEqual(utils.parse_game_result('B+3.5'), go.BLACK)
    self.assertEqual(utils.parse_game_result('W+T'), go.WHITE)
    self.assertEqual(utils.parse_game_result('Void'), 0)


class MiniGoUnitTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.start_time = time.time()

  @classmethod
  def tearDownClass(cls):
    print('\n%s.%s: %.3f seconds' %
          (cls.__module__, cls.__name__, time.time() - cls.start_time))

  def assertEqualNPArray(self, array1, array2):
    if not np.all(array1 == array2):
      raise AssertionError(
          'Arrays differed in one or more locations:\n%s\n%s' % (array1, array2)
      )

  def assertNotEqualNPArray(self, array1, array2):
    if np.all(array1 == array2):
      raise AssertionError('Arrays were identical:\n%s' % array1)

  def assertEqualLibTracker(self, lib_tracker1, lib_tracker2):
    # A lib tracker may have differently numbered groups yet still
    # represent the same set of groups.
    # "Sort" the group_ids to ensure they are the same.
    def find_group_mapping(lib_tracker):
      current_gid = 0
      mapping = {}
      for group_id in lib_tracker.group_index.ravel().tolist():
        if group_id == go.MISSING_GROUP_ID:
          continue
        if group_id not in mapping:
          mapping[group_id] = current_gid
          current_gid += 1
      return mapping

    lt1_mapping = find_group_mapping(lib_tracker1)
    lt2_mapping = find_group_mapping(lib_tracker2)

    remapped_group_index1 = [
        lt1_mapping.get(gid, go.MISSING_GROUP_ID)
        for gid in lib_tracker1.group_index.ravel().tolist()]
    remapped_group_index2 = [
        lt2_mapping.get(gid, go.MISSING_GROUP_ID)
        for gid in lib_tracker2.group_index.ravel().tolist()]
    self.assertEqual(remapped_group_index1, remapped_group_index2)

    remapped_groups1 = {lt1_mapping.get(
        gid): group for gid, group in lib_tracker1.groups.items()}
    remapped_groups2 = {lt2_mapping.get(
        gid): group for gid, group in lib_tracker2.groups.items()}
    self.assertEqual(remapped_groups1, remapped_groups2)

    self.assertEqualNPArray(
        lib_tracker1.liberty_cache, lib_tracker2.liberty_cache)

  def assertEqualPositions(self, pos1, pos2):
    self.assertEqualNPArray(pos1.board, pos2.board)
    self.assertEqualLibTracker(pos1.lib_tracker, pos2.lib_tracker)
    self.assertEqual(pos1.n, pos2.n)
    self.assertEqual(pos1.caps, pos2.caps)
    self.assertEqual(pos1.ko, pos2.ko)
    r_len = min(len(pos1.recent), len(pos2.recent))
    if r_len > 0:  # if a position has no history, then don't bother testing
      self.assertEqual(pos1.recent[-r_len:], pos2.recent[-r_len:])
    self.assertEqual(pos1.to_play, pos2.to_play)

  def assertNoPendingVirtualLosses(self, root):
    """Raise an error if any node in this subtree has vlosses pending."""
    queue = [root]
    while queue:
      current = queue.pop()
      self.assertEqual(current.losses_applied, 0)
      queue.extend(current.children.values())


if __name__ == '__main__':
  tf.test.main()
