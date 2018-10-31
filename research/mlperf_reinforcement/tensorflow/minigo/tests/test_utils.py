# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
import re
import time
import unittest

import go
import utils

assert go.N == 9, "All unit tests must be run with BOARD_SIZE=9"


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
    assert len(string) == go.N ** 2, "Board to load didn't have right dimensions"
    board = np.zeros([go.N, go.N], dtype=np.int8)
    for i, char in enumerate(string):
        np.ravel(board)[i] = reverse_map[char]
    return board


class TestUtils(unittest.TestCase):
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


class MiniGoUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start_time = time.time()

    @classmethod
    def tearDownClass(cls):
        print("\n%s.%s: %.3f seconds" %
              (cls.__module__, cls.__name__, time.time() - cls.start_time))

    def assertEqualNPArray(self, array1, array2):
        if not np.all(array1 == array2):
            raise AssertionError(
                "Arrays differed in one or more locations:\n%s\n%s" % (array1, array2))

    def assertNotEqualNPArray(self, array1, array2):
        if np.all(array1 == array2):
            raise AssertionError("Arrays were identical:\n%s" % array1)

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

        remapped_group_index1 = [lt1_mapping.get(
            gid, go.MISSING_GROUP_ID) for gid in lib_tracker1.group_index.ravel().tolist()]
        remapped_group_index2 = [lt2_mapping.get(
            gid, go.MISSING_GROUP_ID) for gid in lib_tracker2.group_index.ravel().tolist()]
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
