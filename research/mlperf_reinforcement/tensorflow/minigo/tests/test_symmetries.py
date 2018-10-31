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

import itertools
import numpy as np

import coords
import symmetries
from symmetries import apply_symmetry_feat as apply_f
from symmetries import apply_symmetry_pi as apply_p
import go

from tests import test_utils


class TestSymmetryOperations(test_utils.MiniGoUnitTest):
    def setUp(self):
        np.random.seed(1)
        self.feat = np.random.random([go.N, go.N, 3])
        self.pi = np.random.random([go.N ** 2 + 1])
        super().setUp()

    def test_inversions(self):
        for s in symmetries.SYMMETRIES:
            with self.subTest(symmetry=s):
                self.assertEqualNPArray(self.feat,
                                        apply_f(s, apply_f(symmetries.invert_symmetry(s), self.feat)))
                self.assertEqualNPArray(self.feat,
                                        apply_f(symmetries.invert_symmetry(s), apply_f(s, self.feat)))

                self.assertEqualNPArray(self.pi,
                                        apply_p(s, apply_p(symmetries.invert_symmetry(s), self.pi)))
                self.assertEqualNPArray(self.pi,
                                        apply_p(symmetries.invert_symmetry(s), apply_p(s, self.pi)))

    def test_compositions(self):
        test_cases = [
            ('rot90', 'rot90', 'rot180'),
            ('rot90', 'rot180', 'rot270'),
            ('identity', 'rot90', 'rot90'),
            ('fliprot90', 'rot90', 'fliprot180'),
            ('rot90', 'rot270', 'identity'),
        ]
        for s1, s2, composed in test_cases:
            with self.subTest(s1=s1, s2=s2, composed=composed):
                self.assertEqualNPArray(apply_f(composed, self.feat),
                                        apply_f(s2, apply_f(s1, self.feat)))
                self.assertEqualNPArray(apply_p(composed, self.pi),
                                        apply_p(s2, apply_p(s1, self.pi)))

    def test_uniqueness(self):
        all_symmetries_f = [
            apply_f(s, self.feat) for s in symmetries.SYMMETRIES
        ]
        all_symmetries_pi = [
            apply_p(s, self.pi) for s in symmetries.SYMMETRIES
        ]
        for f1, f2 in itertools.combinations(all_symmetries_f, 2):
            self.assertNotEqualNPArray(f1, f2)
        for pi1, pi2 in itertools.combinations(all_symmetries_pi, 2):
            self.assertNotEqualNPArray(pi1, pi2)

    def test_proper_move_transform(self):
        # Check that the reinterpretation of 362 = 19*19 + 1 during symmetry
        # application is consistent with coords.from_flat
        move_array = np.arange(go.N ** 2 + 1)
        coord_array = np.zeros([go.N, go.N])
        for c in range(go.N ** 2):
            coord_array[coords.from_flat(c)] = c
        for s in symmetries.SYMMETRIES:
            with self.subTest(symmetry=s):
                transformed_moves = apply_p(s, move_array)
                transformed_board = apply_f(s, coord_array)
                for new_coord, old_coord in enumerate(transformed_moves[:-1]):
                    self.assertEqual(
                        old_coord,
                        transformed_board[coords.from_flat(new_coord)])
