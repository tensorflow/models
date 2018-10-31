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

import unittest
import numpy

import coords
import go
from tests import test_utils


class TestCoords(test_utils.MiniGoUnitTest):
    def test_upperleft(self):
        self.assertEqual(coords.from_sgf('aa'), (0, 0))
        self.assertEqual(coords.from_flat(0), (0, 0))
        self.assertEqual(coords.from_kgs('A9'), (0, 0))
        self.assertEqual(coords.from_pygtp((1, 9)), (0, 0))

        self.assertEqual(coords.to_sgf((0, 0)), 'aa')
        self.assertEqual(coords.to_flat((0, 0)), 0)
        self.assertEqual(coords.to_kgs((0, 0)), 'A9')
        self.assertEqual(coords.to_pygtp((0, 0)), (1, 9))

    def test_topleft(self):
        self.assertEqual(coords.from_sgf('ia'), (0, 8))
        self.assertEqual(coords.from_flat(8), (0, 8))
        self.assertEqual(coords.from_kgs('J9'), (0, 8))
        self.assertEqual(coords.from_pygtp((9, 9)), (0, 8))

        self.assertEqual(coords.to_sgf((0, 8)), 'ia')
        self.assertEqual(coords.to_flat((0, 8)), 8)
        self.assertEqual(coords.to_kgs((0, 8)), 'J9')
        self.assertEqual(coords.to_pygtp((0, 8)), (9, 9))

    def test_pass(self):
        self.assertEqual(coords.from_sgf(''), None)
        self.assertEqual(coords.from_flat(81), None)
        self.assertEqual(coords.from_kgs('pass'), None)
        self.assertEqual(coords.from_pygtp((0, 0)), None)

        self.assertEqual(coords.to_sgf(None), '')
        self.assertEqual(coords.to_flat(None), 81)
        self.assertEqual(coords.to_kgs(None), 'pass')
        self.assertEqual(coords.to_pygtp(None), (0, 0))

    def test_parsing_9x9(self):
        self.assertEqual(coords.from_sgf('aa'), (0, 0))
        self.assertEqual(coords.from_sgf('ac'), (2, 0))
        self.assertEqual(coords.from_sgf('ca'), (0, 2))
        self.assertEqual(coords.from_sgf(''), None)
        self.assertEqual(coords.to_sgf(None), '')
        self.assertEqual(
            'aa',
            coords.to_sgf(coords.from_sgf('aa')))
        self.assertEqual(
            'sa',
            coords.to_sgf(coords.from_sgf('sa')))
        self.assertEqual(
            (1, 17),
            coords.from_sgf(coords.to_sgf((1, 17))))
        self.assertEqual(coords.from_kgs('A1'), (8, 0))
        self.assertEqual(coords.from_kgs('A9'), (0, 0))
        self.assertEqual(coords.from_kgs('C2'), (7, 2))
        self.assertEqual(coords.from_kgs('J2'), (7, 8))
        self.assertEqual(coords.from_pygtp((1, 1)), (8, 0))
        self.assertEqual(coords.from_pygtp((1, 9)), (0, 0))
        self.assertEqual(coords.from_pygtp((3, 2)), (7, 2))
        self.assertEqual(coords.to_pygtp((8, 0)), (1, 1))
        self.assertEqual(coords.to_pygtp((0, 0)), (1, 9))
        self.assertEqual(coords.to_pygtp((7, 2)), (3, 2))

        self.assertEqual(coords.to_kgs((0, 8)), 'J9')
        self.assertEqual(coords.to_kgs((8, 0)), 'A1')

    def test_flatten(self):
        self.assertEqual(coords.to_flat((0, 0)), 0)
        self.assertEqual(coords.to_flat((0, 3)), 3)
        self.assertEqual(coords.to_flat((3, 0)), 27)
        self.assertEqual(coords.from_flat(27), (3, 0))
        self.assertEqual(coords.from_flat(10), (1, 1))
        self.assertEqual(coords.from_flat(80), (8, 8))
        self.assertEqual(coords.to_flat(
            coords.from_flat(10)), 10)
        self.assertEqual(coords.from_flat(
            coords.to_flat((5, 4))), (5, 4))

    def test_from_flat_ndindex_equivalence(self):
        ndindices = list(numpy.ndindex(go.N, go.N))
        flat_coords = list(range(go.N * go.N))
        self.assertEqual(
            list(map(coords.from_flat, flat_coords)), ndindices)
