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
"""Logic for dealing with coordinates.

This introduces some helpers and terminology that are used throughout MiniGo.

MiniGo Coordinate: This is a tuple of the form (row, column) that is indexed
  starting out at (0, 0) from the upper-left.
Flattened Coordinate: this is a number ranging from 0 - N^2 (so N^2+1
  possible values). The extra value N^2 is used to mark a 'pass' move.
SGF Coordinate: Coordinate used for SGF serialization format. Coordinates use
  two-letter pairs having the form (column, row) indexed from the upper-left
  where 0, 0 = 'aa'.
KGS Coordinate: Human-readable coordinate string indexed from bottom left, with
  the first character a capital letter for the column and the second a number
  from 1-19 for the row. Note that KGS chooses to skip the letter 'I' due to
  its similarity with 'l' (lowercase 'L').
PYGTP Coordinate: Tuple coordinate indexed starting at 1,1 from bottom-left
  in the format (column, row)

So, for a 19x19,

Coord Type      upper_left      upper_right     pass
-------------------------------------------------------
minigo coord    (0, 0)          (0, 18)         None
flat            0               18              361
SGF             'aa'            'sa'            ''
KGS             'A19'           'T19'           'pass'
pygtp           (1, 19)         (19, 19)        (0, 0)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gtp

# We provide more than 19 entries here in case of boards larger than 19 x 19.
_SGF_COLUMNS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
_KGS_COLUMNS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'


def from_flat(board_size, flat):
  """Converts from a flattened coordinate to a MiniGo coordinate."""
  if flat == board_size * board_size:
    return None
  return divmod(flat, board_size)


def to_flat(board_size, coord):
  """Converts from a MiniGo coordinate to a flattened coordinate."""
  if coord is None:
    return board_size * board_size
  return board_size * coord[0] + coord[1]


def from_sgf(sgfc):
  """Converts from an SGF coordinate to a MiniGo coordinate."""
  if not sgfc:
    return None
  return _SGF_COLUMNS.index(sgfc[1]), _SGF_COLUMNS.index(sgfc[0])


def to_sgf(coord):
  """Converts from a MiniGo coordinate to an SGF coordinate."""
  if coord is None:
    return ''
  return _SGF_COLUMNS[coord[1]] + _SGF_COLUMNS[coord[0]]


def from_kgs(board_size, kgsc):
  """Converts from a KGS coordinate to a MiniGo coordinate."""
  if kgsc == 'pass':
    return None
  kgsc = kgsc.upper()
  col = _KGS_COLUMNS.index(kgsc[0])
  row_from_bottom = int(kgsc[1:])
  return board_size - row_from_bottom, col


def to_kgs(board_size, coord):
  """Converts from a MiniGo coordinate to a KGS coordinate."""
  if coord is None:
    return 'pass'
  y, x = coord
  return '{}{}'.format(_KGS_COLUMNS[x], board_size - y)


def from_pygtp(board_size, pygtpc):
  """Converts from a pygtp coordinate to a MiniGo coordinate."""
  # GTP has a notion of both a Pass and a Resign, both of which are mapped to
  # None, so the conversion is not precisely bijective.
  if pygtpc in (gtp.PASS, gtp.RESIGN):
    return None
  return board_size - pygtpc[1], pygtpc[0] - 1


def to_pygtp(board_size, coord):
  """Converts from a MiniGo coordinate to a pygtp coordinate."""
  if coord is None:
    return gtp.PASS
  return coord[1] + 1, board_size - coord[0]
