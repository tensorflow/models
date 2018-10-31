"""TODO(vbittorf): DO NOT SUBMIT without one-line documentation for goparams.

TODO(vbittorf): DO NOT SUBMIT without a detailed description of goparams.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import socket
import time
import base64
import struct

HOST = socket.gethostname()

NOWSEC = int(time.time())
NOWSHORT = base64.b32encode(struct.pack(">I", NOWSEC)).decode('ascii').strip('=')

_PARAMS = None

if 'GOPARAMS' in os.environ:
  with open(os.environ['GOPARAMS']) as f:
    _PARAMS = json.load(f)
else:
  pass
  #raise Exception('GOPARAMS not defined. Use GOPARAMS=path/to/json')


def _set(name, default):
  val = default
  if _PARAMS is not None:
    if name not in _PARAMS:
      raise Exception('Key ' + name + ' Not Defined in GOPARAMS config')
    val = _PARAMS[name]
  globals()[name] = val



# How many games before the selfplay workers will stop trying to play more.
_set('MAX_GAMES_PER_GENERATION', 2)

HOME = os.environ['HOME']


# Root directory for everything and stuff
_set('BASE_DIR', '$HOME/results/minigo/current.$HOST.$NOWSHORT/')
BASE_DIR = BASE_DIR.replace('$HOME', HOME)
BASE_DIR = BASE_DIR.replace('$HOST', HOST)
BASE_DIR = BASE_DIR.replace('$NOWSHORT', NOWSHORT)
BASE_DIR = BASE_DIR.replace('$NOWSEC', str(NOWSEC))

# What percent of games to holdout from training per generation
# HOLDOUT_PCT = 0.05
_set('HOLDOUT_PCT', 0.0)

# number of times to go through the main loop
_set('NUM_MAIN_ITERATIONS', 5000)

#BOARD_SIZE = 19
_set('BOARD_SIZE', 9)

# The shuffle buffer size determines how far an example could end up from
# where it started; this and the interleave parameters in preprocessing can give
# us an approximation of a uniform sampling.  The default of 4M is used in
# training, but smaller numbers can be used for aggregation or validation.
# SHUFFLE_BUFFER_SIZE = int(2*1e6)
_set('SHUFFLE_BUFFER_SIZE', int(1 * 1e5))

# How many positions we should aggregate per 'chunk'.
_set('EXAMPLES_PER_RECORD', 10000)

# How many positions to draw from for our training window.
# AGZ used the most recent 500k games, which, assuming 250 moves/game = 125M
# WINDOW_SIZE = 125000000
#WINDOW_SIZE = 500000
_set('WINDOW_SIZE', 10000000)



# Set to run the dummy model instead of the real one, for speedups
_set('DUMMY_MODEL', False)

_set('NUM_PARALLEL_SELFPLAY', 4)

_set('EVAL_GAMES_PER_SIDE', 2)
_set('EVAL_WIN_PCT_FOR_NEW_MODEL', 0.70)
_set('EVALUATE_MODELS', True)

_set('EVALUATE_PUZZLES', True)
_set('TRIES_PER_PUZZLE', 1)
_set('SP_READOUTS', 200)
_set('TERMINATION_ACCURACY', 100)
