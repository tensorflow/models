from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Utilities specific to this project."""

from collections import namedtuple
from six import string_types


#####################
# BF-lang utilities #
#####################


BF_EOS_INT = 0  # Also used as SOS (start of sequence).
BF_EOS_CHAR = TEXT_EOS_CHAR = '_'
BF_LANG_INTS = range(1, 9)
BF_INT_TO_CHAR = [BF_EOS_CHAR, '>', '<', '+', '-', '[', ']', '.', ',']
BF_CHAR_TO_INT = dict([(c, i) for i, c in enumerate(BF_INT_TO_CHAR)])


RewardInfo = namedtuple('RewardInfo', ['episode_rewards', 'input_case',
                                       'correct_output',
                                       'code_output', 'reason', 'input_type',
                                       'output_type'])


class IOType(object):
  string = 'string'
  integer = 'integer'
  boolean = 'boolean'


class IOTuple(tuple):
  pass


def flatten(lst):
  return [item for row in lst for item in row]


def bf_num_tokens():
  # BF tokens plus EOS.
  return len(BF_INT_TO_CHAR)


def bf_char2int(bf_char):
  """Convert BF code char to int token."""
  return BF_CHAR_TO_INT[bf_char]


def bf_int2char(bf_int):
  """Convert BF int token to code char."""
  return BF_INT_TO_CHAR[bf_int]


def bf_tokens_to_string(bf_tokens, truncate=True):
  """Convert token list to code string. Will truncate at EOS token.

  Args:
    bf_tokens: Python list of ints representing the code string.
    truncate: If true, the output string will end at the first EOS token.
        If false, the entire token list is converted to string.

  Returns:
    String representation of the tokens.

  Raises:
    ValueError: If bf_tokens is not a python list.
  """
  if not isinstance(bf_tokens, list):
    raise ValueError('Only python list supported here.')
  if truncate:
    try:
      eos_index = bf_tokens.index(BF_EOS_INT)
    except ValueError:
      eos_index = len(bf_tokens)
  else:
    eos_index = len(bf_tokens)
  return ''.join([BF_INT_TO_CHAR[t] for t in bf_tokens[:eos_index]])


def bf_string_to_tokens(bf_string):
  """Convert string to token list. Will strip and append EOS token."""
  tokens = [BF_CHAR_TO_INT[char] for char in bf_string.strip()]
  tokens.append(BF_EOS_INT)
  return tokens


def tokens_to_text(tokens):
  """Convert token list to human readable text."""
  return ''.join(
      [TEXT_EOS_CHAR if t == 0 else chr(t - 1 + ord('A')) for t in tokens])


###################################
# Number representation utilities #
###################################


# https://en.wikipedia.org/wiki/Metric_prefix
si_magnitudes = {
    'k': 1e3,
    'm': 1e6,
    'g': 1e9}


def si_to_int(s):
  """Convert string ending with SI magnitude to int.

  Examples: 5K ==> 5000, 12M ==> 12000000.

  Args:
    s: String in the form 'xx..xP' where x is a digit and P is an SI prefix.

  Returns:
    Integer equivalent to the string.
  """
  if isinstance(s, string_types) and s[-1].lower() in si_magnitudes.keys():
    return int(int(s[:-1]) * si_magnitudes[s[-1].lower()])
  return int(s)


def int_to_si(n):
  """Convert integer to string with SI magnitude.

  `n` will be truncated.

  Examples: 5432 ==> 5k, 12345678 ==> 12M

  Args:
    n: Integer to represent as a string.

  Returns:
    String representation of `n` containing SI magnitude.
  """
  m = abs(n)
  sign = -1 if n < 0 else 1
  if m < 1e3:
    return str(n)
  if m < 1e6:
    return '{0}K'.format(sign*int(m / 1e3))
  if m < 1e9:
    return '{0}M'.format(sign*int(m / 1e6))
  if m < 1e12:
    return '{0}G'.format(sign*int(m / 1e9))
  return str(m)

