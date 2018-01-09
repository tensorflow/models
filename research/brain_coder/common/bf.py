from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""BrainF**k interpreter.

Language info: https://en.wikipedia.org/wiki/Brainfuck

Based on public implementation:
https://github.com/pocmo/Python-Brainfuck/blob/master/brainfuck.py
"""

from collections import namedtuple
import time


EvalResult = namedtuple(
    'EvalResult', ['output', 'success', 'failure_reason', 'steps', 'time',
                   'memory', 'program_trace'])


ExecutionSnapshot = namedtuple(
    'ExecutionSnapshot',
    ['codeptr', 'codechar', 'memptr', 'memval', 'memory', 'next_input',
     'output_buffer'])


class Status(object):
  SUCCESS = 'success'
  TIMEOUT = 'timeout'
  STEP_LIMIT = 'step-limit'
  SYNTAX_ERROR = 'syntax-error'


CHARS = INT_TO_CHAR = ['>', '<', '+', '-', '[', ']', '.', ',']
CHAR_TO_INT = dict([(c, i) for i, c in enumerate(INT_TO_CHAR)])


class LookAheadIterator(object):
  """Same API as Python iterator, with additional peek method."""

  def __init__(self, iterable):
    self._it = iter(iterable)
    self._current_element = None
    self._done = False
    self._preload_next()

  def _preload_next(self):
    try:
      self._current_element = self._it.next()
    except StopIteration:
      self._done = True

  def next(self):
    if self._done:
      raise StopIteration
    element = self._current_element
    self._preload_next()
    return element

  def peek(self, default_value=None):
    if self._done:
      if default_value is None:
        raise StopIteration
      return default_value
    return self._current_element


def buildbracemap(code):
  """Build jump map.

  Args:
    code: List or string or BF chars.

  Returns:
    bracemap: dict mapping open and close brace positions in the code to their
        destination jumps. Specifically, positions of matching open/close braces
        if they exist.
    correct_syntax: True if all braces match. False if there are unmatched
        braces in the code. Even if there are unmatched braces, a bracemap will
        be built, and unmatched braces will map to themselves.
  """
  bracestack, bracemap = [], {}

  correct_syntax = True
  for position, command in enumerate(code):
    if command == '[':
      bracestack.append(position)
    if command == ']':
      if not bracestack:  # Unmatched closing brace.
        bracemap[position] = position  # Don't jump to any position.
        correct_syntax = False
        continue
      start = bracestack.pop()
      bracemap[start] = position
      bracemap[position] = start
  if bracestack:  # Unmatched opening braces.
    for pos in bracestack:
      bracemap[pos] = pos  # Don't jump to any position.
      correct_syntax = False
  return bracemap, correct_syntax


def evaluate(code, input_buffer=None, init_memory=None, base=256, timeout=1.0,
             max_steps=None, require_correct_syntax=True, output_memory=False,
             debug=False):
  """Execute BF code.

  Args:
    code: String or list of BF characters. Any character not in CHARS will be
        ignored.
    input_buffer: A list of ints which will be used as the program's input
        stream. Each read op "," will read an int from this list. 0's will be
        read once the end of the list is reached, or if no input buffer is
        given.
    init_memory: A list of ints. Memory for first k positions will be
        initialized to this list (where k = len(init_memory)). Memory positions
        are initialized to 0 by default.
    base: Integer base for the memory. When a memory value is incremented to
        `base` it will overflow to 0. When a memory value is decremented to -1
        it will underflow to `base` - 1.
    timeout: Time limit for program execution in seconds. Set to None to
        disable.
    max_steps: Execution step limit. An execution step is the execution of one
        operation (code character), even if that op has been executed before.
        Execution exits when this many steps are reached. Set to None to
        disable. Disabled by default.
    require_correct_syntax: If True, unmatched braces will cause `evaluate` to
        return without executing the code. The failure reason will be
        `Status.SYNTAX_ERROR`. If False, unmatched braces are ignored
        and execution will continue.
    output_memory: If True, the state of the memory at the end of execution is
        returned.
    debug: If True, then a full program trace will be returned.

  Returns:
    EvalResult namedtuple containing
      output: List of ints which were written out by the program with the "."
          operation.
      success: Boolean. Whether execution completed successfully.
      failure_reason: One of the attributes of `Status`. Gives extra info
          about why execution was not successful.
      steps: Number of execution steps the program ran for.
      time: Amount of time in seconds the program ran for.
      memory: If `output_memory` is True, a list of memory cells up to the last
          one written to. otherwise, None.
  """
  input_iter = (
      LookAheadIterator(input_buffer) if input_buffer is not None
      else LookAheadIterator([]))

  # Null memory value. This is the value of an empty memory. Also the value
  # returned by the read operation when the input buffer is empty, or the
  # end of the buffer is reached.
  null_value = 0

  code = list(code)
  bracemap, correct_syntax = buildbracemap(code)  # will modify code list
  if require_correct_syntax and not correct_syntax:
    return EvalResult([], False, Status.SYNTAX_ERROR, 0, 0.0,
                      [] if output_memory else None, [] if debug else None)

  output_buffer = []

  codeptr, cellptr = 0, 0

  cells = list(init_memory) if init_memory else [0]

  program_trace = [] if debug else None
  success = True
  reason = Status.SUCCESS
  start_time = time.time()
  steps = 0
  while codeptr < len(code):
    command = code[codeptr]

    if debug:
      # Add step to program trace.
      program_trace.append(ExecutionSnapshot(
          codeptr=codeptr, codechar=command, memptr=cellptr,
          memval=cells[cellptr], memory=list(cells),
          next_input=input_iter.peek(null_value),
          output_buffer=list(output_buffer)))

    if command == '>':
      cellptr += 1
      if cellptr == len(cells): cells.append(null_value)

    if command == '<':
      cellptr = 0 if cellptr <= 0 else cellptr - 1

    if command == '+':
      cells[cellptr] = cells[cellptr] + 1 if cells[cellptr] < (base - 1) else 0

    if command == '-':
      cells[cellptr] = cells[cellptr] - 1 if cells[cellptr] > 0 else (base - 1)

    if command == '[' and cells[cellptr] == 0: codeptr = bracemap[codeptr]
    if command == ']' and cells[cellptr] != 0: codeptr = bracemap[codeptr]

    if command == '.': output_buffer.append(cells[cellptr])
    if command == ',': cells[cellptr] = next(input_iter, null_value)

    codeptr += 1
    steps += 1

    if timeout is not None and time.time() - start_time > timeout:
      success = False
      reason = Status.TIMEOUT
      break
    if max_steps is not None and steps >= max_steps:
      success = False
      reason = Status.STEP_LIMIT
      break

  if debug:
    # Add step to program trace.
    command = code[codeptr] if codeptr < len(code) else ''
    program_trace.append(ExecutionSnapshot(
        codeptr=codeptr, codechar=command, memptr=cellptr,
        memval=cells[cellptr], memory=list(cells),
        next_input=input_iter.peek(null_value),
        output_buffer=list(output_buffer)))

  return EvalResult(
      output=output_buffer,
      success=success,
      failure_reason=reason,
      steps=steps,
      time=time.time() - start_time,
      memory=cells if output_memory else None,
      program_trace=program_trace)


