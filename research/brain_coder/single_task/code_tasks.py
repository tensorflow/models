from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tasks for RL."""

import abc
import copy
import itertools
import random

from absl import logging
import numpy as np
from six.moves import xrange

from common import bf  # brain coder
from common import reward as r  # brain coder
from single_task import misc  # brain coder
from single_task import test_tasks  # brain coder


MAX_EXECUTION_STEPS = 5000


def make_task(task_name, override_kwargs=None, max_code_length=100,
              require_correct_syntax=False,
              do_code_simplification=False,
              correct_bonus=2.0, code_length_bonus=1.0):
  """Make tasks with setting from paper."""
  logging.info('Making paper-config task.')
  n = 16  # Number of test cases.
  task_mapping = {
      'print-hello': (
          PrintTask, dict(base=27, fixed_string=[8, 5, 12, 12, 15])),
      'print': (PrintIntTask, dict(base=256, fixed_string=[1, 2, 3, 4, 5])),
      'echo': (EchoTask, dict(base=27, min_length=1, max_length=6)),
      'remove-char': (
          RemoveCharTask, dict(base=256, n=n, min_len=1, max_len=6)),
      'reverse': (
          ReverseTask, dict(base=256, n=n, min_len=1, max_len=6)),
      'reverse-tune': (
          ReverseTaskV2, dict(base=256, reward_type='static-bylen')),
      'remove-char-tune': (RemoveCharTaskV2, dict(base=27)),
      'prefix': (CommonPrefixTask, dict(base=27)),
      'find': (FindSubStrTask, dict(base=27)),
      'sort3': (SortFixedTaskV2, dict(base=27, n=150, length=3)),
      'count-char': (CountCharTaskV2, dict(n=n, max_len=6)),
      'bool-logic': (BooleanLogicTask, dict()),
      'add': (AddTask, dict(n=9)),
      'echo-twice': (EchoTwiceTask, dict(n=n)),
      'echo-thrice': (EchoThriceTask, dict(n=n)),
      'copy-reverse': (CopyReverseTask, dict(n=n)),
      'zero-cascade': (EchoZeroCascadeTask, dict(n=n)),
      'cascade': (EchoCascadeTask, dict(n=n)),
      'shift-left': (ShiftLeftTask, dict(n=n)),
      'shift-right': (ShiftRightTask, dict(n=n)),
      'riffle': (RiffleTask, dict(n=n)),
      'unriffle': (UnriffleTask, dict(n=n)),
      'middle-char': (MiddleCharTask, dict(n=n)),
      'remove-last': (RemoveLastTask, dict(n=n)),
      'remove-last-two': (RemoveLastTwoTask, dict(n=n)),
      'echo-alternating': (EchoAlternatingTask, dict(n=n)),
      'echo-half': (EchoHalfTask, dict(n=n)),
      'length': (LengthTask, dict(n=n)),
      'echo-second-seq': (EchoSecondSequenceTask, dict(n=n)),
      'echo-nth-seq': (EchoNthSequenceTask, dict(n=n)),
      'substring': (SubstringTask, dict(n=n)),
      'divide-2': (Divide2Task, dict(n=n)),
      'dedup': (DedupTask, dict(n=n)),
      'remove-target-char': (RemoveTargetCharTask, dict(n=n)),
      'list-index': (ListIndexTask, dict(n=n)),
      'fib': (FibonacciTask, dict()),
      'count-down': (BottlesOfBeerTask, dict()),
      'split': (SplitTask, dict()),
      'trim-left': (TrimLeftTask, dict()),
      'circle-route': (
          JudgeRouteCircleTask, dict(n=100, max_len=32)),
      'multiply': (MultiplyTask, dict(n=100)),
      'divmod': (DivModTask, dict(n=100)),
  }

  if task_name not in task_mapping:
    # Test tasks.
    if task_name == 'test-hill-climb':
      return test_tasks.BasicTaskManager(test_tasks.HillClimbingTask())
    raise ValueError('Unknown task type "%s"' % task_name)
  task_cls, kwargs = task_mapping[task_name]

  if override_kwargs:
    if not isinstance(override_kwargs, dict):
      raise ValueError(
          'override_kwargs must be a dict, got: %s', override_kwargs)
    kwargs.update(override_kwargs)

  task = task_cls(**kwargs)

  reward_fn = r.absolute_distance_reward
  # reward_fn = r.absolute_mod_distance_reward
  # reward_fn = r.absolute_log_distance_reward
  logging.info('Using reward function: %s', reward_fn.__name__)

  # We want reward with and without code simplification to be scaled the same
  # way. Without code simplification, give the maximum code length bonus
  # every time.
  min_code_length = 0.0 if do_code_simplification else max_code_length

  return MultiIOTaskManager(
      task=task, correct_bonus=correct_bonus,
      code_length_bonus=code_length_bonus,
      max_code_length=max_code_length, min_code_length=min_code_length,
      reward_fn=reward_fn, require_correct_syntax=require_correct_syntax)


def concat(lists):
  if not lists:
    return []
  l = lists[0]
  for k in lists[1:]:
    l += k
  return l


def concat_join(lists, sep):
  if not lists:
    return []
  l = lists[0]
  for k in lists[1:]:
    l += [sep] + k
  return l


def clipped_linear(x, x0, y0, slope, y_range):
  min_y, max_y = y_range
  return min(max(slope * (x - x0) + y0, min_y), max_y)


class MultiIOTaskManager(object):
  """Supports tasks which test the code with multiple I/O examples."""

  def __init__(self, task, max_code_length=32, min_code_length=0,
               max_execution_steps=MAX_EXECUTION_STEPS, correct_bonus=1.0,
               code_length_bonus=1.0, failure_reward=-2.0, reward_fn=None,
               require_correct_syntax=False):
    assert isinstance(task, BaseTask)
    self.task = task
    self.max_code_length = max_code_length
    self.min_code_length = min_code_length
    self.max_execution_steps = max_execution_steps
    self.require_correct_syntax = require_correct_syntax
    self.correct_bonus = correct_bonus
    self.code_length_bonus = code_length_bonus
    self.failure_reward = failure_reward
    self.time_penalty = (
        1.0 / (max_code_length - min_code_length)
        if max_code_length > min_code_length else 0.0)
    if reward_fn is None:
      self.reward_fn = r.absolute_distance_reward
    else:
      self.reward_fn = reward_fn
    self.input_type = (
        task.input_type if hasattr(task, 'input_type') else misc.IOType.integer)
    self.output_type = (
        task.output_type if hasattr(task, 'output_type')
        else misc.IOType.integer)
    self._compute_best_reward()

  def _compute_best_reward(self):
    io_seqs = self.task.make_io_set()
    reward = 0.0
    for _, output_seq in io_seqs:
      reward += self.reward_fn(output_seq, output_seq, self.task.base)
      reward += self.correct_bonus
      reward += self.code_length_bonus  # Bonus for shortest code.
    self.best_reward = reward
    self.good_reward = 0.75 * reward
    logging.info('Known best reward: %.4f', self.best_reward)

  def _score_batch(self, code_strings):
    return [self._score_code(code) for code in code_strings]

  def _score_code(self, code):
    """Run test cases on code and compute reward.

    Args:
      code: A single BF code string.

    Returns:
      misc.RewardInfo namedtuple instance containing reward and code execution
          information, including inputs, expected outputs, code outputs, input
          and output types, and reason for the reward obtained.
    """
    # Get list of 2-tuples, each containing an input sequence and an output
    # sequence.
    io_seqs = self.task.make_io_set()
    terminal_reward = 0.0
    results = []
    reason = 'correct'
    for input_seq, output_seq in io_seqs:
      eval_result = bf.evaluate(
          code, input_buffer=input_seq, timeout=0.1,
          max_steps=self.max_execution_steps,
          base=self.task.base,
          require_correct_syntax=self.require_correct_syntax)
      result, success = eval_result.output, eval_result.success
      if not success:
        # Code execution timed out.
        terminal_reward = self.failure_reward
        results = []
        reason = eval_result.failure_reason
        break
      else:
        terminal_reward += self.reward_fn(result, output_seq, self.task.base)
        if result == output_seq:
          terminal_reward += self.correct_bonus  # Bonus for correct answer.

          # Only add additional reward for shorter code. Subtracting reward
          # interferes with the main objective. Only optimize for length once
          # any solution is found.
          if self.min_code_length == self.max_code_length:
            terminal_reward += self.code_length_bonus
          else:
            terminal_reward += self.code_length_bonus * clipped_linear(
                x=len(code), x0=self.min_code_length, y0=1.0,
                slope=-self.time_penalty, y_range=(0.0, 1.0))

          # reason remains 'correct' if it is already
        elif reason == 'correct':
          reason = 'wrong'
      results.append(result)

    # Return list of rewards, one for each char in the code. All are 0 except
    # for the terminal reward.
    terminal_reward /= self.best_reward
    return misc.RewardInfo(
        episode_rewards=[0.0] * (len(code) - 1) + [terminal_reward],
        input_case=misc.IOTuple(i for i, o in io_seqs),
        correct_output=misc.IOTuple(o for i, o in io_seqs),
        code_output=misc.IOTuple(results),
        input_type=self.input_type,
        output_type=self.output_type,
        reason=reason)

  def rl_batch(self, batch_size):
    """Produces list of reward functions. One for each program in the batch."""
    return [self._score_code] * batch_size


def conditional_overwrite(current_value, new_value, allowed_overwrite_values):
  if current_value in allowed_overwrite_values:
    return new_value
  return current_value


class BaseTask(object):
  """A coding task.

  All coding tasks should inherit this class.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, base=256):
    self.base = base  # All tasks must set the integer base that the expect.

  @abc.abstractmethod
  def make_io_set(self):
    """Generate a set of test cases for the task.

    Returns:
      List of tuples, where each tuple is (input_case, output_case).
      input_case and output_case are lists of integers.
    """
    pass


# ==============================================================================
# ICLR tasks.
# ==============================================================================


class PrintTask(BaseTask):
  """Print string coding task.

  Code needs to output a fixed string (given as a hyperparameter to the
  task constructor). Program input is ignored.
  """

  def __init__(self, base, fixed_string=None):
    super(type(self), self).__init__()
    self.base = base  # base includes EOS
    self.eos = 0
    if fixed_string:
      self.fixed_string = fixed_string
    else:
      self.fixed_string = [1, 2, 3, 0]  # ABC<EOS>
    self.min_length = self.max_length = len(self.fixed_string)

  def make_io_set(self):
    return [(list(), list(self.fixed_string))]


class RemoveCharTaskV2(BaseTask):
  """Remove character coding task (version 2).

  Code needs to pipe input to output, but with all the 'A' (value 1) chars
  removed. 'A' appears exactly once in each input.

  Test cases are hard-coded.
  """

  def __init__(self, base):
    super(type(self), self).__init__()
    self.base = base
    self.eos = 0
    self.remove_char = 1
    assert base >= 27

  def make_io_set(self):
    rm = self.remove_char
    return [
        ([rm, 0], [0]),
        ([20, rm, 0], [20, 0]),
        ([rm, 13, 0], [13, 0]),
        ([6, rm, 17, 0], [6, 17, 0]),
        ([rm, 11, 24, 0], [11, 24, 0]),
        ([2, 16, 21, rm, 0], [2, 16, 21, 0]),
        ([18, rm, 12, 26, 7, 0], [18, 12, 26, 7, 0]),
        ([9, 10, 22, rm, 4, 0], [9, 10, 22, 4, 0])]


class RemoveCharTask(BaseTask):
  """Remove character coding task.

  Code needs to pipe input to output, but with all the 'A' (value 1) chars
  removed. 'A' appears at least once in each input.

  Test cases are dynamically generated, allowing for the number of test cases
  to be a hyperparameter.
  """

  def __init__(self, base, n, min_len, max_len):
    super(type(self), self).__init__()
    self.base = base
    self.eos = 0
    self.remove_char = 1
    assert base >= 27
    self._io_pairs = self._make_io_examples(n, min_len, max_len)

  def _make_io_examples(self, n, min_len, max_len):
    """Generate test cases for the task."""
    rand = random.Random(6849275409234)  # Test cases are fixed, but varied.
    io_examples = []
    for _ in xrange(n):
      length = rand.randrange(min_len, max_len + 1)
      rm_char_pos = rand.randrange(0, length)
      input_seq = [rand.randrange(1, self.base) for _ in xrange(length)]
      input_seq[rm_char_pos] = self.remove_char
      output_seq = list(input_seq)
      del output_seq[rm_char_pos]
      output_seq.append(0)
      io_examples.append((input_seq, output_seq))
    return io_examples

  def make_io_set(self):
    return copy.deepcopy(self._io_pairs)


class ReverseTaskV2(BaseTask):
  """Reverse string coding task (version 2).

  Code needs to pipe input to output, but in reverse order.

  Stochastic test case = new test case randomly generated for every run of
  `make_io_set`, i.e. different test cases every time code is scored.

  Task supports different types of test cases:
    rand-one: Code is scored on one stochastic test case.
    rand-many: Code is scored on 5 stochastic test cases.
    static-bylen: Code is scored on 5 static test cases. There is one test
        case for string lengths 1 through 5.
    rand-bylen: Code is scored on 5 stochastic test cases, where there is one
        test case for string lengths 1 through 5.
  """

  def __init__(self, base, reward_type):
    super(type(self), self).__init__()
    self.base = base  # base includes EOS
    assert base >= 27
    self.eos = 0
    self.io_pair_fn = {
        # One random example at a time.
        'rand-one': lambda: self._io_rand(1),
        # K randomy examples at a time (any lengths).
        'rand-many': lambda: self._io_rand(5),
        # Static examples, one for each length.
        'static-bylen': self._io_static_by_len,
        # Random examples, one for each length.
        'rand-bylen': self._io_rand_by_len}[reward_type]

  def _make_io_examples(self, sequences):
    outputs = [list(i) for i in sequences]
    for o in outputs:
      o.reverse()
      o.append(0)
    inputs = [i + [0] for i in sequences]
    return zip(inputs, outputs)

  def _io_rand(self, k):
    inputs = [(np.random.choice(26, random.randrange(1, 6)) + 1).tolist()
              for _ in xrange(k)]
    return self._make_io_examples(inputs)

  def _io_rand_by_len(self, k=5):
    inputs = [(np.random.choice(26, length) + 1).tolist()
              for length in xrange(1, k + 1)]
    return self._make_io_examples(inputs)

  def _io_static_by_len(self):
    return [
        ([7, 0], [7, 0]),
        ([6, 2, 0], [2, 6, 0]),
        ([5, 1, 10, 0], [10, 1, 5, 0]),
        ([8, 6, 5, 15, 0], [15, 5, 6, 8, 0]),
        ([10, 12, 5, 2, 7, 0], [7, 2, 5, 12, 10, 0])]

  def make_io_set(self):
    return self.io_pair_fn()


class ReverseTask(BaseTask):
  """Reverse string coding task.

  Code needs to pipe input to output, but in reverse order.

  Test cases are dynamically generated, allowing for the number of test cases
  to be a hyperparameter.
  """

  def __init__(self, base, n, min_len, max_len):
    super(type(self), self).__init__()
    self.base = base  # base includes EOS
    assert base >= 27
    self.eos = 0
    self._io_pairs = self._make_io_examples(n, min_len, max_len)

  def _make_io_examples(self, n, min_len, max_len):
    """Generate test cases for the task."""
    rand = random.Random(6849275409234)  # Test cases are fixed, but varied.
    io_examples = []
    for _ in xrange(n):
      length = rand.randrange(min_len, max_len + 1)
      input_seq = [rand.randrange(1, self.base) for _ in xrange(length)]
      output_seq = list(input_seq)
      output_seq.reverse()
      output_seq.append(0)
      io_examples.append((input_seq, output_seq))
    return io_examples

  def make_io_set(self):
    return copy.deepcopy(self._io_pairs)


class CommonPrefixTask(BaseTask):
  """Common prefix coding task.

  Code needs to output the common prefix between two input lists. Input lists
  are variable length, where each list ends with a 0. A common prefix is a
  sequence which both lists start with.
  """

  def __init__(self, base):
    super(type(self), self).__init__()
    assert base >= 27
    self.base = base
    self.eos = 0

  def make_io_set(self):
    return [
        ([12, 24, 18, 0, 12, 5, 0], [12, 0]),
        ([1, 2, 3, 0, 1, 2, 17, 14, 0], [1, 2, 0]),
        ([15, 2, 1, 9, 2, 0, 15, 2, 1, 25, 8, 14, 0], [15, 2, 1, 0]),
        ([14, 9, 7, 8, 6, 16, 0, 14, 9, 7, 8, 8, 6, 8, 26, 0],
         [14, 9, 7, 8, 0]),
        ([12, 4, 16, 22, 1, 17, 0, 12, 4, 16, 22, 1, 8, 10, 0],
         [12, 4, 16, 22, 1, 0])]


class CountCharTask(BaseTask):

  def __init__(self):
    super(type(self), self).__init__()
    self.base = 27
    self.eos = 0
    self.char = 1
    self.input_type = misc.IOType.string
    self.output_type = misc.IOType.integer

  def make_io_set(self):
    return [
        ([10, 0], [0]),
        ([1, 0], [1]),
        ([1, 1, 0], [2]),
        ([11, 1, 0], [1]),
        ([1, 24, 0], [1]),
        ([13, 6, 0], [0]),
        ([9, 2, 7, 0], [0]),
        ([1, 24, 11, 0], [1]),
        ([19, 1, 1, 0], [2]),
        ([1, 6, 1, 0], [2]),
        ([22, 16, 17, 9, 0], [0]),
        ([1, 1, 1, 19, 0], [3]),
        ([1, 1, 1, 1, 0], [4]),
        ([9, 4, 19, 11, 5, 0], [0]),
        ([24, 11, 26, 1, 15, 0], [1]),
        ([1, 1, 20, 1, 1, 0], [4]),
        ([1, 1, 1, 1, 1, 0], [5])]


class CountCharTaskV2(BaseTask):
  """Count char coding task (version 2).

  Code must output the number of occurances of character 'A' (value 1) in an
  input string.

  Test cases are dynamically generated, allowing for the number of test cases
  to be a hyperparameter.
  """

  def __init__(self, n, max_len):
    super(type(self), self).__init__()
    self.base = 27
    self.eos = 0
    self.char = 1
    self.other_chars = [c for c in xrange(self.base)
                        if c not in (self.eos, self.char)]
    self.input_type = misc.IOType.string
    self.output_type = misc.IOType.integer
    self._io_pairs = self._make_io_examples(n, max_len)

  def _make_io_examples(self, n, max_len):
    """Generate test cases for the task."""
    rand = random.Random(6849275409234)  # Test cases are fixed, but varied.
    io_examples = []
    io_examples.append(([10, 0], [0]))
    io_examples.append(([1, 0], [1]))
    io_examples.append(([1, 1, 0], [2]))
    io_examples.append(([9, 4, 19, 11, 5, 0], [0]))
    io_examples.append(([24, 11, 26, 1, 15, 0], [1]))
    for _ in xrange(n - 5):
      length = rand.randrange(2, max_len + 1)
      num_chars = rand.randrange(0, max_len + 1)
      input_seq = [self.char] * num_chars + [0] * (length - num_chars)
      rand.shuffle(input_seq)
      for i in xrange(len(input_seq)):
        if not input_seq[i]:
          input_seq[i] = self.other_chars[rand.randrange(len(self.other_chars))]
      output_seq = [num_chars]
      io_examples.append((input_seq, output_seq))
    return io_examples

  def make_io_set(self):
    return copy.deepcopy(self._io_pairs)


class AddTask(BaseTask):
  """Addition coding task.

  Code needs to read in two integers and output their sum mod the BF base,
  followed by a terminating 0.
  """

  def __init__(self, n=16):
    super(type(self), self).__init__()
    self.base = 256
    self.input_type = misc.IOType.integer
    self.output_type = misc.IOType.integer
    self._io_pairs = self._make_io_examples(n)

  def _make_io_examples(self, n):
    """Generate test cases for the task."""
    rand = random.Random(6849275409234)  # Test cases are fixed, but varied.
    io_examples = [
        ([4, 0], [4, 0]),
        ([0, 5], [5, 0]),
        ([1, 2], [3, 0]),
        ([67, 21], [88, 0]),
        ([55, 56], [111, 0]),
        ([128, 33], [161, 0]),
        ([221, 251], [216, 0]),
        ([130, 127], [1, 0]),
        ([255, 1], [0, 0])]
    extra_examples = max(n - len(io_examples), 0)
    for _ in xrange(extra_examples):
      a = rand.randrange(256)
      b = rand.randrange(256)
      input_seq = [a, b]
      output_seq = [(a + b) % 256, 0]
      io_examples.append((input_seq, output_seq))
    return io_examples

  def make_io_set(self):
    return copy.deepcopy(self._io_pairs)


class BooleanLogicTask(BaseTask):
  """Boolean logic (truth table) coding task.

  Code needs to memorize a boolean truth table. Specifically, it must encode a
  mapping from triple of bools to a single bool.
  """

  def __init__(self):
    super(type(self), self).__init__()
    self.base = 2
    self.input_type = misc.IOType.boolean
    self.output_type = misc.IOType.boolean
    # X(~Z) + (~Y)(~Z) + (~X)YZ
    self._truth_fn = (
        lambda x, y, z:  # pylint: disable=g-long-lambda
        (x and not z) or (not y and not z) or (not x and y and z))
    self._test_cases = [
        ([x, y, z], [int(self._truth_fn(x, y, z))])
        for x, y, z in itertools.product(range(2), range(2), range(2))]

  def make_io_set(self):
    return copy.deepcopy(self._test_cases)


# ------------------------------------------------------------------------------
# The following tasks are generated from known BF solutions. This guarantees
# that each task can be solved within the maximum code length, and maximum
# execution steps.
# ------------------------------------------------------------------------------


def default_input_fn_factory(min_length=1, max_length=6, base=256):
  def _input_gen(rand):
    l = rand.randrange(min_length, max_length + 1)
    return [rand.randrange(base) for _ in xrange(l)]
  return _input_gen


class KnownCodeBaseTask(BaseTask):
  """These tasks generate their test cases from a known BF solution.

  This ensures that each task has a solution which is under the max character
  length, and that it solves the test cases under the max number of execution
  steps.
  """

  def __init__(self, code_solution, make_input_fn, n=100, base=256,
               max_steps=5000, seed=6849275409234):
    super(KnownCodeBaseTask, self).__init__()
    # Make sure known solution is less than the code length used in experiments.
    assert len(code_solution) < 100
    self.code_solution = code_solution
    self.make_input_fn = make_input_fn
    self.n = n
    self.base = base
    self.max_steps = max_steps
    self.seed = seed
    self._test_cases = list(self._test_case_generator(code_solution))

  def _test_case_generator(self, code_solution):
    rand = random.Random(self.seed)
    for _ in xrange(self.n):
      input_case = self.make_input_fn(rand)
      result = bf.evaluate(
          code_solution, input_buffer=input_case, max_steps=self.max_steps,
          base=self.base, require_correct_syntax=False)
      if not result.success:
        raise RuntimeError(
            'Program must succeed. Failed on input: %s' % input_case)
      yield input_case, result.output

  def make_io_set(self):
    return copy.deepcopy(self._test_cases)


class EchoTwiceTask(KnownCodeBaseTask):
  """Echo twice."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        '>,.[>,.]<[<]>[.>].',
        default_input_fn_factory(),
        **kwargs)


class EchoThriceTask(KnownCodeBaseTask):
  """Echo three times."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        '>,.[>,.]<[<]>[.>].<[<]>[.>].',
        default_input_fn_factory(),
        **kwargs)


class CopyReverseTask(KnownCodeBaseTask):
  """Echo forwards, backwards, and then forwards again."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        '>,.[>,.]<[.<].>[.>].',
        default_input_fn_factory(),
        **kwargs)


class EchoZeroCascadeTask(KnownCodeBaseTask):
  """Print k-th char with k zeros inbetween (1-indexed)."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        ',[.>[->+>.<<]>+[-<+>]<<,]',
        default_input_fn_factory(),
        **kwargs)


class EchoCascadeTask(KnownCodeBaseTask):
  """Print k-th char k times (1-indexed)."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        ',>>+<<[>>[-<+>]<[->+<<.>]>+<<,].',
        default_input_fn_factory(base=20),
        **kwargs)


class ShiftLeftTask(KnownCodeBaseTask):
  """Circulate shift input left."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        ',>,[.,]<.,.',
        default_input_fn_factory(),
        **kwargs)


class ShiftRightTask(KnownCodeBaseTask):
  """Circular shift input right."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        '>,[>,]<.[-]<[<]>[.>].',
        default_input_fn_factory(),
        **kwargs)


class RiffleTask(KnownCodeBaseTask):
  """Shuffle like a deck of cards.

  For input of length N, output values in the following index order:
  N-1, 0, N-2, 1, N-3, 2, ...
  """

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        '>,[>,]<[.[-]<[<]>.[-]>[>]<]',
        default_input_fn_factory(base=20, max_length=8),
        **kwargs)


class UnriffleTask(KnownCodeBaseTask):
  """Inverse of riffle."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        '>,[>,[.[-]],]<[.<].',
        default_input_fn_factory(base=20, max_length=8),
        **kwargs)


class MiddleCharTask(KnownCodeBaseTask):
  """Print middle char if length is odd, or 0 if even."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        '>,[>,]<<[[>]<[,<[<]>,>[>]][>]<<]>.',
        default_input_fn_factory(max_length=10),
        **kwargs)


class RemoveLastTask(KnownCodeBaseTask):
  """Remove last character."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        ',>,[[<.[-]>[-<+>]],].',
        default_input_fn_factory(base=20),
        **kwargs)


class RemoveLastTwoTask(KnownCodeBaseTask):
  """Remove last two characters."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        ',>,>,[[<<.[-]>[-<+>]>[-<+>]],].',
        default_input_fn_factory(base=10),
        **kwargs)


class EchoAlternatingTask(KnownCodeBaseTask):
  # Print even numbered chars first (0-indexed), then odd numbered chars

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        '>,[.,>,]<<[<]>[.>].',
        default_input_fn_factory(base=20, max_length=8),
        **kwargs)


class EchoHalfTask(KnownCodeBaseTask):
  """Echo only first half of the input (round down when odd lengthed)."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        '>>+>,[[<]>+[>],]<[<]>-[-[-<<+>]<[>]>]<<[->+<]>[[>]>.,<+[<]>-].',
        default_input_fn_factory(base=20, max_length=9),
        **kwargs)


class LengthTask(KnownCodeBaseTask):
  """Print length of the input sequence."""

  def __init__(self, **kwargs):
    super(type(self), self).__init__(
        '>+>,[[<]>+[>],]<[<]>-.',
        default_input_fn_factory(max_length=14),
        **kwargs)


class EchoSecondSequenceTask(KnownCodeBaseTask):
  """Echo second sequence. Sequences are separated by 0."""

  def __init__(self, **kwargs):
    def echo_second_gen(rand):
      l = rand.randrange(1, 6)
      x = [rand.randrange(256) for _ in xrange(l)]
      l = rand.randrange(1, 6)
      y = [rand.randrange(256) for _ in xrange(l)]
      return x + [0] + y + [0]
    super(type(self), self).__init__(
        ',[,],[.,].',
        echo_second_gen,
        **kwargs)


class EchoNthSequenceTask(KnownCodeBaseTask):
  """Echo n-th sequence (1-indexed). Sequences are separated by 0."""

  def __init__(self, **kwargs):
    def echo_nth_gen(rand):
      k = rand.randrange(1, 7)
      n = rand.randrange(1, k + 1)
      x = []
      for _ in xrange(k):
        l = rand.randrange(0, 4)
        x += [rand.randrange(256) for _ in xrange(l)] + [0]
      return [n] + x
    super(type(self), self).__init__(
        ',-[->,[,]<],[.,].',
        echo_nth_gen,
        **kwargs)


class SubstringTask(KnownCodeBaseTask):
  """Echo substring.

  First two inputs are i and l, where i is the starting index (0-indexed)
  and l is the length of the substring.
  """

  def __init__(self, **kwargs):
    def substring_gen(rand):
      l = rand.randrange(2, 16)
      i, j = sorted([rand.randrange(l), rand.randrange(l)])
      n = j - i
      x = [rand.randrange(256) for _ in xrange(l)] + [0]
      return [i, n] + x
    super(type(self), self).__init__(
        '>,<,>[->,<]>,<<[->>.,<<]',
        substring_gen,
        **kwargs)


class Divide2Task(KnownCodeBaseTask):
  """Divide by 2 (integer floor division)."""

  def __init__(self, **kwargs):
    def int_input_gen(rand):
      return [rand.randrange(256)]
    super(type(self), self).__init__(
        ',[-[->>+<]>[<]<]>>.',
        int_input_gen,
        **kwargs)


class DedupTask(KnownCodeBaseTask):
  """Deduplicate adjacent duplicate chars."""

  def __init__(self, **kwargs):
    def dedup_input_gen(rand):
      np_random = np.random.RandomState(rand.randrange(2147483647))
      num_unique = rand.randrange(1, 5)
      unique = np_random.choice(6, num_unique, replace=False) + 1
      return [v for v in unique for _ in xrange(rand.randrange(1, 5))] + [0]
    super(type(self), self).__init__(
        '>>,.[[-<+<+>>],[-<->]<[[-<->]<.>]<[->>+<<]>>]',
        dedup_input_gen,
        **kwargs)


# ==============================================================================
# Extra tasks.
# ==============================================================================


class PrintIntTask(BaseTask):
  """Print integer coding task.

  Code needs to output a fixed single value (given as a hyperparameter to the
  task constructor). Program input is ignored.
  """

  def __init__(self, base, fixed_string):
    super(type(self), self).__init__()
    self.base = base
    self.eos = 0
    self.fixed_string = fixed_string
    self.input_type = misc.IOType.integer
    self.output_type = misc.IOType.integer

  def make_io_set(self):
    return [(list(), list(self.fixed_string))]


class EchoTask(BaseTask):
  """Echo string coding task.

  Code needs to pipe input to putput (without any modifications).
  """

  def __init__(self, base, min_length=1, max_length=5):
    super(type(self), self).__init__()
    self.base = base  # base includes EOS
    self.eos = 0
    self.min_length = min_length
    self.max_length = max_length
    self._io_pairs = self._make_io_examples(25)

  def _make_io_examples(self, n):
    # Test cases are fixed, but varied.
    np_random = np.random.RandomState(1234567890)
    io_pairs = []
    for _ in xrange(n):
      length = np_random.randint(self.min_length, self.max_length + 1)
      input_seq = np_random.randint(1, self.base, length).tolist() + [self.eos]
      output_seq = list(input_seq)
      io_pairs.append((input_seq, output_seq))
    return io_pairs

  def make_io_set(self):
    return copy.deepcopy(self._io_pairs)


class JudgeRouteCircleTask(BaseTask):
  """Judge route circle coding task.

  Code needs to determine if the given route makes a closed loop.
  Encoding: U = 1, R = 2, D = 3, L = 4.

  Based on
  https://leetcode.com/problems/judge-route-circle/description/
  """
  base = 256
  input_type = misc.IOType.integer
  output_type = misc.IOType.integer

  def __init__(self, n, max_len=12):
    super(type(self), self).__init__()
    self.eos = 0
    self._io_pairs = self._make_io_examples(n, max_len)
    self.input_type = misc.IOType.integer
    self.output_type = misc.IOType.integer

  def _solve(self, input_seq):
    assert input_seq[-1] == 0
    pos = [0, 0]  # (x, y)
    for move in input_seq[:-1]:
      assert 0 < move <= 4
      if move & 1 == 0:  # Left or Right.
        pos[0] += 3 - move  # Add or subtract 1.
      else:
        pos[1] += 2 - move  # Add or subtract 1.
    return [int(not pos[0] and not pos[1])]

  def _make_io_examples(self, n, max_len):
    """Generate test cases for the task."""
    rand = random.Random(6849275409234)  # Test cases are fixed, but varied.
    io_examples = []
    io_examples.append(([0], [1]))
    io_examples.append(([4, 2, 0], [1]))
    io_examples.append(([2, 4, 0], [1]))
    io_examples.append(([3, 1, 0], [1]))
    io_examples.append(([1, 3, 0], [1]))
    io_examples.append(([1, 0], [0]))
    io_examples.append(([2, 0], [0]))
    io_examples.append(([3, 0], [0]))
    io_examples.append(([4, 0], [0]))
    for _ in xrange(n):
      is_true = rand.randrange(2)
      length = rand.randrange(1, max_len + 1)
      if is_true:
        # Make a true case.
        length = (length >> 1) << 1  # Make even.
        partition = (rand.randrange(length + 1) >> 1) << 1
        a = partition >> 1
        b = (length - partition) >> 1
        counts = {1: a, 2: b, 3: a, 4: b}
      else:
        # Make a false case.
        partitions = (
            [0]
            + sorted([rand.randrange(length + 1) for _ in range(3)])
            + [length])
        counts = {n: partitions[n] - partitions[n - 1] for n in range(1, 5)}
        if counts[1] == counts[3] and counts[2] == counts[4]:
          # By chance we sampled a true case. Make it false by exchanging
          # one count between even and odd pairs.
          base = 1 + 2 * rand.randrange(2)
          a, b = (base, base + 1) if rand.randrange(2) else (base + 1, base)
          if counts[a] == length or counts[b] == 0:
            # If counts are at their extreme values, then swap who gets
            # incremented and decremented.
            a, b = b, a
          counts[a] += 1
          counts[b] -= 1
          assert counts[a] <= length and counts[b] >= 0
      assert sum(counts.values()) == length
      input_seq = [n for n in xrange(1, 5) for _ in xrange(counts[n])]
      rand.shuffle(input_seq)
      input_seq += [0]
      output_seq = self._solve(input_seq)
      assert output_seq[0] == is_true
      io_examples.append((input_seq, output_seq))
    return io_examples

  def make_io_set(self):
    return copy.deepcopy(self._io_pairs)


class MultiplyTask(BaseTask):
  """Multiply coding task.

  Code needs to multiple two ints.

  Solution:
  http://robl.co/brief-look-at-brainfuck/
  ,>,><<[->[->+>+<<]>>[-<<+>>]<<<]>>.
  """
  base = 512
  input_type = misc.IOType.integer
  output_type = misc.IOType.integer

  def __init__(self, n):
    super(type(self), self).__init__()
    self.eos = 0
    self._io_pairs = self._make_io_examples(n)
    self.input_type = misc.IOType.integer
    self.output_type = misc.IOType.integer

  def _factors(self, n):
    return set(i for i in range(1, int(n**0.5) + 1) if n % i == 0)

  def _make_io_examples(self, n):
    """Generate test cases for the task."""
    rand = random.Random(6849275409234)  # Test cases are fixed, but varied.
    io_examples = []
    for _ in xrange(n):
      n = rand.randrange(self.base)
      if n == 0:
        a, b = 0, rand.randrange(self.base)
      else:
        f = list(self._factors(n))
        a = f[rand.randrange(len(f))]
        b = n // a
      if rand.randrange(2):
        a, b = b, a
      io_examples.append(([a, b], [n]))
    return io_examples

  def make_io_set(self):
    return copy.deepcopy(self._io_pairs)


class DivModTask(BaseTask):
  """Divmod coding task.

  Code needs to take the quotient and remainder of two ints.

  Solution:
  http://robl.co/brief-look-at-brainfuck/
  ,>,><<[>[->+>+<<]>[-<<-[>]>>>[<[-<->]<[>]>>[[-]>>+<]>-<]<<]>>>+<<[-<<+>>]<<<]>
  >>>>[-<<<<<+>>>>>]<<<<<.>.>
  """
  base = 512
  input_type = misc.IOType.integer
  output_type = misc.IOType.integer

  def __init__(self, n):
    super(type(self), self).__init__()
    self.eos = 0
    self._io_pairs = self._make_io_examples(n)
    self.input_type = misc.IOType.integer
    self.output_type = misc.IOType.integer

  def _make_io_examples(self, n):
    rand = random.Random(6849275409234)  # Test cases are fixed, but varied.
    io_examples = []
    for _ in xrange(n):
      n = rand.randrange(0, self.base)
      k = rand.randrange(1, self.base)  # Divisor cannot be 0.
      io_examples.append(([n, k], list(divmod(n, k))))
    return io_examples

  def make_io_set(self):
    return copy.deepcopy(self._io_pairs)


class FibonacciTask(BaseTask):

  def __init__(self):
    super(type(self), self).__init__()
    self.base = 256
    self.input_type = misc.IOType.integer
    self.output_type = misc.IOType.integer

  def make_io_set(self):
    return [
        ([0], [0, 1]),
        ([1], [1, 1]),
        ([2], [1, 2]),
        ([3], [2, 3]),
        ([4], [3, 5]),
        ([5], [5, 8]),
        ([6], [8, 13]),
        ([7], [13, 21]),
        ([8], [21, 34]),
        ([9], [34, 55]),
        ([10], [55, 89]),
        ([11], [89, 144]),
        ([12], [144, 233]),
        ([13], [233, 121])]


class FindSubStrTask(BaseTask):
  """Find sub-string coding task.

  Code needs to output a bool: True if the input string contains a hard-coded
  substring, 'AB' (values [1, 2]).
  """

  def __init__(self, base):
    super(type(self), self).__init__()
    assert base >= 27
    self.base = base
    self.eos = 0
    self.find_str = [1, 2]
    self.input_type = misc.IOType.string
    self.output_type = misc.IOType.boolean

  def make_io_set(self):
    return [
        ([1, 1, 23, 0], [0]),
        ([21, 3, 2, 0], [0]),
        ([2, 1, 19, 0], [0]),
        ([2, 24, 15, 3, 0], [0]),
        ([24, 6, 10, 16, 4, 0], [0]),
        ([1, 2, 12, 0], [1]),
        ([7, 1, 2, 0], [1]),
        ([1, 2, 11, 3, 0], [1]),
        ([1, 1, 2, 18, 0], [1]),
        ([7, 25, 1, 2, 0], [1]),
        ([3, 1, 2, 11, 8, 0], [1]),
        ([15, 16, 20, 1, 2, 0], [1])]


class SortFixedTask(BaseTask):
  """Sort list coding task.

  Code needs to output a sorted input list. The task consists of lists of the
  same length L, where L is provided to this task's constructor as a
  hyperparameter.
  """

  def __init__(self, base, length=3):
    super(type(self), self).__init__()
    assert base >= 27
    self.base = base
    self.eos = 0
    self.length = length
    assert length == 3  # More lengths will be supported.

  def make_io_set(self):
    if self.length == 3:
      return [
          ([1, 20, 6], [1, 6, 20]),
          ([13, 6, 7], [6, 7, 13]),
          ([24, 2, 23], [2, 23, 24]),
          ([16, 12, 3], [3, 12, 16]),
          ([11, 24, 4], [4, 11, 24]),
          ([10, 1, 19], [1, 10, 19])]


class SortFixedTaskV2(BaseTask):
  """Sort list coding task (version 2).

  Code needs to output a sorted input list. The task consists of lists of the
  same length L, where L is provided to this task's constructor as a
  hyperparameter.

  Test cases are dynamically generated, allowing for the number of test cases
  to be a hyperparameter.
  """

  def __init__(self, base, n, length=3):
    super(type(self), self).__init__()
    assert base >= 27
    self.base = base
    self.eos = 0
    self._io_pairs = self._make_io_examples(n, length)
    self.input_type = misc.IOType.integer
    self.output_type = misc.IOType.integer

  def _make_io_examples(self, n, length):
    rand = random.Random(6849275409234)  # Test cases are fixed, but varied.
    io_examples = []
    for _ in xrange(n):
      input_seq = [rand.randrange(1, self.base) for _ in xrange(length)]
      output_seq = sorted(input_seq)
      io_examples.append((input_seq, output_seq))
    return io_examples

  def make_io_set(self):
    return copy.deepcopy(self._io_pairs)


class RemoveTargetCharTask(KnownCodeBaseTask):
  """Remove target character from string, where first input is the target.

  Target can appear multiple times.
  """

  def __init__(self, **kwargs):
    def randrange_hole(rand, a, hole, b):
      x = rand.randrange(a, b - 1)
      if x >= hole:
        return x + 1
      return x
    def remove_target_char_gen(rand):
      char = rand.randrange(1, 6)
      l = rand.randrange(1, 8)
      input_seq = [randrange_hole(rand, 1, char, 256) for _ in xrange(l)]
      idx = range(l)
      rand.shuffle(idx)
      num_targets = rand.randrange(0, l)
      for pos in idx[:num_targets]:
        input_seq[pos] = char
      return [char] + input_seq + [0]
    super(type(self), self).__init__(
        ',>>>,[<<<[->+>+<<]>>[->->+<<]>[>[-<+>]<.[-]]>[-]<<<[-<+>]>>,].',
        remove_target_char_gen,
        **kwargs)


class ListIndexTask(KnownCodeBaseTask):
  """Echo i-th value in the given list."""

  def __init__(self, **kwargs):
    def array_index_gen(rand):
      l = rand.randrange(1, 16)
      i = rand.randrange(l)
      return [i] + [rand.randrange(256) for _ in xrange(l)] + [0]
    super(type(self), self).__init__(
        ',[->,<]>,.',
        array_index_gen,
        **kwargs)


# ==============================================================================
# Tasks based on primaryobjects paper.
# ==============================================================================


def string2tokens(string):
  return [ord(c) for c in string]


def stringlist2tokens(strings):
  return [string2tokens(string) for string in strings]


def string2tokens_b27(string):
  return [ord(c.lower()) - ord('a') + 1 for c in string]


def stringlist2tokens_b27(strings):
  return [string2tokens_b27(string) for string in strings]


class BottlesOfBeerTask(BaseTask):
  """Bottles of beer coding task.

  This is a counting task. Code needs to read in an int N and then output
  every int from N to 0, each separated by a 0.
  """
  base = 256
  input_type = misc.IOType.integer
  output_type = misc.IOType.integer

  def make_io_set(self):
    return [
        ([1], [1, 0]),
        ([2], [2, 0, 1, 0]),
        ([3], [3, 0, 2, 0, 1, 0]),
        ([4], [4, 0, 3, 0, 2, 0, 1, 0]),
        ([5], [5, 0, 4, 0, 3, 0, 2, 0, 1, 0]),
        ([6], [6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0])]


class SplitTask(BaseTask):
  """Split coding task.

  Code needs to pipe input strings to output, but insert a 0 after every 3
  characters. This is in essence splitting the string into intervals of length
  3.
  """
  base = 28
  input_type = misc.IOType.string
  output_type = misc.IOType.integer

  def _splicer(self, lst, insert, interval=3):
    for i, item in enumerate(lst):
      yield item
      if (i + 1) % interval == 0 and i < len(lst) - 1:
        yield insert

  def __init__(self):
    super(type(self), self).__init__()
    inputs = stringlist2tokens_b27(
        ['hello', 'orange', 'spaghetti', 'wins', 'one'])
    targets = [list(self._splicer(i, 27)) for i in inputs]
    self._test_cases = list(zip(inputs, targets))

  def make_io_set(self):
    return copy.deepcopy(self._test_cases)


class TrimLeftTask(BaseTask):
  """Trim left coding task.

  Code needs to pipe input strings to output, but remove everything before the
  first quotation char (").
  """
  base = 256
  input_type = misc.IOType.integer
  output_type = misc.IOType.integer

  def __init__(self):
    super(type(self), self).__init__()
    inputs = stringlist2tokens(
        ['a "inside" over', 'xy "test" rights', 'ca6 "foresting" service',
         'abc"def"yz.', 'A"B"'])
    targets = stringlist2tokens(
        ['"inside" over', '"test" rights', '"foresting" service', '"def"yz.',
         '"B"'])
    self._test_cases = list(zip(inputs, targets))

  def make_io_set(self):
    return copy.deepcopy(self._test_cases)
