"""A simple Python callstack sampler."""

import contextlib
import datetime
import signal
import traceback


class CallstackSampler(object):
  """A simple signal-based Python callstack sampler.
  """

  def __init__(self, interval=None):
    self.stacks = []
    self.interval = 0.001 if interval is None else interval

  def _sample(self, signum, frame):
    """Samples the current stack."""
    del signum
    stack = traceback.extract_stack(frame)
    formatted_stack = []
    formatted_stack.append(datetime.datetime.utcnow())
    for filename, lineno, function_name, text in stack:
      formatted_frame = '{}:{}({})({})'.format(filename, lineno, function_name,
                                               text)
      formatted_stack.append(formatted_frame)
    self.stacks.append(formatted_stack)
    signal.setitimer(signal.ITIMER_VIRTUAL, self.interval, 0)

  @contextlib.contextmanager
  def profile(self):
    signal.signal(signal.SIGVTALRM, self._sample)
    signal.setitimer(signal.ITIMER_VIRTUAL, self.interval, 0)
    try:
      yield
    finally:
      signal.setitimer(signal.ITIMER_VIRTUAL, 0)

  def save(self, fname):
    with open(fname, 'w') as f:
      for s in self.stacks:
        for l in s:
          f.write('%s\n' % l)
        f.write('\n')


@contextlib.contextmanager
def callstack_sampling(filename, interval=None):
  """Periodically samples the Python callstack.

  Args:
    filename: the filename
    interval: the sampling interval, in seconds. Defaults to 0.001.

  Yields:
   nothing
  """
  sampler = CallstackSampler(interval=interval)
  with sampler.profile():
    yield
  sampler.save(filename)

