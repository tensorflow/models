
import time
import os
import json
import cProfile
import pstats
import io
import uuid



TIMERS = {}
PROFILERS = {}

FILE_HANDLE = None
LAST_FLUSH = None


def _time_now():
    return time.time()


class Timer(object):
    def __init__(self):
        self.start_ts = None
        self.num_starts = 0
        self.history = []
        self.cumulative = 0
        self.total = 0

    def start(self):
        if self.start_ts is not None:
            return
        self.num_starts += 1
        self.start_ts = _time_now()

    def pause(self):
        if self.start_ts is None:
            return
        delta = _time_now() - self.start_ts
        self.cumulative += delta
        self.total += delta
        self.start_ts = None

    def stop(self):
        if self.start_ts is not None:
            self.pause()
        if self.cumulative > 0:
            self.history.append(self.cumulative)
        self.cumulative = 0
        self.start_ts = None

    def read_total(self):
        delta = 0
        if self.start_ts is not None:
            delta = _time_now() - self.start_ts
        return self.total + delta

    def read_time(self):
        delta = 0
        if self.start_ts is not None:
            delta = _time_now() - self.start_ts
        return self.cumulative + delta


def start_time(name):
    if name not in TIMERS:
        TIMERS[name] = Timer()
    TIMERS[name].start()
    record_timer_event(name, 'start')


def pause_time(name):
    if name not in TIMERS:
        return
    TIMERS[name].pause()


def stop_time(name):
    if name not in TIMERS:
        return
    record_timer_event(name, 'stop')
    TIMERS[name].stop()


def read_timers_into_json():
    times = {}
    for name in TIMERS:
        timer = TIMERS[name]
        times[name] ={'name': name, 'total': timer.read_total(), 'time': timer.read_time(), 'run': timer.num_starts}
    return times


def record_timer_event(name, event):
    times = read_timers_into_json()
    json = {'type': 'timer', 'event': event, 'times': times, 'name': name, 'wall_time': _time_now() }
    _log(json)


def record(name, val):
    times = read_timers_into_json()
    json = {'type': 'value', 'times': times, 'name': name, 'value': val, 'wall_time': _time_now()}
    _log(json)


def record_const(name, val):
    times = read_timers_into_json()
    json = {'type': 'const', 'name': name, 'value': val, 'wall_time': _time_now(), 'times': times}
    _log(json)

def create_main_profiler():
  pr = cProfile.Profile()
  pr.enable()
  PROFILERS['__main__'] = pr


def record_profiler():
  pr = PROFILERS['__main__']
  pr.disable()
  s = io.StringIO()
  sortby = 'cumulative'
  ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
  pr.enable()
  ps.print_stats()
  times = read_timers_into_json()
  json = {'type': 'profile', 'name': '__main__', 'val': s.getvalue(), 'wall_time': _time_now(), 'times': times}
  _log(json)



def end():
    if FILE_HANDLE:
        FILE_HANDLE.close()


def _log(j):
    if FILE_HANDLE:
        FILE_HANDLE.write(json.dumps(j) + '\n')
    if LAST_FLUSH is None or time.time() - LAST_FLUSH > 60.0:
        _flush()


def _flush():
    if FILE_HANDLE:
        FILE_HANDLE.flush()
    global LAST_FLUSH
    LAST_FLUSH = time.time()


def start(path='/tmp/qmeas'):
    if not os.path.exists(path):
      os.mkdir(path)
    global FILE_HANDLE
    filename = os.path.join(path, '{}.json'.format(str(uuid.uuid4())))
    start_time('main')
    FILE_HANDLE = open(filename, 'w')


