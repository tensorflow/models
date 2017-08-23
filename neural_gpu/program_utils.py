# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Utilities for generating program synthesis and evaluation data."""

import contextlib
import sys
import StringIO
import random
import os

class ListType(object):
  def __init__(self, arg):
    self.arg = arg

  def __str__(self):
    return "[" + str(self.arg) + "]"

  def __eq__(self, other):
    if not isinstance(other, ListType):
      return False
    return self.arg == other.arg
  
  def __hash__(self):
    return hash(self.arg)

class VarType(object):
  def __init__(self, arg):
    self.arg = arg

  def __str__(self):
    return str(self.arg)

  def __eq__(self, other):
    if not isinstance(other, VarType):
      return False
    return self.arg == other.arg

  def __hash__(self):
    return hash(self.arg)

class FunctionType(object):
  def __init__(self, args):
    self.args = args

  def __str__(self):
    return str(self.args[0]) + " -> " + str(self.args[1])

  def __eq__(self, other):
    if not isinstance(other, FunctionType):
      return False
    return self.args == other.args

  def __hash__(self):
    return hash(tuple(self.args))


class Function(object):
  def __init__(self, name, arg_types, output_type, fn_arg_types = None):
    self.name = name 
    self.arg_types = arg_types
    self.fn_arg_types = fn_arg_types or []
    self.output_type = output_type

Null = 100
## Functions
f_head = Function("c_head", [ListType("Int")], "Int")
def c_head(xs): return xs[0] if len(xs) > 0 else Null

f_last = Function("c_last", [ListType("Int")], "Int")
def c_last(xs): return xs[-1] if len(xs) > 0 else Null

f_take = Function("c_take", ["Int", ListType("Int")], ListType("Int"))
def c_take(n, xs): return xs[:n]

f_drop = Function("c_drop", ["Int", ListType("Int")], ListType("Int"))
def c_drop(n, xs): return xs[n:]

f_access = Function("c_access", ["Int", ListType("Int")], "Int")
def c_access(n, xs): return xs[n] if n >= 0 and len(xs) > n else Null

f_max = Function("c_max", [ListType("Int")], "Int")
def c_max(xs): return max(xs) if len(xs) > 0 else Null

f_min = Function("c_min", [ListType("Int")], "Int")
def c_min(xs): return min(xs) if len(xs) > 0 else Null

f_reverse = Function("c_reverse", [ListType("Int")], ListType("Int"))
def c_reverse(xs): return list(reversed(xs))

f_sort = Function("sorted", [ListType("Int")], ListType("Int"))
# def c_sort(xs): return sorted(xs)

f_sum = Function("sum", [ListType("Int")], "Int")
# def c_sum(xs): return sum(xs)


## Lambdas
# Int -> Int
def plus_one(x): return x + 1
def minus_one(x): return x - 1
def times_two(x): return x * 2
def neg(x): return x * (-1)
def div_two(x): return int(x/2)
def sq(x): return x**2 
def times_three(x): return x * 3
def div_three(x): return int(x/3)
def times_four(x): return x * 4
def div_four(x): return int(x/4)

# Int -> Bool 
def pos(x): return x > 0 
def neg(x): return x < 0
def even(x): return x%2 == 0
def odd(x): return x%2 == 1

# Int -> Int -> Int
def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y

# HOFs
f_map = Function("map", [ListType("Int")], 
                        ListType("Int"), 
                        [FunctionType(["Int", "Int"])])
f_filter = Function("filter", [ListType("Int")], 
                              ListType("Int"), 
                              [FunctionType(["Int", "Bool"])])
f_count = Function("c_count", [ListType("Int")], 
                              "Int", 
                              [FunctionType(["Int", "Bool"])])
def c_count(f, xs): return len([x for x in xs if f(x)])

f_zipwith = Function("c_zipwith", [ListType("Int"), ListType("Int")], 
                                  ListType("Int"), 
                                  [FunctionType(["Int", "Int", "Int"])]) #FIX
def c_zipwith(f, xs, ys): return [f(x, y) for (x, y) in zip(xs, ys)]

f_scan = Function("c_scan", [ListType("Int")],
                            ListType("Int"), 
                            [FunctionType(["Int", "Int", "Int"])])
def c_scan(f, xs):
  out = xs
  for i in range(1, len(xs)):
    out[i] = f(xs[i], xs[i -1])
  return out

@contextlib.contextmanager
def stdoutIO(stdout=None):
  old = sys.stdout
  if stdout is None:
    stdout = StringIO.StringIO()
  sys.stdout = stdout
  yield stdout
  sys.stdout = old


def evaluate(program_str, input_names_to_vals, default="ERROR"):
  exec_str = []
  for name, val in input_names_to_vals.iteritems():
    exec_str += name + " = " + str(val) + "; "
  exec_str += program_str
  if type(exec_str) is list:
    exec_str = "".join(exec_str)

  with stdoutIO() as s:
    # pylint: disable=bare-except
    try:
      exec exec_str + " print(out)"
      return s.getvalue()[:-1]
    except:
      return default
   # pylint: enable=bare-except


class Statement(object):
  """Statement class."""
  
  def __init__(self, fn, output_var, arg_vars, fn_args=None):
    self.fn = fn
    self.output_var = output_var
    self.arg_vars = arg_vars
    self.fn_args = fn_args or []

  def __str__(self):
    return "%s = %s(%s%s%s)"%(self.output_var,
                              self.fn.name,
                              ", ".join(self.fn_args),
                              ", " if self.fn_args else "",
                              ", ".join(self.arg_vars))

  def substitute(self, env):
    self.output_var = env.get(self.output_var, self.output_var)
    self.arg_vars = [env.get(v, v) for v in self.arg_vars]


class ProgramGrower(object):
  """Grow programs."""

  def __init__(self, functions, types_to_lambdas):
    self.functions = functions
    self.types_to_lambdas = types_to_lambdas

  def grow_body(self, new_var_name, dependencies, types_to_vars):
    """Grow the program body."""
    choices = []
    for f in self.functions:
      if all([a in types_to_vars.keys() for a in f.arg_types]):
        choices.append(f)

    f = random.choice(choices)
    args = []
    for t in f.arg_types:
      possible_vars = random.choice(types_to_vars[t])
      var = random.choice(possible_vars)
      args.append(var)
      dependencies.setdefault(new_var_name, []).extend(
          [var] + (dependencies[var]))

    fn_args = [random.choice(self.types_to_lambdas[t]) for t in f.fn_arg_types]
    types_to_vars.setdefault(f.output_type, []).append(new_var_name)

    return Statement(f, new_var_name, args, fn_args)

  def grow(self, program_len, input_types):
    """Grow the program."""
    var_names = list(reversed(map(chr, range(97, 123))))
    dependencies = dict()
    types_to_vars = dict()
    input_names = []
    for t in input_types:
      var = var_names.pop()
      dependencies[var] = []
      types_to_vars.setdefault(t, []).append(var)
      input_names.append(var)

    statements = []
    for _ in range(program_len - 1):
      var = var_names.pop()
      statements.append(self.grow_body(var, dependencies, types_to_vars))
    statements.append(self.grow_body("out", dependencies, types_to_vars))

    new_var_names = [c for c in map(chr, range(97, 123))
                     if c not in input_names]
    new_var_names.reverse()
    keep_statements = []
    env = dict()
    for s in statements:
      if s.output_var in dependencies["out"]:
        keep_statements.append(s)
        env[s.output_var] = new_var_names.pop()
      if s.output_var == "out":
        keep_statements.append(s)

    for k in keep_statements:
      k.substitute(env)

    return Program(input_names, input_types, ";".join(
        [str(k) for k in keep_statements]))


class Program(object):
  """The program class."""

  def __init__(self, input_names, input_types, body):
    self.input_names = input_names
    self.input_types = input_types
    self.body = body

  def evaluate(self, inputs):
    """Evaluate this program."""
    if len(inputs) != len(self.input_names):
      raise AssertionError("inputs and input_names have to"
                           "have the same len. inp: %s , names: %s" %
                           (str(inputs), str(self.input_names)))
    inp_str = ""
    for (name, inp) in zip(self.input_names, inputs):
      inp_str += name + " = " + str(inp) + "; "

    with stdoutIO() as s:
      # pylint: disable=exec-used
      exec inp_str + self.body + "; print(out)"
      # pylint: enable=exec-used
    return s.getvalue()[:-1]

  def flat_str(self):
    out = ""
    for s in self.body.split(";"):
      out += s + ";"
    return out

  def __str__(self):
    out = ""
    for (n, t) in zip(self.input_names, self.input_types):
      out += n + " = " + str(t) + "\n"
    for s in self.body.split(";"):
      out += s + "\n"
    return out


prog_vocab = []
prog_rev_vocab = {}


def tokenize(string, tokens=None):
  """Tokenize the program string."""
  if tokens is None:
    tokens = prog_vocab
  tokens = sorted(tokens, key=len, reverse=True)
  out = []
  string = string.strip()
  while string:
    found = False
    for t in tokens:
      if string.startswith(t):
        out.append(t)
        string = string[len(t):]
        found = True
        break
    if not found:
      raise ValueError("Couldn't tokenize this: " + string)
    string = string.strip()
  return out


def clean_up(output, max_val=100):
  o = eval(str(output))
  if isinstance(o, bool):
    return o
  if isinstance(o, int):
    if o >= 0:
      return min(o, max_val)
    else:
      return max(o, -1 * max_val)
  if isinstance(o, list):
    return [clean_up(l) for l in o]


def make_vocab():
  gen(2, 0)


def gen(max_len, how_many):
  """Generate some programs."""
  functions = [f_head, f_last, f_take, f_drop, f_access, f_max, f_min,
               f_reverse, f_sort, f_sum, f_map, f_filter, f_count, f_zipwith,
               f_scan]

  types_to_lambdas = {
      FunctionType(["Int", "Int"]): ["plus_one", "minus_one", "times_two",
                                     "div_two", "sq", "times_three",
                                     "div_three", "times_four", "div_four"],
      FunctionType(["Int", "Bool"]): ["pos", "neg", "even", "odd"],
      FunctionType(["Int", "Int", "Int"]): ["add", "sub", "mul"]
  }

  tokens = []
  for f in functions:
    tokens.append(f.name)
  for v in types_to_lambdas.values():
    tokens.extend(v)
  tokens.extend(["=", ";", ",", "(", ")", "[", "]", "Int", "out"])
  tokens.extend(map(chr, range(97, 123)))

  io_tokens = map(str, range(-220, 220))
  if not prog_vocab:
    prog_vocab.extend(["_PAD", "_EOS"] + tokens + io_tokens)
    for i, t in enumerate(prog_vocab):
      prog_rev_vocab[t] = i

  io_tokens += [",", "[", "]", ")", "(", "None"]
  grower = ProgramGrower(functions=functions,
                         types_to_lambdas=types_to_lambdas)

  def mk_inp(l):
    return [random.choice(range(-5, 5)) for _ in range(l)]

  tar = [ListType("Int")]
  inps = [[mk_inp(3)], [mk_inp(5)], [mk_inp(7)], [mk_inp(15)]]

  save_prefix = None
  outcomes_to_programs = dict()
  tried = set()
  counter = 0
  choices = [0] if max_len == 0 else range(max_len)
  while counter < 100 * how_many and len(outcomes_to_programs) < how_many:
    counter += 1
    length = random.choice(choices)
    t = grower.grow(length, tar)
    while t in tried:
      length = random.choice(choices)
      t = grower.grow(length, tar)
    # print(t.flat_str())
    tried.add(t)
    outcomes = [clean_up(t.evaluate(i)) for i in inps]
    outcome_str = str(zip(inps, outcomes))
    if outcome_str in outcomes_to_programs:
      outcomes_to_programs[outcome_str] = min(
          [t.flat_str(), outcomes_to_programs[outcome_str]],
          key=lambda x: len(tokenize(x, tokens)))
    else:
      outcomes_to_programs[outcome_str] = t.flat_str()
    if counter % 5000 == 0:
      print "== proggen: tried: " + str(counter)
      print "== proggen: kept:  " + str(len(outcomes_to_programs))

    if counter % 250000 == 0 and save_prefix is not None:
      print "saving..."
      save_counter = 0
      progfilename = os.path.join(save_prefix, "prog_" + str(counter) + ".txt")
      iofilename = os.path.join(save_prefix, "io_" + str(counter) + ".txt")
      prog_token_filename = os.path.join(save_prefix,
                                         "prog_tokens_" + str(counter) + ".txt")
      io_token_filename = os.path.join(save_prefix,
                                       "io_tokens_" + str(counter) + ".txt")
      with open(progfilename, "a+") as fp,  \
           open(iofilename, "a+") as fi, \
           open(prog_token_filename, "a+") as ftp, \
           open(io_token_filename, "a+") as fti:
        for (o, p) in outcomes_to_programs.iteritems():
          save_counter += 1
          if save_counter % 500 == 0:
            print "saving %d of %d" % (save_counter, len(outcomes_to_programs))
          fp.write(p+"\n")
          fi.write(o+"\n")
          ftp.write(str(tokenize(p, tokens))+"\n")
          fti.write(str(tokenize(o, io_tokens))+"\n")

  return list(outcomes_to_programs.values())
