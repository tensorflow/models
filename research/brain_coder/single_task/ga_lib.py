from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Genetic algorithm for BF tasks.

Inspired by https://github.com/primaryobjects/AI-Programmer.
GA function code borrowed from https://github.com/DEAP/deap.
"""

from collections import namedtuple
import random

from absl import flags
from absl import logging
import numpy as np
from six.moves import xrange

from common import bf  # brain coder
from common import utils  # brain coder
from single_task import misc  # brain coder

FLAGS = flags.FLAGS

# Saving reward of previous programs saves computation if a program appears
# again.
USE_REWARD_CACHE = True  # Disable this if GA is using up too much memory.
GENES = bf.CHARS
MAX_PROGRAM_STEPS = 500
STEP_BONUS = True

ALPHANUM_CHARS = (
    ['_'] +
    [chr(ord('a') + i_) for i_ in range(26)] +
    [chr(ord('A') + i_) for i_ in range(26)] +
    [chr(ord('0') + i_) for i_ in range(10)])

Result = namedtuple(
    'Result',
    ['reward', 'inputs', 'code_outputs', 'target_outputs', 'type_in',
     'type_out', 'base', 'correct'])


class IOType(object):
  string = 'string'
  integer = 'integer'


class CustomType(object):

  def __init__(self, to_str_fn):
    self.to_str_fn = to_str_fn

  def __call__(self, obj):
    return self.to_str_fn(obj)


def tokens_list_repr(tokens, repr_type, base):
  """Make human readable representation of program IO."""
  if isinstance(repr_type, CustomType):
    return repr_type(tokens)
  elif repr_type == IOType.string:
    chars = (
        [ALPHANUM_CHARS[t] for t in tokens] if base < len(ALPHANUM_CHARS)
        else [chr(t) for t in tokens])
    return ''.join(chars)
  elif repr_type == IOType.integer:
    return str(tokens)
  raise ValueError('No such representation type "%s"', repr_type)


def io_repr(result):
  """Make human readable representation of test cases."""
  inputs = ','.join(
      tokens_list_repr(tokens, result.type_in, result.base)
      for tokens in result.inputs)
  code_outputs = ','.join(
      tokens_list_repr(tokens, result.type_out, result.base)
      for tokens in result.code_outputs)
  target_outputs = ','.join(
      tokens_list_repr(tokens, result.type_out, result.base)
      for tokens in result.target_outputs)
  return inputs, target_outputs, code_outputs


def make_task_eval_fn(task_manager):
  """Returns a wrapper that converts an RL task into a GA task.

  Args:
    task_manager: Is a task manager object from code_tasks.py

  Returns:
    A function that takes as input a single list of a code chars, and outputs
    a Result namedtuple instance containing the reward and information about
    code execution.
  """
  def to_data_list(single_or_tuple):
    if isinstance(single_or_tuple, misc.IOTuple):
      return list(single_or_tuple)
    return [single_or_tuple]

  def to_ga_type(rl_type):
    if rl_type == misc.IOType.string:
      return IOType.string
    return IOType.integer

  # Wrapper function.
  def evalbf(bf_chars):
    result = task_manager._score_code(''.join(bf_chars))
    reward = sum(result.episode_rewards)
    correct = result.reason == 'correct'
    return Result(
        reward=reward,
        inputs=to_data_list(result.input_case),
        code_outputs=to_data_list(result.code_output),
        target_outputs=to_data_list(result.correct_output),
        type_in=to_ga_type(result.input_type),
        type_out=to_ga_type(result.output_type),
        correct=correct,
        base=task_manager.task.base)

  return evalbf


def debug_str(individual, task_eval_fn):
  res = task_eval_fn(individual)
  input_str, target_output_str, code_output_str = io_repr(res)
  return (
      ''.join(individual) +
      ' | ' + input_str +
      ' | ' + target_output_str +
      ' | ' + code_output_str +
      ' | ' + str(res.reward) +
      ' | ' + str(res.correct))


def mutate_single(code_tokens, mutation_rate):
  """Mutate a single code string.

  Args:
    code_tokens: A string/list/Individual of BF code chars. Must end with EOS
        symbol '_'.
    mutation_rate: Float between 0 and 1 which sets the probability of each char
        being mutated.

  Returns:
    An Individual instance containing the mutated code string.

  Raises:
    ValueError: If `code_tokens` does not end with EOS symbol.
  """
  if len(code_tokens) <= 1:
    return code_tokens
  if code_tokens[-1] == '_':
    # Do this check to ensure that the code strings have not been corrupted.
    raise ValueError('`code_tokens` must end with EOS symbol.')
  else:
    cs = Individual(code_tokens)
    eos = []
  mutated = False
  for pos in range(len(cs)):
    if random.random() < mutation_rate:
      mutated = True
      new_char = GENES[random.randrange(len(GENES))]
      x = random.random()
      if x < 0.25 and pos != 0 and pos != len(cs) - 1:
        # Insertion mutation.
        if random.random() < 0.50:
          # Shift up.
          cs = cs[:pos] + [new_char] + cs[pos:-1]
        else:
          # Shift down.
          cs = cs[1:pos] + [new_char] + cs[pos:]
      elif x < 0.50:
        # Deletion mutation.
        if random.random() < 0.50:
          # Shift down.
          cs = cs[:pos] + cs[pos + 1:] + [new_char]
        else:
          # Shift up.
          cs = [new_char] + cs[:pos] + cs[pos + 1:]
      elif x < 0.75:
        # Shift rotate mutation (position invariant).
        if random.random() < 0.50:
          # Shift down.
          cs = cs[1:] + [cs[0]]
        else:
          # Shift up.
          cs = [cs[-1]] + cs[:-1]
      else:
        # Replacement mutation.
        cs = cs[:pos] + [new_char] + cs[pos + 1:]
  assert len(cs) + len(eos) == len(code_tokens)
  if mutated:
    return Individual(cs + eos)
  else:
    return Individual(code_tokens)


def crossover(parent1, parent2):
  """Performs crossover mating between two code strings.

  Crossover mating is where a random position is selected, and the chars
  after that point are swapped. The resulting new code strings are returned.

  Args:
    parent1: First code string.
    parent2: Second code string.

  Returns:
    A 2-tuple of children, i.e. the resulting code strings after swapping.
  """
  max_parent, min_parent = (
      (parent1, parent2) if len(parent1) > len(parent2)
      else (parent2, parent1))
  pos = random.randrange(len(max_parent))
  if pos >= len(min_parent):
    child1 = max_parent[:pos]
    child2 = min_parent + max_parent[pos:]
  else:
    child1 = max_parent[:pos] + min_parent[pos:]
    child2 = min_parent[:pos] + max_parent[pos:]
  return Individual(child1), Individual(child2)


def _make_even(n):
  """Return largest even integer less than or equal to `n`."""
  return (n >> 1) << 1


def mutate_and_crossover(population, mutation_rate, crossover_rate):
  """Take a generational step over a population.

  Transforms population of parents into population of children (of the same
  size) via crossover mating and then mutation on the resulting children.

  Args:
    population: Parent population. A list of Individual objects.
    mutation_rate: Probability of mutation. See `mutate_single`.
    crossover_rate: Probability that two parents will mate.

  Returns:
    Child population. A list of Individual objects.
  """
  children = [None] * len(population)
  for i in xrange(0, _make_even(len(population)), 2):
    p1 = population[i]
    p2 = population[i + 1]
    if random.random() < crossover_rate:
      p1, p2 = crossover(p1, p2)
    c1 = mutate_single(p1, mutation_rate)
    c2 = mutate_single(p2, mutation_rate)
    children[i] = c1
    children[i + 1] = c2
  if children[-1] is None:
    children[-1] = population[-1]
  return children


def ga_loop(population, cxpb, mutpb, ngen, task_eval_fn, halloffame=None,
            checkpoint_writer=None):
  """A bare bones genetic algorithm.

  Similar to chapter 7 of Back, Fogel and Michalewicz, "Evolutionary
  Computation 1 : Basic Algorithms and Operators", 2000.

  Args:
    population: A list of individuals.
    cxpb: The probability of mating two individuals.
    mutpb: The probability of mutating a gene.
    ngen: The number of generation. Unlimited if zero.
    task_eval_fn: A python function which maps an Individual to a Result
        namedtuple.
    halloffame: (optional) a utils.MaxUniquePriorityQueue object that will be
        used to aggregate the best individuals found during search.
    checkpoint_writer: (optional) an object that can save and load populations.
        Needs to have `write`, `load`, and `has_checkpoint` methods. Used to
        periodically save progress. In event of a restart, the population will
        be loaded from disk.

  Returns:
    GaResult namedtuple instance. This contains information about the GA run,
    including the resulting population, best reward (fitness) obtained, and
    the best code string found.
  """

  has_checkpoint = False
  if checkpoint_writer and checkpoint_writer.has_checkpoint():
    try:
      gen, population, halloffame = checkpoint_writer.load()
    except EOFError:  # Data was corrupted. Start over.
      pass
    else:
      has_checkpoint = True
      logging.info(
          'Loaded population from checkpoint. Starting at generation %d', gen)

      # Evaluate the individuals with an invalid fitness
      invalid_ind = [ind for ind in population if not ind.fitness.valid]
      for ind in invalid_ind:
        ind.fitness.values = task_eval_fn(ind).reward,
      for _, ind in halloffame.iter_in_order():
        ind.fitness.values = task_eval_fn(ind).reward,

  if not has_checkpoint:
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    for ind in invalid_ind:
      ind.fitness.values = task_eval_fn(ind).reward,

    if halloffame is not None:
      for ind in population:
        halloffame.push(ind.fitness.values, tuple(ind), ind)

    logging.info('Initialized new population.')

    gen = 1

  pop_size = len(population)
  program_reward_cache = {} if USE_REWARD_CACHE else None

  # Begin the generational process
  while ngen == 0 or gen <= ngen:
    # Select the next generation individuals
    offspring = roulette_selection(population, pop_size - len(halloffame))

    # Vary the pool of individuals
    # offspring = varAnd(offspring, toolbox, cxpb, mutpb)
    offspring = mutate_and_crossover(
        offspring, mutation_rate=mutpb, crossover_rate=cxpb)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_ind:
      str_repr = ''.join(ind)
      if program_reward_cache is not None and str_repr in program_reward_cache:
        ind.fitness.values = (program_reward_cache[str_repr],)
      else:
        eval_result = task_eval_fn(ind)
        ind.fitness.values = (eval_result.reward,)
        if program_reward_cache is not None:
          program_reward_cache[str_repr] = eval_result.reward

    # Replace the current population by the offspring
    population = list(offspring)

    # Update the hall of fame with the generated individuals
    if halloffame is not None:
      for ind in population:
        halloffame.push(ind.fitness.values, tuple(ind), ind)

    # elitism
    population.extend([ind for _, ind in halloffame.iter_in_order()])

    if gen % 100 == 0:
      top_code = '\n'.join([debug_str(ind, task_eval_fn)
                            for ind in topk(population, k=4)])
      logging.info('gen: %d\nNPE: %d\n%s\n\n', gen, gen * pop_size, top_code)

      best_code = ''.join(halloffame.get_max()[1])
      res = task_eval_fn(best_code)

      # Write population and hall-of-fame to disk.
      if checkpoint_writer:
        checkpoint_writer.write(gen, population, halloffame)

      if res.correct:
        logging.info('Solution found:\n%s\nreward = %s\n',
                     best_code, res.reward)
        break

    gen += 1

  best_code = ''.join(halloffame.get_max()[1])
  res = task_eval_fn(best_code)

  return GaResult(
      population=population, best_code=best_code, reward=res.reward,
      solution_found=res.correct, generations=gen,
      num_programs=gen * len(population),
      max_generations=ngen, max_num_programs=ngen * len(population))


GaResult = namedtuple(
    'GaResult',
    ['population', 'best_code', 'reward', 'generations', 'num_programs',
     'solution_found', 'max_generations', 'max_num_programs'])


def reward_conversion(reward):
  """Convert real value into positive value."""
  if reward <= 0:
    return 0.05
  return reward + 0.05


def roulette_selection(population, k):
  """Select `k` individuals with prob proportional to fitness.

  Each of the `k` selections is independent.

  Warning:
    The roulette selection by definition cannot be used for minimization
    or when the fitness can be smaller or equal to 0.

  Args:
    population: A list of Individual objects to select from.
    k: The number of individuals to select.

  Returns:
    A list of selected individuals.
  """
  fitnesses = np.asarray(
      [reward_conversion(ind.fitness.values[0])
       for ind in population])
  assert np.all(fitnesses > 0)

  sum_fits = fitnesses.sum()
  chosen = [None] * k
  for i in xrange(k):
    u = random.random() * sum_fits
    sum_ = 0
    for ind, fitness in zip(population, fitnesses):
      sum_ += fitness
      if sum_ > u:
        chosen[i] = Individual(ind)
        break
    if not chosen[i]:
      chosen[i] = Individual(population[-1])

  return chosen


def make_population(make_individual_fn, n):
  return [make_individual_fn() for _ in xrange(n)]


def best(population):
  best_ind = None
  for ind in population:
    if best_ind is None or best_ind.fitness.values < ind.fitness.values:
      best_ind = ind
  return best_ind


def topk(population, k):
  q = utils.MaxUniquePriorityQueue(k)
  for ind in population:
    q.push(ind.fitness.values, tuple(ind), ind)
  return [ind for _, ind in q.iter_in_order()]


class Fitness(object):

  def __init__(self):
    self.values = ()

  @property
  def valid(self):
    """Assess if a fitness is valid or not."""
    return bool(self.values)


class Individual(list):

  def __init__(self, *args):
    super(Individual, self).__init__(*args)
    self.fitness = Fitness()


def random_individual(genome_size):
  return lambda: Individual(np.random.choice(GENES, genome_size).tolist())
