# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tensorflow implementation to solve the Linear Sum Assignment problem.

The Linear Sum Assignment problem involves determining the minimum weight
matching for bipartite graphs. For example, this problem can be defined by
a 2D matrix C, where each element i,j determines the cost of matching worker i
with job j. The solution to the problem is a complete assignment of jobs to
workers, such that no job is assigned to more than one work and no worker is
assigned more than one job, with minimum cost.

This implementation builds off of the Hungarian
Matching Algorithm (https://www.cse.ust.hk/~golin/COMP572/Notes/Matching.pdf).

Based on the original implementation by Jiquan Ngiam <jngiam@google.com>.
"""
import tensorflow as tf, tf_keras
from official.modeling import tf_utils


def _prepare(weights):
  """Prepare the cost matrix.

  To speed up computational efficiency of the algorithm, all weights are shifted
  to be non-negative. Each element is reduced by the row / column minimum. Note
  that neither operation will effect the resulting solution but will provide
  a better starting point for the greedy assignment. Note this corresponds to
  the pre-processing and step 1 of the Hungarian algorithm from Wikipedia.

  Args:
    weights: A float32 [batch_size, num_elems, num_elems] tensor, where each
      inner matrix represents weights to be use for matching.

  Returns:
    A prepared weights tensor of the same shape and dtype.
  """
  # Since every worker needs a job and every job needs a worker, we can subtract
  # the minimum from each.
  weights -= tf.reduce_min(weights, axis=2, keepdims=True)
  weights -= tf.reduce_min(weights, axis=1, keepdims=True)
  return weights


def _greedy_assignment(adj_matrix):
  """Greedily assigns workers to jobs based on an adjaceny matrix.

  Starting with an adjacency matrix representing the available connections
  in the bi-partite graph, this function greedily chooses elements such
  that each worker is matched to at most one job (or each job is assigned to
  at most one worker). Note, if the adjacency matrix has no available values
  for a particular row/column, the corresponding job/worker may go unassigned.

  Args:
    adj_matrix: A bool [batch_size, num_elems, num_elems] tensor, where each
      element of the inner matrix represents whether the worker (row) can be
      matched to the job (column).

  Returns:
    A bool [batch_size, num_elems, num_elems] tensor, where each element of the
    inner matrix represents whether the worker has been matched to the job.
    Each row and column can have at most one true element. Some of the rows
    and columns may not be matched.
  """
  _, num_elems, _ = tf_utils.get_shape_list(adj_matrix, expected_rank=3)
  adj_matrix = tf.transpose(adj_matrix, [1, 0, 2])

  # Create a dynamic TensorArray containing the assignments for each worker/job
  assignment = tf.TensorArray(tf.bool, num_elems)

  # Store the elements assigned to each column to update each iteration
  col_assigned = tf.zeros_like(adj_matrix[0, ...], dtype=tf.bool)

  # Iteratively assign each row using tf.foldl. Intuitively, this is a loop
  # over rows, where we incrementally assign each row.
  def _assign_row(accumulator, row_adj):
    # The accumulator tracks the row assignment index.
    idx, assignment, col_assigned = accumulator

    # Viable candidates cannot already be assigned to another job.
    candidates = row_adj & (~col_assigned)

    # Deterministically assign to the candidates of the highest index count.
    max_candidate_idx = tf.argmax(
        tf.cast(candidates, tf.int32), axis=1, output_type=tf.int32)

    candidates_indicator = tf.one_hot(
        max_candidate_idx,
        num_elems,
        on_value=True,
        off_value=False,
        dtype=tf.bool)
    candidates_indicator &= candidates

    # Make assignment to the column.
    col_assigned |= candidates_indicator
    assignment = assignment.write(idx, candidates_indicator)

    return (idx + 1, assignment, col_assigned)

  _, assignment, _ = tf.foldl(
      _assign_row, adj_matrix, (0, assignment, col_assigned), back_prop=False)

  assignment = assignment.stack()
  assignment = tf.transpose(assignment, [1, 0, 2])
  return assignment


def _find_augmenting_path(assignment, adj_matrix):
  """Finds an augmenting path given an assignment and an adjacency matrix.

  The augmenting path search starts from the unassigned workers, then goes on
  to find jobs (via an unassigned pairing), then back again to workers (via an
  existing pairing), and so on. The path alternates between unassigned and
  existing pairings. Returns the state after the search.

  Note: In the state the worker and job, indices are 1-indexed so that we can
  use 0 to represent unreachable nodes. State contains the following keys:

  - jobs: A [batch_size, 1, num_elems] tensor containing the highest index
      unassigned worker that can reach this job through a path.
  - jobs_from_worker: A [batch_size, num_elems] tensor containing the worker
      reached immediately before this job.
  - workers: A [batch_size, num_elems, 1] tensor containing the highest index
      unassigned worker that can reach this worker through a path.
  - workers_from_job: A [batch_size, num_elems] tensor containing the job
      reached immediately before this worker.
  - new_jobs: A bool [batch_size, num_elems] tensor containing True if the
      unassigned job can be reached via a path.

  State can be used to recover the path via backtracking.

  Args:
    assignment: A bool [batch_size, num_elems, num_elems] tensor, where each
      element of the inner matrix represents whether the worker has been matched
      to the job. This may be a partial assignment.
    adj_matrix: A bool [batch_size, num_elems, num_elems] tensor, where each
      element of the inner matrix represents whether the worker (row) can be
      matched to the job (column).

  Returns:
    A state dict, which represents the outcome of running an augmenting
    path search on the graph given the assignment.
  """
  batch_size, num_elems, _ = tf_utils.get_shape_list(
      assignment, expected_rank=3)
  unassigned_workers = ~tf.reduce_any(assignment, axis=2, keepdims=True)
  unassigned_jobs = ~tf.reduce_any(assignment, axis=1, keepdims=True)

  unassigned_pairings = tf.cast(adj_matrix & ~assignment, tf.int32)
  existing_pairings = tf.cast(assignment, tf.int32)

  # Initialize unassigned workers to have non-zero ids, assigned workers will
  # have ids = 0.
  worker_indices = tf.range(1, num_elems + 1, dtype=tf.int32)
  init_workers = tf.tile(worker_indices[tf.newaxis, :, tf.newaxis],
                         [batch_size, 1, 1])
  init_workers *= tf.cast(unassigned_workers, tf.int32)

  state = {
      "jobs": tf.zeros((batch_size, 1, num_elems), dtype=tf.int32),
      "jobs_from_worker": tf.zeros((batch_size, num_elems), dtype=tf.int32),
      "workers": init_workers,
      "workers_from_job": tf.zeros((batch_size, num_elems), dtype=tf.int32)
  }

  def _has_active_workers(state, curr_workers):
    """Check if there are still active workers."""
    del state
    return tf.reduce_sum(curr_workers) > 0

  def _augment_step(state, curr_workers):
    """Performs one search step."""

    # Note: These steps could be potentially much faster if sparse matrices are
    # supported. The unassigned_pairings and existing_pairings matrices can be
    # very sparse.

    # Find potential jobs using current workers.
    potential_jobs = curr_workers * unassigned_pairings
    curr_jobs = tf.reduce_max(potential_jobs, axis=1, keepdims=True)
    curr_jobs_from_worker = 1 + tf.argmax(
        potential_jobs, axis=1, output_type=tf.int32)

    # Remove already accessible jobs from curr_jobs.
    default_jobs = tf.zeros_like(state["jobs"], dtype=state["jobs"].dtype)
    curr_jobs = tf.where(state["jobs"] > 0, default_jobs, curr_jobs)
    curr_jobs_from_worker *= tf.cast(curr_jobs > 0, tf.int32)[:, 0, :]

    # Find potential workers from current jobs.
    potential_workers = curr_jobs * existing_pairings
    curr_workers = tf.reduce_max(potential_workers, axis=2, keepdims=True)
    curr_workers_from_job = 1 + tf.argmax(
        potential_workers, axis=2, output_type=tf.int32)

    # Remove already accessible workers from curr_workers.
    default_workers = tf.zeros_like(state["workers"])
    curr_workers = tf.where(
        state["workers"] > 0, default_workers, curr_workers)
    curr_workers_from_job *= tf.cast(curr_workers > 0, tf.int32)[:, :, 0]

    # Update state so that we can backtrack later.
    state = state.copy()
    state["jobs"] = tf.maximum(state["jobs"], curr_jobs)
    state["jobs_from_worker"] = tf.maximum(state["jobs_from_worker"],
                                           curr_jobs_from_worker)
    state["workers"] = tf.maximum(state["workers"], curr_workers)
    state["workers_from_job"] = tf.maximum(state["workers_from_job"],
                                           curr_workers_from_job)

    return state, curr_workers

  state, _ = tf.while_loop(
      _has_active_workers,
      _augment_step, (state, init_workers),
      back_prop=False)

  # Compute new jobs, this is useful for determnining termnination of the
  # maximum bi-partite matching and initialization for backtracking.
  new_jobs = (state["jobs"] > 0) & unassigned_jobs
  state["new_jobs"] = new_jobs[:, 0, :]
  return state


def _improve_assignment(assignment, state):
  """Improves an assignment by backtracking the augmented path using state.

  Args:
    assignment: A bool [batch_size, num_elems, num_elems] tensor, where each
      element of the inner matrix represents whether the worker has been matched
      to the job. This may be a partial assignment.
    state: A dict, which represents the outcome of running an augmenting path
      search on the graph given the assignment.

  Returns:
    A new assignment matrix of the same shape and type as assignment, where the
    assignment has been updated using the augmented path found.
  """
  batch_size, num_elems, _ = tf_utils.get_shape_list(assignment, 3)

  # We store the current job id and iteratively backtrack using jobs_from_worker
  # and workers_from_job until we reach an unassigned worker. We flip all the
  # assignments on this path to discover a better overall assignment.

  # Note: The indices in state are 1-indexed, where 0 represents that the
  # worker / job cannot be reached.

  # Obtain initial job indices based on new_jobs.
  curr_job_idx = tf.argmax(
      tf.cast(state["new_jobs"], tf.int32), axis=1, output_type=tf.int32)

  # Track whether an example is actively being backtracked. Since we are
  # operating on a batch, not all examples in the batch may be active.
  active = tf.gather(state["new_jobs"], curr_job_idx, batch_dims=1)
  batch_range = tf.range(0, batch_size, dtype=tf.int32)

  # Flip matrix tracks which assignments we need to flip - corresponding to the
  # augmenting path taken. We use an integer tensor here so that we can use
  # tensor_scatter_nd_add to update the tensor, and then cast it back to bool
  # after the loop.
  flip_matrix = tf.zeros((batch_size, num_elems, num_elems), dtype=tf.int32)

  def _has_active_backtracks(flip_matrix, active, curr_job_idx):
    """Check if there are still active workers."""
    del flip_matrix, curr_job_idx
    return tf.reduce_any(active)

  def _backtrack_one_step(flip_matrix, active, curr_job_idx):
    """Take one step in backtracking."""
    # Discover the worker that the job originated from, note that this worker
    # must exist by construction.
    curr_worker_idx = tf.gather(
        state["jobs_from_worker"], curr_job_idx, batch_dims=1) - 1
    curr_worker_idx = tf.maximum(curr_worker_idx, 0)
    update_indices = tf.stack([batch_range, curr_worker_idx, curr_job_idx],
                              axis=1)
    update_indices = tf.maximum(update_indices, 0)
    flip_matrix = tf.tensor_scatter_nd_add(flip_matrix, update_indices,
                                           tf.cast(active, tf.int32))

    # Discover the (potential) job that the worker originated from.
    curr_job_idx = tf.gather(
        state["workers_from_job"], curr_worker_idx, batch_dims=1) - 1
    # Note that jobs may not be active, and we track that here (before
    # adjusting indices so that they are all >= 0 for gather).
    active &= curr_job_idx >= 0
    curr_job_idx = tf.maximum(curr_job_idx, 0)
    update_indices = tf.stack([batch_range, curr_worker_idx, curr_job_idx],
                              axis=1)
    update_indices = tf.maximum(update_indices, 0)
    flip_matrix = tf.tensor_scatter_nd_add(flip_matrix, update_indices,
                                           tf.cast(active, tf.int32))

    return flip_matrix, active, curr_job_idx

  flip_matrix, _, _ = tf.while_loop(
      _has_active_backtracks,
      _backtrack_one_step, (flip_matrix, active, curr_job_idx),
      back_prop=False)

  flip_matrix = tf.cast(flip_matrix, tf.bool)
  assignment = tf.math.logical_xor(assignment, flip_matrix)

  return assignment


def _maximum_bipartite_matching(adj_matrix, assignment=None):
  """Performs maximum bipartite matching using augmented paths.

  Args:
    adj_matrix: A bool [batch_size, num_elems, num_elems] tensor, where each
      element of the inner matrix represents whether the worker (row) can be
      matched to the job (column).
    assignment: An optional bool [batch_size, num_elems, num_elems] tensor,
      where each element of the inner matrix represents whether the worker has
      been matched to the job. This may be a partial assignment. If specified,
      this assignment will be used to seed the iterative algorithm.

  Returns:
    A state dict representing the final augmenting path state search, and
    a maximum bipartite matching assignment tensor. Note that the state outcome
    can be used to compute a minimum vertex cover for the bipartite graph.
  """

  if assignment is None:
    assignment = _greedy_assignment(adj_matrix)

  state = _find_augmenting_path(assignment, adj_matrix)

  def _has_new_jobs(state, assignment):
    del assignment
    return tf.reduce_any(state["new_jobs"])

  def _improve_assignment_and_find_new_path(state, assignment):
    assignment = _improve_assignment(assignment, state)
    state = _find_augmenting_path(assignment, adj_matrix)
    return state, assignment

  state, assignment = tf.while_loop(
      _has_new_jobs,
      _improve_assignment_and_find_new_path, (state, assignment),
      back_prop=False)

  return state, assignment


def _compute_cover(state, assignment):
  """Computes a cover for the bipartite graph.

  We compute a cover using the construction provided at
  https://en.wikipedia.org/wiki/K%C5%91nig%27s_theorem_(graph_theory)#Proof
  which uses the outcome from the alternating path search.

  Args:
    state: A state dict, which represents the outcome of running an augmenting
      path search on the graph given the assignment.
    assignment: An optional bool [batch_size, num_elems, num_elems] tensor,
      where each element of the inner matrix represents whether the worker has
      been matched to the job. This may be a partial assignment. If specified,
      this assignment will be used to seed the iterative algorithm.

  Returns:
    A tuple of (workers_cover, jobs_cover) corresponding to row and column
    covers for the bipartite graph. workers_cover is a boolean tensor of shape
    [batch_size, num_elems, 1] and jobs_cover is a boolean tensor of shape
    [batch_size, 1, num_elems].
  """
  assigned_workers = tf.reduce_any(assignment, axis=2, keepdims=True)
  assigned_jobs = tf.reduce_any(assignment, axis=1, keepdims=True)

  reachable_workers = state["workers"] > 0
  reachable_jobs = state["jobs"] > 0

  workers_cover = assigned_workers & (~reachable_workers)
  jobs_cover = assigned_jobs & reachable_jobs

  return workers_cover, jobs_cover


def _update_weights_using_cover(workers_cover, jobs_cover, weights):
  """Updates weights for hungarian matching using a cover.

  We first find the minimum uncovered weight. Then, we subtract this from all
  the uncovered weights, and add it to all the doubly covered weights.

  Args:
    workers_cover: A boolean tensor of shape [batch_size, num_elems, 1].
    jobs_cover: A boolean tensor of shape [batch_size, 1, num_elems].
    weights: A float32 [batch_size, num_elems, num_elems] tensor, where each
      inner matrix represents weights to be use for matching.

  Returns:
    A new weight matrix with elements adjusted by the cover.
  """
  max_value = tf.reduce_max(weights)

  covered = workers_cover | jobs_cover
  double_covered = workers_cover & jobs_cover

  uncovered_weights = tf.where(covered,
                               tf.ones_like(weights) * max_value, weights)
  min_weight = tf.reduce_min(uncovered_weights, axis=[-2, -1], keepdims=True)

  add_weight = tf.where(double_covered,
                        tf.ones_like(weights) * min_weight,
                        tf.zeros_like(weights))
  sub_weight = tf.where(covered, tf.zeros_like(weights),
                        tf.ones_like(weights) * min_weight)

  return weights + add_weight - sub_weight


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  expected_rank_dict = {}
  if isinstance(expected_rank, int):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = len(tensor.shape)
  if actual_rank not in expected_rank_dict:
    raise ValueError(
        "For the tensor `%s`, the actual tensor rank `%d` (shape = %s) is not "
        "equal to the expected tensor rank `%s`" %
        (name, actual_rank, str(tensor.shape), str(expected_rank)))


def hungarian_matching(weights):
  """Computes the minimum linear sum assignment using the Hungarian algorithm.

  Args:
    weights: A float32 [batch_size, num_elems, num_elems] tensor, where each
      inner matrix represents weights to be use for matching.

  Returns:
    A bool [batch_size, num_elems, num_elems] tensor, where each element of the
    inner matrix represents whether the worker has been matched to the job.
    The returned matching will always be a perfect match.
  """
  batch_size, num_elems, _ = tf_utils.get_shape_list(weights, 3)

  weights = _prepare(weights)
  adj_matrix = tf.equal(weights, 0.)
  state, assignment = _maximum_bipartite_matching(adj_matrix)
  workers_cover, jobs_cover = _compute_cover(state, assignment)

  def _cover_incomplete(workers_cover, jobs_cover, *args):
    del args
    cover_sum = (
        tf.reduce_sum(tf.cast(workers_cover, tf.int32)) +
        tf.reduce_sum(tf.cast(jobs_cover, tf.int32)))
    return tf.less(cover_sum, batch_size * num_elems)

  def _update_weights_and_match(workers_cover, jobs_cover, weights, assignment):
    weights = _update_weights_using_cover(workers_cover, jobs_cover, weights)
    adj_matrix = tf.equal(weights, 0.)
    state, assignment = _maximum_bipartite_matching(adj_matrix, assignment)
    workers_cover, jobs_cover = _compute_cover(state, assignment)
    return workers_cover, jobs_cover, weights, assignment

  workers_cover, jobs_cover, weights, assignment = tf.while_loop(
      _cover_incomplete,
      _update_weights_and_match,
      (workers_cover, jobs_cover, weights, assignment),
      back_prop=False)
  return weights, assignment

