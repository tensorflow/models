# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Tests for fivo.data.datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os

import numpy as np
import tensorflow as tf

from fivo.data import datasets

FLAGS = tf.app.flags.FLAGS


class DatasetsTest(tf.test.TestCase):

  def test_sparse_pianoroll_to_dense_empty_at_end(self):
    sparse_pianoroll = [(0, 1), (1, 0), (), (1,), (), ()]
    dense_pianoroll, num_timesteps = datasets.sparse_pianoroll_to_dense(
        sparse_pianoroll, min_note=0, num_notes=2)
    self.assertEqual(num_timesteps, 6)
    self.assertAllEqual([[1, 1],
                         [1, 1],
                         [0, 0],
                         [0, 1],
                         [0, 0],
                         [0, 0]], dense_pianoroll)

  def test_sparse_pianoroll_to_dense_with_chord(self):
    sparse_pianoroll = [(0, 1), (1, 0), (), (1,)]
    dense_pianoroll, num_timesteps = datasets.sparse_pianoroll_to_dense(
        sparse_pianoroll, min_note=0, num_notes=2)
    self.assertEqual(num_timesteps, 4)
    self.assertAllEqual([[1, 1],
                         [1, 1],
                         [0, 0],
                         [0, 1]], dense_pianoroll)

  def test_sparse_pianoroll_to_dense_simple(self):
    sparse_pianoroll = [(0,), (), (1,)]
    dense_pianoroll, num_timesteps = datasets.sparse_pianoroll_to_dense(
        sparse_pianoroll, min_note=0, num_notes=2)
    self.assertEqual(num_timesteps, 3)
    self.assertAllEqual([[1, 0],
                         [0, 0],
                         [0, 1]], dense_pianoroll)

  def test_sparse_pianoroll_to_dense_subtracts_min_note(self):
    sparse_pianoroll = [(4, 5), (5, 4), (), (5,), (), ()]
    dense_pianoroll, num_timesteps = datasets.sparse_pianoroll_to_dense(
        sparse_pianoroll, min_note=4, num_notes=2)
    self.assertEqual(num_timesteps, 6)
    self.assertAllEqual([[1, 1],
                         [1, 1],
                         [0, 0],
                         [0, 1],
                         [0, 0],
                         [0, 0]], dense_pianoroll)

  def test_sparse_pianoroll_to_dense_uses_num_notes(self):
    sparse_pianoroll = [(4, 5), (5, 4), (), (5,), (), ()]
    dense_pianoroll, num_timesteps = datasets.sparse_pianoroll_to_dense(
        sparse_pianoroll, min_note=4, num_notes=3)
    self.assertEqual(num_timesteps, 6)
    self.assertAllEqual([[1, 1, 0],
                         [1, 1, 0],
                         [0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0],
                         [0, 0, 0]], dense_pianoroll)

  def test_pianoroll_dataset(self):
    pianoroll_data = [[(0,), (), (1,)],
                      [(0, 1), (1,)],
                      [(1,), (0,), (), (0, 1), (), ()]]
    pianoroll_mean = np.zeros([3])
    pianoroll_mean[-1] = 1
    data = {"train": pianoroll_data, "train_mean": pianoroll_mean}
    path = os.path.join(tf.test.get_temp_dir(), "test.pkl")
    pickle.dump(data, open(path, "wb"))
    with self.test_session() as sess:
      inputs, targets, lens, mean = datasets.create_pianoroll_dataset(
          path, "train", 2, num_parallel_calls=1,
          shuffle=False, repeat=False,
          min_note=0, max_note=2)
      i1, t1, l1 = sess.run([inputs, targets, lens])
      i2, t2, l2 = sess.run([inputs, targets, lens])
      m = sess.run(mean)
      # Check the lengths.
      self.assertAllEqual([3, 2], l1)
      self.assertAllEqual([6], l2)
      # Check the mean.
      self.assertAllEqual(pianoroll_mean, m)
      # Check the targets. The targets should not be mean-centered and should
      # be padded with zeros to a common length within a batch.
      self.assertAllEqual([[1, 0, 0],
                           [0, 0, 0],
                           [0, 1, 0]], t1[:, 0, :])
      self.assertAllEqual([[1, 1, 0],
                           [0, 1, 0],
                           [0, 0, 0]], t1[:, 1, :])
      self.assertAllEqual([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], t2[:, 0, :])
      # Check the inputs. Each sequence should start with zeros on the first
      # timestep. Each sequence should be padded with zeros to a common length
      # within a batch. The mean should be subtracted from all timesteps except
      # the first and the padding.
      self.assertAllEqual([[0, 0, 0],
                           [1, 0, -1],
                           [0, 0, -1]], i1[:, 0, :])
      self.assertAllEqual([[0, 0, 0],
                           [1, 1, -1],
                           [0, 0, 0]], i1[:, 1, :])
      self.assertAllEqual([[0, 0, 0],
                           [0, 1, -1],
                           [1, 0, -1],
                           [0, 0, -1],
                           [1, 1, -1],
                           [0, 0, -1]], i2[:, 0, :])

  def test_human_pose_dataset(self):
    pose_data = [
        [[0, 0], [2, 2]],
        [[2, 2]],
        [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]],
    ]
    pose_data = [np.array(x, dtype=np.float64) for x in pose_data]
    pose_data_mean = np.array([1, 1], dtype=np.float64)
    data = {
        "train": pose_data,
        "train_mean": pose_data_mean,
    }
    path = os.path.join(tf.test.get_temp_dir(), "test_human_pose_dataset.pkl")
    with open(path, "wb") as out:
      pickle.dump(data, out)
    with self.test_session() as sess:
      inputs, targets, lens, mean = datasets.create_human_pose_dataset(
          path, "train", 2, num_parallel_calls=1, shuffle=False, repeat=False)
      i1, t1, l1 = sess.run([inputs, targets, lens])
      i2, t2, l2 = sess.run([inputs, targets, lens])
      m = sess.run(mean)
      # Check the lengths.
      self.assertAllEqual([2, 1], l1)
      self.assertAllEqual([5], l2)
      # Check the mean.
      self.assertAllEqual(pose_data_mean, m)
      # Check the targets. The targets should not be mean-centered and should
      # be padded with zeros to a common length within a batch.
      self.assertAllEqual([[0, 0], [2, 2]], t1[:, 0, :])
      self.assertAllEqual([[2, 2], [0, 0]], t1[:, 1, :])
      self.assertAllEqual([[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]], t2[:, 0, :])
      # Check the inputs. Each sequence should start with zeros on the first
      # timestep. Each sequence should be padded with zeros to a common length
      # within a batch. The mean should be subtracted from all timesteps except
      # the first and the padding.
      self.assertAllEqual([[0, 0], [-1, -1]], i1[:, 0, :])
      self.assertAllEqual([[0, 0], [0, 0]], i1[:, 1, :])
      self.assertAllEqual([[0, 0], [-1, -1], [-1, -1], [1, 1], [1, 1]],
                          i2[:, 0, :])

  def test_speech_dataset(self):
    with self.test_session() as sess:
      path = os.path.join(
          os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
          "test_data",
          "tiny_speech_dataset.tfrecord")
      inputs, targets, lens = datasets.create_speech_dataset(
          path, 3, samples_per_timestep=2, num_parallel_calls=1,
          prefetch_buffer_size=3, shuffle=False, repeat=False)
      inputs1, targets1, lengths1 = sess.run([inputs, targets, lens])
      inputs2, targets2, lengths2 = sess.run([inputs, targets, lens])
      # Check the lengths.
      self.assertAllEqual([1, 2, 3], lengths1)
      self.assertAllEqual([4], lengths2)
      # Check the targets. The targets should be padded with zeros to a common
      # length within a batch.
      self.assertAllEqual([[[0., 1.], [0., 1.], [0., 1.]],
                           [[0., 0.], [2., 3.], [2., 3.]],
                           [[0., 0.], [0., 0.], [4., 5.]]],
                          targets1)
      self.assertAllEqual([[[0., 1.]],
                           [[2., 3.]],
                           [[4., 5.]],
                           [[6., 7.]]],
                          targets2)
      # Check the inputs. Each sequence should start with zeros on the first
      # timestep. Each sequence should be padded with zeros to a common length
      # within a batch.
      self.assertAllEqual([[[0., 0.], [0., 0.], [0., 0.]],
                           [[0., 0.], [0., 1.], [0., 1.]],
                           [[0., 0.], [0., 0.], [2., 3.]]],
                          inputs1)
      self.assertAllEqual([[[0., 0.]],
                           [[0., 1.]],
                           [[2., 3.]],
                           [[4., 5.]]],
                          inputs2)

  def test_chain_graph_raises_error_on_wrong_steps_per_observation(self):
    with self.assertRaises(ValueError):
      datasets.create_chain_graph_dataset(
          batch_size=4,
          num_timesteps=10,
          steps_per_observation=9)

  def test_chain_graph_single_obs(self):
    with self.test_session() as sess:
      np.random.seed(1234)
      num_observations = 1
      num_timesteps = 5
      batch_size = 2
      state_size = 1
      observations, lengths = datasets.create_chain_graph_dataset(
          batch_size=batch_size,
          num_timesteps=num_timesteps,
          state_size=state_size)
      out_observations, out_lengths = sess.run([observations, lengths])
      self.assertAllEqual([num_observations, num_observations], out_lengths)
      self.assertAllClose(
          [[[1.426677], [-1.789461]]],
          out_observations)

  def test_chain_graph_multiple_obs(self):
    with self.test_session() as sess:
      np.random.seed(1234)
      num_observations = 3
      num_timesteps = 6
      batch_size = 2
      state_size = 1
      observations, lengths = datasets.create_chain_graph_dataset(
          batch_size=batch_size,
          num_timesteps=num_timesteps,
          steps_per_observation=num_timesteps/num_observations,
          state_size=state_size)
      out_observations, out_lengths = sess.run([observations, lengths])
      self.assertAllEqual([num_observations, num_observations], out_lengths)
      self.assertAllClose(
          [[[0.40051451], [1.07405114]],
           [[1.73932898], [3.16880035]],
           [[-1.98377144], [2.82669163]]],
          out_observations)

  def test_chain_graph_state_dims(self):
    with self.test_session() as sess:
      np.random.seed(1234)
      num_observations = 1
      num_timesteps = 5
      batch_size = 2
      state_size = 3
      observations, lengths = datasets.create_chain_graph_dataset(
          batch_size=batch_size,
          num_timesteps=num_timesteps,
          state_size=state_size)
      out_observations, out_lengths = sess.run([observations, lengths])
      self.assertAllEqual([num_observations, num_observations], out_lengths)
      self.assertAllClose(
          [[[1.052287, -4.560759, 3.07988],
            [2.008926, 0.495567, 3.488678]]],
          out_observations)

  def test_chain_graph_fixed_obs(self):
    with self.test_session() as sess:
      np.random.seed(1234)
      num_observations = 3
      num_timesteps = 6
      batch_size = 2
      state_size = 1
      observations, lengths = datasets.create_chain_graph_dataset(
          batch_size=batch_size,
          num_timesteps=num_timesteps,
          steps_per_observation=num_timesteps/num_observations,
          state_size=state_size,
          fixed_observation=4.)
      out_observations, out_lengths = sess.run([observations, lengths])
      self.assertAllEqual([num_observations, num_observations], out_lengths)
      self.assertAllClose(
          np.ones([num_observations, batch_size, state_size]) * 4.,
          out_observations)

if __name__ == "__main__":
  tf.test.main()
