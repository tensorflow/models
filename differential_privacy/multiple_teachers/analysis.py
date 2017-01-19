# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""
This script computes bounds on the privacy cost of training the
student model from noisy aggregation of labels predicted by teachers.
It should be used only after training the student (and therefore the
teachers as well). We however include the label files required to
reproduce key results from our paper (https://arxiv.org/abs/1610.05755):
the epsilon bounds for MNIST and SVHN students.

The command that computes the epsilon bound associated
with the training of the MNIST student model (100 label queries
with a (1/20)*2=0.1 epsilon bound each) is:

python analysis.py
  --counts_file=mnist_250_teachers_labels.npy
  --indices_file=mnist_250_teachers_100_indices_used_by_student.npy

The command that computes the epsilon bound associated
with the training of the SVHN student model (1000 label queries
with a (1/20)*2=0.1 epsilon bound each) is:

python analysis.py
  --counts_file=svhn_250_teachers_labels.npy
  --max_examples=1000
  --delta=1e-6
"""
import os
import math
import numpy as np
import tensorflow as tf

from differential_privacy.multiple_teachers.input import maybe_download

# These parameters can be changed to compute bounds for different failure rates
# or different model predictions.

tf.flags.DEFINE_integer("moments",8, "Number of moments")
tf.flags.DEFINE_float("noise_eps", 0.1, "Eps value for each call to noisymax.")
tf.flags.DEFINE_float("delta", 1e-5, "Target value of delta.")
tf.flags.DEFINE_float("beta", 0.09, "Value of beta for smooth sensitivity")
tf.flags.DEFINE_string("counts_file","","Numpy matrix with raw counts")
tf.flags.DEFINE_string("indices_file","",
    "File containting a numpy matrix with indices used."
    "Optional. Use the first max_examples indices if this is not provided.")
tf.flags.DEFINE_integer("max_examples",1000,
    "Number of examples to use. We will use the first"
    " max_examples many examples from the counts_file"
    " or indices_file to do the privacy cost estimate")
tf.flags.DEFINE_float("too_small", 1e-10, "Small threshold to avoid log of 0")
tf.flags.DEFINE_bool("input_is_counts", False, "False if labels, True if counts")

FLAGS = tf.flags.FLAGS


def compute_q_noisy_max(counts, noise_eps):
  """returns ~ Pr[outcome != winner].

  Args:
    counts: a list of scores
    noise_eps: privacy parameter for noisy_max
  Returns:
    q: the probability that outcome is different from true winner.
  """
  # For noisy max, we only get an upper bound.
  # Pr[ j beats i*] \leq (2+gap(j,i*))/ 4 exp(gap(j,i*)
  # proof at http://mathoverflow.net/questions/66763/
  # tight-bounds-on-probability-of-sum-of-laplace-random-variables

  winner = np.argmax(counts)
  counts_normalized = noise_eps * (counts - counts[winner])
  counts_rest = np.array(
      [counts_normalized[i] for i in xrange(len(counts)) if i != winner])
  q = 0.0
  for c in counts_rest:
    gap = -c
    q += (gap + 2.0) / (4.0 * math.exp(gap))
  return min(q, 1.0 - (1.0/len(counts)))


def compute_q_noisy_max_approx(counts, noise_eps):
  """returns ~ Pr[outcome != winner].

  Args:
    counts: a list of scores
    noise_eps: privacy parameter for noisy_max
  Returns:
    q: the probability that outcome is different from true winner.
  """
  # For noisy max, we only get an upper bound.
  # Pr[ j beats i*] \leq (2+gap(j,i*))/ 4 exp(gap(j,i*)
  # proof at http://mathoverflow.net/questions/66763/
  # tight-bounds-on-probability-of-sum-of-laplace-random-variables
  # This code uses an approximation that is faster and easier
  # to get local sensitivity bound on.

  winner = np.argmax(counts)
  counts_normalized = noise_eps * (counts - counts[winner])
  counts_rest = np.array(
      [counts_normalized[i] for i in xrange(len(counts)) if i != winner])
  gap = -max(counts_rest)
  q = (len(counts) - 1) * (gap + 2.0) / (4.0 * math.exp(gap))
  return min(q, 1.0 - (1.0/len(counts)))


def logmgf_exact(q, priv_eps, l):
  """Computes the logmgf value given q and privacy eps.

  The bound used is the min of three terms. The first term is from
  https://arxiv.org/pdf/1605.02065.pdf.
  The second term is based on the fact that when event has probability (1-q) for
  q close to zero, q can only change by exp(eps), which corresponds to a
  much smaller multiplicative change in (1-q)
  The third term comes directly from the privacy guarantee.
  Args:
    q: pr of non-optimal outcome
    priv_eps: eps parameter for DP
    l: moment to compute.
  Returns:
    Upper bound on logmgf
  """
  if q < 0.5:
    t_one = (1-q) * math.pow((1-q) / (1 - math.exp(priv_eps) * q), l)
    t_two = q * math.exp(priv_eps * l)
    t = t_one + t_two
    try:
      log_t = math.log(t)
    except ValueError:
      print "Got ValueError in math.log for values :" + str((q, priv_eps, l, t))
      log_t = priv_eps * l
  else:
    log_t = priv_eps * l

  return min(0.5 * priv_eps * priv_eps * l * (l + 1), log_t, priv_eps * l)


def logmgf_from_counts(counts, noise_eps, l):
  """
  ReportNoisyMax mechanism with noise_eps with 2*noise_eps-DP
  in our setting where one count can go up by one and another
  can go down by 1.
  """

  q = compute_q_noisy_max(counts, noise_eps)
  return logmgf_exact(q, 2.0 * noise_eps, l)


def sens_at_k(counts, noise_eps, l, k):
  """Return sensitivity at distane k.

  Args:
    counts: an array of scores
    noise_eps: noise parameter used
    l: moment whose sensitivity is being computed
    k: distance
  Returns:
    sensitivity: at distance k
  """
  counts_sorted = sorted(counts, reverse=True)
  if 0.5 * noise_eps * l > 1:
    print "l too large to compute sensitivity"
    return 0
  # Now we can assume that at k, gap remains positive
  # or we have reached the point where logmgf_exact is
  # determined by the first term and ind of q.
  if counts[0] < counts[1] + k:
    return 0
  counts_sorted[0] -= k
  counts_sorted[1] += k
  val = logmgf_from_counts(counts_sorted, noise_eps, l)
  counts_sorted[0] -= 1
  counts_sorted[1] += 1
  val_changed = logmgf_from_counts(counts_sorted, noise_eps, l)
  return val_changed - val


def smoothed_sens(counts, noise_eps, l, beta):
  """Compute beta-smooth sensitivity.

  Args:
    counts: array of scors
    noise_eps: noise parameter
    l: moment of interest
    beta: smoothness parameter
  Returns:
    smooth_sensitivity: a beta smooth upper bound
  """
  k = 0
  smoothed_sensitivity = sens_at_k(counts, noise_eps, l, k)
  while k < max(counts):
    k += 1
    sensitivity_at_k = sens_at_k(counts, noise_eps, l, k)
    smoothed_sensitivity = max(
        smoothed_sensitivity,
        math.exp(-beta * k) * sensitivity_at_k)
    if sensitivity_at_k == 0.0:
      break
  return smoothed_sensitivity


def main(unused_argv):
  ##################################################################
  # If we are reproducing results from paper https://arxiv.org/abs/1610.05755,
  # download the required binaries with label information.
  ##################################################################
  
  # Binaries for MNIST results
  paper_binaries_mnist = \
    ["https://github.com/npapernot/multiple-teachers-for-privacy/blob/master/mnist_250_teachers_labels.npy?raw=true", 
    "https://github.com/npapernot/multiple-teachers-for-privacy/blob/master/mnist_250_teachers_100_indices_used_by_student.npy?raw=true"]
  if FLAGS.counts_file == "mnist_250_teachers_labels.npy" \
    or FLAGS.indices_file == "mnist_250_teachers_100_indices_used_by_student.npy":
    maybe_download(paper_binaries_mnist, os.getcwd())

  # Binaries for SVHN results
  paper_binaries_svhn = ["https://github.com/npapernot/multiple-teachers-for-privacy/blob/master/svhn_250_teachers_labels.npy?raw=true"]
  if FLAGS.counts_file == "svhn_250_teachers_labels.npy":
    maybe_download(paper_binaries_svhn, os.getcwd())

  input_mat = np.load(FLAGS.counts_file)
  if FLAGS.input_is_counts:
    counts_mat = input_mat
  else:
    # In this case, the input is the raw predictions. Transform
    num_teachers, n = input_mat.shape
    counts_mat = np.zeros((n, 10)).astype(np.int32)
    for i in range(n):
      for j in range(num_teachers):
        counts_mat[i, input_mat[j, i]] += 1
  n = counts_mat.shape[0]
  num_examples = min(n, FLAGS.max_examples)

  if not FLAGS.indices_file:
    indices = np.array(range(num_examples))
  else:
    index_list = np.load(FLAGS.indices_file)
    indices = index_list[:num_examples]

  l_list = 1.0 + np.array(xrange(FLAGS.moments))
  beta = FLAGS.beta
  total_log_mgf_nm = np.array([0.0 for _ in l_list])
  total_ss_nm = np.array([0.0 for _ in l_list])
  noise_eps = FLAGS.noise_eps
  
  for i in indices:
    total_log_mgf_nm += np.array(
        [logmgf_from_counts(counts_mat[i], noise_eps, l)
         for l in l_list])
    total_ss_nm += np.array(
        [smoothed_sens(counts_mat[i], noise_eps, l, beta)
         for l in l_list])
  delta = FLAGS.delta

  # We want delta = exp(alpha - eps l).
  # Solving gives eps = (alpha - ln (delta))/l
  eps_list_nm = (total_log_mgf_nm - math.log(delta)) / l_list

  print "Epsilons (Noisy Max): " + str(eps_list_nm)
  print "Smoothed sensitivities (Noisy Max): " + str(total_ss_nm / l_list)

  # If beta < eps / 2 ln (1/delta), then adding noise Lap(1) * 2 SS/eps
  # is eps,delta DP
  # Also if beta < eps / 2(gamma +1), then adding noise 2(gamma+1) SS eta / eps
  # where eta has density proportional to 1 / (1+|z|^gamma) is eps-DP
  # Both from Corolloary 2.4 in
  # http://www.cse.psu.edu/~ads22/pubs/NRS07/NRS07-full-draft-v1.pdf
  # Print the first one's scale
  ss_eps = 2.0 * beta * math.log(1/delta)
  ss_scale = 2.0 / ss_eps
  print "To get an " + str(ss_eps) + "-DP estimate of epsilon, "
  print "..add noise ~ " + str(ss_scale)
  print "... times " + str(total_ss_nm / l_list)
  print "Epsilon = " + str(min(eps_list_nm)) + "."
  if min(eps_list_nm) == eps_list_nm[-1]:
    print "Warning: May not have used enough values of l"

  # Data independent bound, as mechanism is
  # 2*noise_eps DP.
  data_ind_log_mgf = np.array([0.0 for _ in l_list])
  data_ind_log_mgf += num_examples * np.array(
      [logmgf_exact(1.0, 2.0 * noise_eps, l) for l in l_list])

  data_ind_eps_list = (data_ind_log_mgf - math.log(delta)) / l_list
  print "Data independent bound = " + str(min(data_ind_eps_list)) + "."

  return


if __name__ == "__main__":
  tf.app.run()
