# Copyright 2017 Google Inc. All Rights Reserved.
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
#
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lfads import LFADS
import numpy as np
import os
import tensorflow as tf
import re
import utils
import sys
MAX_INT = sys.maxsize

# Lots of hyperparameters, but most are pretty insensitive.  The
# explanation of these hyperparameters is found below, in the flags
# session.

CHECKPOINT_PB_LOAD_NAME = "checkpoint"
CHECKPOINT_NAME = "lfads_vae"
CSV_LOG = "fitlog"
OUTPUT_FILENAME_STEM = ""
DEVICE = "gpu:0" # "cpu:0", or other gpus, e.g. "gpu:1"
MAX_CKPT_TO_KEEP = 5
MAX_CKPT_TO_KEEP_LVE = 5
PS_NEXAMPLES_TO_PROCESS = MAX_INT # if larger than number of examples, process all
EXT_INPUT_DIM = 0
IC_DIM = 64
FACTORS_DIM = 50
IC_ENC_DIM = 128
GEN_DIM = 200
GEN_CELL_INPUT_WEIGHT_SCALE = 1.0
GEN_CELL_REC_WEIGHT_SCALE = 1.0
CELL_WEIGHT_SCALE = 1.0
BATCH_SIZE = 128
LEARNING_RATE_INIT = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.95
LEARNING_RATE_STOP = 0.00001
LEARNING_RATE_N_TO_COMPARE = 6
INJECT_EXT_INPUT_TO_GEN = False
DO_TRAIN_IO_ONLY = False
DO_TRAIN_ENCODER_ONLY = False
DO_RESET_LEARNING_RATE = False
FEEDBACK_FACTORS_OR_RATES = "factors"
DO_TRAIN_READIN = True

# Calibrated just above the average value for the rnn synthetic data.
MAX_GRAD_NORM = 200.0
CELL_CLIP_VALUE = 5.0
KEEP_PROB = 0.95
TEMPORAL_SPIKE_JITTER_WIDTH = 0
OUTPUT_DISTRIBUTION = 'poisson' # 'poisson' or 'gaussian'
NUM_STEPS_FOR_GEN_IC = MAX_INT # set to num_steps if greater than num_steps

DATA_DIR = "/tmp/rnn_synth_data_v1.0/"
DATA_FILENAME_STEM = "chaotic_rnn_inputs_g1p5"
LFADS_SAVE_DIR = "/tmp/lfads_chaotic_rnn_inputs_g1p5/"
CO_DIM = 1
DO_CAUSAL_CONTROLLER = False
DO_FEED_FACTORS_TO_CONTROLLER = True
CONTROLLER_INPUT_LAG = 1
PRIOR_AR_AUTOCORRELATION = 10.0
PRIOR_AR_PROCESS_VAR = 0.1
DO_TRAIN_PRIOR_AR_ATAU = True
DO_TRAIN_PRIOR_AR_NVAR = True
CI_ENC_DIM = 128
CON_DIM = 128
CO_PRIOR_VAR_SCALE = 0.1
KL_INCREASE_STEPS = 2000
L2_INCREASE_STEPS = 2000
L2_GEN_SCALE = 2000.0
L2_CON_SCALE = 0.0
# scale of regularizer on time correlation of inferred inputs
CO_MEAN_CORR_SCALE = 0.0
KL_IC_WEIGHT = 1.0
KL_CO_WEIGHT = 1.0
KL_START_STEP = 0
L2_START_STEP = 0
IC_PRIOR_VAR_MIN = 0.1
IC_PRIOR_VAR_SCALE = 0.1
IC_PRIOR_VAR_MAX = 0.1
IC_POST_VAR_MIN = 0.0001      # protection from KL blowing up

flags = tf.app.flags
flags.DEFINE_string("kind", "train",
                    "Type of model to build {train, \
                    posterior_sample_and_average, \
                    prior_sample, write_model_params")
flags.DEFINE_string("output_dist", OUTPUT_DISTRIBUTION,
                    "Type of output distribution, 'poisson' or 'gaussian'")
flags.DEFINE_boolean("allow_gpu_growth", False,
                     "If true, only allocate amount of memory needed for \
                     Session. Otherwise, use full GPU memory.")

# DATA
flags.DEFINE_string("data_dir", DATA_DIR, "Data for training")
flags.DEFINE_string("data_filename_stem", DATA_FILENAME_STEM,
                    "Filename stem for data dictionaries.")
flags.DEFINE_string("lfads_save_dir", LFADS_SAVE_DIR, "model save dir")
flags.DEFINE_string("checkpoint_pb_load_name", CHECKPOINT_PB_LOAD_NAME,
                    "Name of checkpoint files, use 'checkpoint_lve' for best \
                    error")
flags.DEFINE_string("checkpoint_name", CHECKPOINT_NAME,
                    "Name of checkpoint files (.ckpt appended)")
flags.DEFINE_string("output_filename_stem", OUTPUT_FILENAME_STEM,
                    "Name of output file (postfix will be added)")
flags.DEFINE_string("device", DEVICE,
                    "Which device to use (default: \"gpu:0\", can also be \
                    \"cpu:0\", \"gpu:1\", etc)")
flags.DEFINE_string("csv_log", CSV_LOG,
                    "Name of file to keep running log of fit likelihoods, \
                    etc (.csv appended)")
flags.DEFINE_integer("max_ckpt_to_keep", MAX_CKPT_TO_KEEP,
                 "Max # of checkpoints to keep (rolling)")
flags.DEFINE_integer("ps_nexamples_to_process", PS_NEXAMPLES_TO_PROCESS,
                 "Number of examples to process for posterior sample and \
                 average (not number of samples to average over).")
flags.DEFINE_integer("max_ckpt_to_keep_lve", MAX_CKPT_TO_KEEP_LVE,
                 "Max # of checkpoints to keep for lowest validation error \
                 models (rolling)")
flags.DEFINE_integer("ext_input_dim", EXT_INPUT_DIM, "Dimension of external \
inputs")
flags.DEFINE_integer("num_steps_for_gen_ic", NUM_STEPS_FOR_GEN_IC,
                     "Number of steps to train the generator initial conditon.")


# If there are observed inputs, there are two ways to add that observed
# input to the model.  The first is by treating as something to be
# inferred, and thus encoding the observed input via the encoders, and then
# input to the generator via the "inferred inputs" channel.  Second, one
# can input the input directly into the generator.  This has the downside
# of making the generation process strictly dependent on knowing the
# observed input for any generated trial.
flags.DEFINE_boolean("inject_ext_input_to_gen",
                     INJECT_EXT_INPUT_TO_GEN,
                     "Should observed inputs be input to model via encoders, \
                     or injected directly into generator?")

# CELL

# The combined recurrent and input weights of the encoder and
# controller cells are by default set to scale at ws/sqrt(#inputs),
# with ws=1.0.  You can change this scaling with this parameter.
flags.DEFINE_float("cell_weight_scale", CELL_WEIGHT_SCALE,
                     "Input scaling for input weights in generator.")


# GENERATION

# Note that the dimension of the initial conditions is separated from the
# dimensions of the generator initial conditions (and a linear matrix will
# adapt the shapes if necessary).  This is just another way to control
# complexity.  In all likelihood, setting the ic dims to the size of the
# generator hidden state is just fine.
flags.DEFINE_integer("ic_dim", IC_DIM, "Dimension of h0")
# Setting the dimensions of the factors to something smaller than the data
# dimension is a way to get a reduced dimensionality representation of your
# data.
flags.DEFINE_integer("factors_dim", FACTORS_DIM,
                     "Number of factors from generator")
flags.DEFINE_integer("ic_enc_dim", IC_ENC_DIM,
                     "Cell hidden size, encoder of h0")

# Controlling the size of the generator is one way to control complexity of
# the dynamics (there is also l2, which will squeeze out unnecessary
# dynamics also).  The modern deep learning approach is to make these cells
# as large as tolerable (from a waiting perspective), and then regularize
# them to death with drop out or whatever.  I don't know if this is correct
# for the LFADS application or not.
flags.DEFINE_integer("gen_dim", GEN_DIM,
                     "Cell hidden size, generator.")
# The weights of the generator cell by default set to scale at
# ws/sqrt(#inputs), with ws=1.0.  You can change ws for
# the input weights or the recurrent weights with these hyperparameters.
flags.DEFINE_float("gen_cell_input_weight_scale", GEN_CELL_INPUT_WEIGHT_SCALE,
                     "Input scaling for input weights in generator.")
flags.DEFINE_float("gen_cell_rec_weight_scale", GEN_CELL_REC_WEIGHT_SCALE,
                     "Input scaling for rec weights in generator.")

# KL DISTRIBUTIONS
# If you don't know what you are donig here, please leave alone, the
# defaults should be fine for most cases, irregardless of other parameters.
#
# If you don't want the prior variance to be learned, set the
# following values to the same thing: ic_prior_var_min,
# ic_prior_var_scale, ic_prior_var_max.  The prior mean will be
# learned regardless.
flags.DEFINE_float("ic_prior_var_min", IC_PRIOR_VAR_MIN,
                   "Minimum variance in posterior h0 codes.")
flags.DEFINE_float("ic_prior_var_scale", IC_PRIOR_VAR_SCALE,
                   "Variance of ic prior distribution")
flags.DEFINE_float("ic_prior_var_max", IC_PRIOR_VAR_MAX,
                   "Maximum variance of IC prior distribution.")
# If you really want to limit the information from encoder to decoder,
# Increase ic_post_var_min above 0.0.
flags.DEFINE_float("ic_post_var_min", IC_POST_VAR_MIN,
                   "Minimum variance of IC posterior distribution.")
flags.DEFINE_float("co_prior_var_scale", CO_PRIOR_VAR_SCALE,
                   "Variance of control input prior distribution.")


flags.DEFINE_float("prior_ar_atau",  PRIOR_AR_AUTOCORRELATION,
                   "Initial autocorrelation of AR(1) priors.")
flags.DEFINE_float("prior_ar_nvar", PRIOR_AR_PROCESS_VAR,
                   "Initial noise variance for AR(1) priors.")
flags.DEFINE_boolean("do_train_prior_ar_atau", DO_TRAIN_PRIOR_AR_ATAU,
                     "Is the value for atau an init, or the constant value?")
flags.DEFINE_boolean("do_train_prior_ar_nvar", DO_TRAIN_PRIOR_AR_NVAR,
                     "Is the value for noise variance an init, or the constant \
                     value?")

# CONTROLLER
# This parameter critically controls whether or not there is a controller
# (along with controller encoders placed into the LFADS graph.  If CO_DIM >
# 1, that means there is a 1 dimensional controller outputs, if equal to 0,
# then no controller.
flags.DEFINE_integer("co_dim", CO_DIM,
    "Number of control net outputs (>0 builds that graph).")

# The controller will be more powerful if it can see the encoding of the entire
# trial.  However, this allows the controller to create inferred inputs that are
# acausal with respect to the actual data generation process.  E.g. the data
# generator could have an input at time t, but the controller, after seeing the
# entirety of the trial could infer that the input is coming a little before
# time t, because there are no restrictions on the data the controller sees.
# One can force the controller to be causal (with respect to perturbations in
# the data generator) so that it only sees forward encodings of the data at time
# t that originate at times before or at time t.  One can also control the data
# the controller sees by using an input lag (forward encoding at time [t-tlag]
# for controller input at time t.  The same can be done in the reverse direction
# (controller input at time t from reverse encoding at time [t+tlag], in the
# case of an acausal controller).  Setting this lag > 0 (even lag=1) can be a
# powerful way of avoiding very spiky decodes. Finally, one can manually control
# whether the factors at time t-1 are fed to the controller at time t.
#
# If you don't care about any of this, and just want to smooth your data, set
#    do_causal_controller = False
#    do_feed_factors_to_controller = True
#    causal_input_lag = 0
flags.DEFINE_boolean("do_causal_controller",
                     DO_CAUSAL_CONTROLLER,
                     "Restrict the controller create only causal inferred \
                     inputs?")
# Strictly speaking, feeding either the factors or the rates to the controller
# violates causality, since the g0 gets to see all the data. This may or may not
# be only a theoretical concern.
flags.DEFINE_boolean("do_feed_factors_to_controller",
                     DO_FEED_FACTORS_TO_CONTROLLER,
                     "Should factors[t-1] be input to controller at time t?")
flags.DEFINE_string("feedback_factors_or_rates", FEEDBACK_FACTORS_OR_RATES,
                    "Feedback the factors or the rates to the controller? \
                     Acceptable values: 'factors' or 'rates'.")
flags.DEFINE_integer("controller_input_lag", CONTROLLER_INPUT_LAG,
                     "Time lag on the encoding to controller t-lag for \
                     forward, t+lag for reverse.")

flags.DEFINE_integer("ci_enc_dim", CI_ENC_DIM,
                     "Cell hidden size, encoder of control inputs")
flags.DEFINE_integer("con_dim", CON_DIM,
                     "Cell hidden size, controller")


# OPTIMIZATION
flags.DEFINE_integer("batch_size", BATCH_SIZE,
                     "Batch size to use during training.")
flags.DEFINE_float("learning_rate_init", LEARNING_RATE_INIT,
                   "Learning rate initial value")
flags.DEFINE_float("learning_rate_decay_factor", LEARNING_RATE_DECAY_FACTOR,
                   "Learning rate decay, decay by this fraction every so \
                   often.")
flags.DEFINE_float("learning_rate_stop", LEARNING_RATE_STOP,
                   "The lr is adaptively reduced, stop training at this value.")
# Rather put the learning rate on an exponentially decreasiong schedule,
# the current algorithm pays attention to the learning rate, and if it
# isn't regularly decreasing, it will decrease the learning rate.  So far,
# it works fine, though it is not perfect.
flags.DEFINE_integer("learning_rate_n_to_compare", LEARNING_RATE_N_TO_COMPARE,
                     "Number of previous costs current cost has to be worse \
                     than, to lower learning rate.")

# This sets a value, above which, the gradients will be clipped.  This hp
# is extremely useful to avoid an infrequent, but highly pathological
# problem whereby the gradient is so large that it destroys the
# optimziation by setting parameters too large, leading to a vicious cycle
# that ends in NaNs.  If it's too large, it's useless, if it's too small,
# it essentially becomes the learning rate.  It's pretty insensitive, though.
flags.DEFINE_float("max_grad_norm", MAX_GRAD_NORM,
                   "Max norm of gradient before clipping.")

# If your optimizations start "NaN-ing out", reduce this value so that
# the values of the network don't grow out of control.  Typically, once
# this parameter is set to a reasonable value, one stops having numerical
# problems.
flags.DEFINE_float("cell_clip_value", CELL_CLIP_VALUE,
                   "Max value recurrent cell can take before being clipped.")

# This flag is used for an experiment where one sees if training a model with
# many days data can be used to learn the dynamics from a held-out days data.
# If you don't care about that particular experiment, this flag should always be
# false.
flags.DEFINE_boolean("do_train_io_only", DO_TRAIN_IO_ONLY,
                     "Train only the input (readin) and output (readout) \
                     affine functions.")

# This flag is used for an experiment where one wants to know if the dynamics
# learned by the generator generalize across conditions. In that case, you might
# train up a model on one set of data, and then only further train the encoder on 
# another set of data (the conditions to be tested) so that the model is forced
# to use the same dynamics to describe that data.
# If you don't care about that particular experiment, this flag should always be
# false.
flags.DEFINE_boolean("do_train_encoder_only", DO_TRAIN_ENCODER_ONLY,
                     "Train only the encoder weights.")

flags.DEFINE_boolean("do_reset_learning_rate", DO_RESET_LEARNING_RATE,
                     "Reset the learning rate to initial value.")


# for multi-session "stitching" models, the per-session readin matrices map from
# neurons to input factors which are fed into the shared encoder. These are
# initialized by alignment_matrix_cxf and alignment_bias_c in the input .h5
# files. They can be fixed or made trainable.
flags.DEFINE_boolean("do_train_readin", DO_TRAIN_READIN, "Whether to train the \
                     readin matrices and bias vectors. False leaves them fixed \
                     at their initial values specified by the alignment \
                     matrices and vectors.")


# OVERFITTING
# Dropout is done on the input data, on controller inputs (from
# encoder), on outputs from generator to factors.
flags.DEFINE_float("keep_prob", KEEP_PROB, "Dropout keep probability.")
# It appears that the system will happily fit spikes (blessing or
# curse, depending).  You may not want this.  Jittering the spikes a
# bit will help (-/+ bin size, as specified here).
flags.DEFINE_integer("temporal_spike_jitter_width",
                     TEMPORAL_SPIKE_JITTER_WIDTH,
                     "Shuffle spikes around this window.")

# General note about helping ascribe controller inputs vs dynamics:
#
# If controller is heavily penalized, then it won't have any output.
# If dynamics are heavily penalized, then generator won't make
# dynamics.  Note this l2 penalty is only on the recurrent portion of
# the RNNs, as dropout is also available, penalizing the feed-forward
# connections.
flags.DEFINE_float("l2_gen_scale", L2_GEN_SCALE,
                   "L2 regularization cost for the generator only.")
flags.DEFINE_float("l2_con_scale", L2_CON_SCALE,
                   "L2 regularization cost for the controller only.")
flags.DEFINE_float("co_mean_corr_scale", CO_MEAN_CORR_SCALE,
                   "Cost of correlation (thru time)in the means of \
                   controller output.")

# UNDERFITTING
# If the primary task of LFADS is "filtering" of data and not
# generation, then it is possible that the KL penalty is too strong.
# Empirically, we have found this to be the case.  So we add a
# hyperparameter in front of the the two KL terms (one for the initial
# conditions to the generator, the other for the controller outputs).
# You should always think of the the default values as 1.0, and that
# leads to a standard VAE formulation whereby the numbers that are
# optimized are a lower-bound on the log-likelihood of the data. When
# these 2 HPs deviate from 1.0, one cannot make any statement about
# what those LL lower bounds mean anymore, and they cannot be compared
# (AFAIK).
flags.DEFINE_float("kl_ic_weight", KL_IC_WEIGHT,
                   "Strength of KL weight on initial conditions KL penatly.")
flags.DEFINE_float("kl_co_weight", KL_CO_WEIGHT,
                   "Strength of KL weight on controller output KL penalty.")

# Sometimes the task can be sufficiently hard to learn that the
# optimizer takes the 'easy route', and simply minimizes the KL
# divergence, setting it to near zero, and the optimization gets
# stuck.  These two parameters will help avoid that by by getting the
# optimization to 'latch' on to the main optimization, and only
# turning in the regularizers later.
flags.DEFINE_integer("kl_start_step", KL_START_STEP,
                     "Start increasing weight after this many steps.")
# training passes, not epochs, increase by 0.5 every kl_increase_steps
flags.DEFINE_integer("kl_increase_steps", KL_INCREASE_STEPS,
                     "Increase weight of kl cost to avoid local minimum.")
# Same story for l2 regularizer.  One wants a simple generator, for scientific
# reasons, but not at the expense of hosing the optimization.
flags.DEFINE_integer("l2_start_step", L2_START_STEP,
                     "Start increasing l2 weight after this many steps.")
flags.DEFINE_integer("l2_increase_steps", L2_INCREASE_STEPS,
                     "Increase weight of l2 cost to avoid local minimum.")

FLAGS = flags.FLAGS


def build_model(hps, kind="train", datasets=None):
  """Builds a model from either random initialization, or saved parameters.

  Args:
    hps: The hyper parameters for the model.
    kind: (optional) The kind of model to build.  Training vs inference require
      different graphs.
    datasets: The datasets structure (see top of lfads.py).

  Returns:
    an LFADS model.
  """

  build_kind = kind
  if build_kind == "write_model_params":
    build_kind = "train"
  with tf.variable_scope("LFADS", reuse=None):
    model = LFADS(hps, kind=build_kind, datasets=datasets)

  if not os.path.exists(hps.lfads_save_dir):
    print("Save directory %s does not exist, creating it." % hps.lfads_save_dir)
    os.makedirs(hps.lfads_save_dir)

  cp_pb_ln = hps.checkpoint_pb_load_name
  cp_pb_ln = 'checkpoint' if cp_pb_ln == "" else cp_pb_ln
  if cp_pb_ln == 'checkpoint':
    print("Loading latest training checkpoint in: ", hps.lfads_save_dir)
    saver = model.seso_saver
  elif cp_pb_ln == 'checkpoint_lve':
    print("Loading lowest validation checkpoint in: ", hps.lfads_save_dir)
    saver = model.lve_saver
  else:
    print("Loading checkpoint: ", cp_pb_ln, ", in: ", hps.lfads_save_dir)
    saver = model.seso_saver

  ckpt = tf.train.get_checkpoint_state(hps.lfads_save_dir,
                                       latest_filename=cp_pb_ln)

  session = tf.get_default_session()
  print("ckpt: ", ckpt)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    if kind in ["posterior_sample_and_average", "prior_sample",
                "write_model_params"]:
      print("Possible error!!! You are running ", kind, " on a newly \
      initialized model!")
      # cant print ckpt.model_check_point path if no ckpt
      print("Are you sure you sure a checkpoint in ", hps.lfads_save_dir,
            " exists?")

    tf.global_variables_initializer().run()

  if ckpt:
    train_step_str = re.search('-[0-9]+$', ckpt.model_checkpoint_path).group()
  else:
    train_step_str = '-0'

  fname = 'hyperparameters' + train_step_str + '.txt'
  hp_fname = os.path.join(hps.lfads_save_dir, fname)
  hps_for_saving = jsonify_dict(hps)
  utils.write_data(hp_fname, hps_for_saving, use_json=True)

  return model


def jsonify_dict(d):
  """Turns python booleans into strings so hps dict can be written in json.
  Creates a shallow-copied dictionary first, then accomplishes string
  conversion.

  Args:
    d: hyperparameter dictionary

  Returns: hyperparameter dictionary with bool's as strings
  """

  d2 = d.copy()   # shallow copy is fine by assumption of d being shallow
  def jsonify_bool(boolean_value):
    if boolean_value:
      return "true"
    else:
      return "false"

  for key in d2.keys():
    if isinstance(d2[key], bool):
      d2[key] = jsonify_bool(d2[key])
  return d2


def build_hyperparameter_dict(flags):
  """Simple script for saving hyper parameters.  Under the hood the
  flags structure isn't a dictionary, so it has to be simplified since we
  want to be able to view file as text.

  Args:
    flags: From tf.app.flags

  Returns:
    dictionary of hyper parameters (ignoring other flag types).
  """
  d = {}
  # Data
  d['output_dist'] = flags.output_dist
  d['data_dir'] = flags.data_dir
  d['lfads_save_dir'] = flags.lfads_save_dir
  d['checkpoint_pb_load_name'] = flags.checkpoint_pb_load_name
  d['checkpoint_name'] = flags.checkpoint_name
  d['output_filename_stem'] = flags.output_filename_stem
  d['max_ckpt_to_keep'] = flags.max_ckpt_to_keep
  d['max_ckpt_to_keep_lve'] = flags.max_ckpt_to_keep_lve
  d['ps_nexamples_to_process'] = flags.ps_nexamples_to_process
  d['ext_input_dim'] = flags.ext_input_dim
  d['data_filename_stem'] = flags.data_filename_stem
  d['device'] = flags.device
  d['csv_log'] = flags.csv_log
  d['num_steps_for_gen_ic'] = flags.num_steps_for_gen_ic
  d['inject_ext_input_to_gen'] = flags.inject_ext_input_to_gen
  # Cell
  d['cell_weight_scale'] = flags.cell_weight_scale
  # Generation
  d['ic_dim'] = flags.ic_dim
  d['factors_dim'] = flags.factors_dim
  d['ic_enc_dim'] = flags.ic_enc_dim
  d['gen_dim'] = flags.gen_dim
  d['gen_cell_input_weight_scale'] = flags.gen_cell_input_weight_scale
  d['gen_cell_rec_weight_scale'] = flags.gen_cell_rec_weight_scale
  # KL distributions
  d['ic_prior_var_min'] = flags.ic_prior_var_min
  d['ic_prior_var_scale'] = flags.ic_prior_var_scale
  d['ic_prior_var_max'] = flags.ic_prior_var_max
  d['ic_post_var_min'] = flags.ic_post_var_min
  d['co_prior_var_scale'] = flags.co_prior_var_scale
  d['prior_ar_atau'] = flags.prior_ar_atau
  d['prior_ar_nvar'] =  flags.prior_ar_nvar
  d['do_train_prior_ar_atau'] = flags.do_train_prior_ar_atau
  d['do_train_prior_ar_nvar'] = flags.do_train_prior_ar_nvar
  # Controller
  d['do_causal_controller'] = flags.do_causal_controller
  d['controller_input_lag'] = flags.controller_input_lag
  d['do_feed_factors_to_controller'] = flags.do_feed_factors_to_controller
  d['feedback_factors_or_rates'] = flags.feedback_factors_or_rates
  d['co_dim'] = flags.co_dim
  d['ci_enc_dim'] = flags.ci_enc_dim
  d['con_dim'] = flags.con_dim
  d['co_mean_corr_scale'] = flags.co_mean_corr_scale
  # Optimization
  d['batch_size'] = flags.batch_size
  d['learning_rate_init'] = flags.learning_rate_init
  d['learning_rate_decay_factor'] = flags.learning_rate_decay_factor
  d['learning_rate_stop'] = flags.learning_rate_stop
  d['learning_rate_n_to_compare'] = flags.learning_rate_n_to_compare
  d['max_grad_norm'] = flags.max_grad_norm
  d['cell_clip_value'] = flags.cell_clip_value
  d['do_train_io_only'] = flags.do_train_io_only
  d['do_train_encoder_only'] = flags.do_train_encoder_only
  d['do_reset_learning_rate'] = flags.do_reset_learning_rate
  d['do_train_readin'] = flags.do_train_readin

  # Overfitting
  d['keep_prob'] = flags.keep_prob
  d['temporal_spike_jitter_width'] = flags.temporal_spike_jitter_width
  d['l2_gen_scale'] = flags.l2_gen_scale
  d['l2_con_scale'] = flags.l2_con_scale
  # Underfitting
  d['kl_ic_weight'] = flags.kl_ic_weight
  d['kl_co_weight'] = flags.kl_co_weight
  d['kl_start_step'] = flags.kl_start_step
  d['kl_increase_steps'] = flags.kl_increase_steps
  d['l2_start_step'] = flags.l2_start_step
  d['l2_increase_steps'] = flags.l2_increase_steps

  return d


class hps_dict_to_obj(dict):
  """Helper class allowing us to access hps dictionary more easily."""

  def __getattr__(self, key):
    if key in self:
      return self[key]
    else:
      assert False, ("%s does not exist." % key)
  def __setattr__(self, key, value):
    self[key] = value


def train(hps, datasets):
  """Train the LFADS model.

  Args:
    hps: The dictionary of hyperparameters.
    datasets: A dictionary of data dictionaries.  The dataset dict is simply a
      name(string)-> data dictionary mapping (See top of lfads.py).
  """
  model = build_model(hps, kind="train", datasets=datasets)
  if hps.do_reset_learning_rate:
    sess = tf.get_default_session()
    sess.run(model.learning_rate.initializer)

  model.train_model(datasets)


def write_model_runs(hps, datasets, output_fname=None):
  """Run the model on the data in data_dict, and save the computed values.

  LFADS generates a number of outputs for each examples, and these are all
  saved.  They are:
    The mean and variance of the prior of g0.
    The mean and variance of approximate posterior of g0.
    The control inputs (if enabled)
    The initial conditions, g0, for all examples.
    The generator states for all time.
    The factors for all time.
    The rates for all time.

  Args:
    hps: The dictionary of hyperparameters.
    datasets: A dictionary of data dictionaries.  The dataset dict is simply a
      name(string)-> data dictionary mapping (See top of lfads.py).
    output_fname (optional): output filename stem to write the model runs.
  """
  model = build_model(hps, kind=hps.kind, datasets=datasets)
  model.write_model_runs(datasets, output_fname)


def write_model_samples(hps, datasets, dataset_name=None, output_fname=None):
  """Use the prior distribution to generate samples from the model.
  Generates batch_size number of samples (set through FLAGS).

  LFADS generates a number of outputs for each examples, and these are all
  saved.  They are:
    The mean and variance of the prior of g0.
    The control inputs (if enabled)
    The initial conditions, g0, for all examples.
    The generator states for all time.
    The factors for all time.
    The output distribution parameters (e.g. rates) for all time.

  Args:
    hps: The dictionary of hyperparameters.
    datasets: A dictionary of data dictionaries.  The dataset dict is simply a
      name(string)-> data dictionary mapping (See top of lfads.py).
    dataset_name: The name of the dataset to grab the factors -> rates
      alignment matrices from. Only a concern with models trained on
      multi-session data. By default, uses the first dataset in the data dict.
    output_fname: The name prefix of the file in which to save the generated
      samples.
  """
  if not output_fname:
    output_fname = "model_runs_" + hps.kind
  else:
    output_fname = output_fname + "model_runs_" + hps.kind
  if not dataset_name:
    dataset_name = datasets.keys()[0]
  else:
    if dataset_name not in datasets.keys():
      raise ValueError("Invalid dataset name '%s'."%(dataset_name))
  model = build_model(hps, kind=hps.kind, datasets=datasets)
  model.write_model_samples(dataset_name, output_fname)


def write_model_parameters(hps, output_fname=None, datasets=None):
  """Save all the model parameters

  Save all the parameters to hps.lfads_save_dir.

  Args:
    hps: The dictionary of hyperparameters.
    output_fname: The prefix of the file in which to save the generated
      samples.
    datasets: A dictionary of data dictionaries.  The dataset dict is simply a
      name(string)-> data dictionary mapping (See top of lfads.py).
  """
  if not output_fname:
    output_fname = "model_params"
  else:
    output_fname = output_fname + "_model_params"
  fname = os.path.join(hps.lfads_save_dir, output_fname)
  print("Writing model parameters to: ", fname)
  # save the optimizer params as well
  model = build_model(hps, kind="write_model_params", datasets=datasets)
  model_params = model.eval_model_parameters(use_nested=False,
                                             include_strs="LFADS")
  utils.write_data(fname, model_params, compression=None)
  print("Done.")


def clean_data_dict(data_dict):
  """Add some key/value pairs to the data dict, if they are missing.
  Args:
    data_dict - dictionary containing data for LFADS
  Returns:
    data_dict with some keys filled in, if they are absent.
  """

  keys = ['train_truth', 'train_ext_input', 'valid_data',
          'valid_truth', 'valid_ext_input', 'valid_train']
  for k in keys:
    if k not in data_dict:
      data_dict[k] = None

  return data_dict


def load_datasets(data_dir, data_filename_stem):
  """Load the datasets from a specified directory.

  Example files look like
    >data_dir/my_dataset_first_day
    >data_dir/my_dataset_second_day

  If my_dataset (filename) stem is in the directory, the read routine will try
  and load it.  The datasets dictionary will then look like
  dataset['first_day'] -> (first day data dictionary)
  dataset['second_day'] -> (first day data dictionary)

  Args:
    data_dir: The directory from which to load the datasets.
    data_filename_stem: The stem of the filename for the datasets.

  Returns:
    datasets: a dataset dictionary, with one name->data dictionary pair for
    each dataset file.
  """
  print("Reading data from ", data_dir)
  datasets = utils.read_datasets(data_dir, data_filename_stem)
  for k, data_dict in datasets.items():
    datasets[k] = clean_data_dict(data_dict)

    train_total_size = len(data_dict['train_data'])
    if train_total_size == 0:
      print("Did not load training set.")
    else:
      print("Found training set with number examples: ", train_total_size)

    valid_total_size = len(data_dict['valid_data'])
    if valid_total_size == 0:
      print("Did not load validation set.")
    else:
      print("Found validation set with number examples: ", valid_total_size)

  return datasets


def main(_):
  """Get this whole shindig off the ground."""
  d = build_hyperparameter_dict(FLAGS)
  hps = hps_dict_to_obj(d)    # hyper parameters
  kind = FLAGS.kind

  # Read the data, if necessary.
  train_set = valid_set = None
  if kind in ["train", "posterior_sample_and_average", "prior_sample",
              "write_model_params"]:
    datasets = load_datasets(hps.data_dir, hps.data_filename_stem)
  else:
    raise ValueError('Kind {} is not supported.'.format(kind))

  # infer the dataset names and dataset dimensions from the loaded files
  hps.kind = kind     # needs to be added here, cuz not saved as hyperparam
  hps.dataset_names = []
  hps.dataset_dims = {}
  for key in datasets:
    hps.dataset_names.append(key)
    hps.dataset_dims[key] = datasets[key]['data_dim']

  # also store down the dimensionality of the data
  # - just pull from one set, required to be same for all sets
  hps.num_steps = datasets.values()[0]['num_steps']
  hps.ndatasets = len(hps.dataset_names)

  if hps.num_steps_for_gen_ic > hps.num_steps:
    hps.num_steps_for_gen_ic = hps.num_steps

  # Build and run the model, for varying purposes.
  config = tf.ConfigProto(allow_soft_placement=True,
                          log_device_placement=False)
  if FLAGS.allow_gpu_growth:
    config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  with sess.as_default():
    with tf.device(hps.device):
      if kind == "train":
        train(hps, datasets)
      elif kind == "posterior_sample_and_average":
        write_model_runs(hps, datasets, hps.output_filename_stem)
      elif kind == "prior_sample":
        write_model_samples(hps, datasets, hps.output_filename_stem)
      elif kind == "write_model_params":
        write_model_parameters(hps, hps.output_filename_stem, datasets)
      else:
        assert False, ("Kind %s is not implemented. " % kind)


if __name__ == "__main__":
    tf.app.run()
