from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Default configuration for agent and environment."""

from absl import logging

from common import config_lib  # brain coder


def default_config():
  return config_lib.Config(
      agent=config_lib.OneOf(
          [config_lib.Config(
              algorithm='pg',
              policy_lstm_sizes=[35,35],
              # Set value_lstm_sizes to None to share weights with policy.
              value_lstm_sizes=[35,35],
              obs_embedding_size=10,
              grad_clip_threshold=10.0,
              param_init_factor=1.0,
              lr=5e-5,
              pi_loss_hparam=1.0,
              vf_loss_hparam=0.5,
              entropy_beta=1e-2,
              regularizer=0.0,
              softmax_tr=1.0,  # Reciprocal temperature.
              optimizer='rmsprop',  # 'adam', 'sgd', 'rmsprop'
              topk=0,  # Top-k unique codes will be stored.
              topk_loss_hparam=0.0,  # off policy loss multiplier.
              # Uniformly sample this many episodes from topk buffer per batch.
              # If topk is 0, this has no effect.
              topk_batch_size=1,
              # Exponential moving average baseline for REINFORCE.
              # If zero, A2C is used.
              # If non-zero, should be close to 1, like .99, .999, etc.
              ema_baseline_decay=0.99,
              # Whether agent can emit EOS token. If true, agent can emit EOS
              # token which ends the episode early (ends the sequence).
              # If false, agent must emit tokens until the timestep limit is
              # reached. e.g. True means variable length code, False means fixed
              # length code.
              # WARNING: Making this false slows things down.
              eos_token=False,
              replay_temperature=1.0,
              # Replay probability. 1 = always replay, 0 = always on policy.
              alpha=0.0,
              # Whether to normalize importance weights in each minibatch.
              iw_normalize=True),
           config_lib.Config(
              algorithm='ga',
              crossover_rate=0.99,
              mutation_rate=0.086),
           config_lib.Config(
              algorithm='rand')],
          algorithm='pg',
      ),
      env=config_lib.Config(
          # If True, task-specific settings are not needed.
          task='',  # 'print', 'echo', 'reverse', 'remove', ...
          task_cycle=[],  # If non-empty, reptitions will cycle through tasks.
          task_kwargs='{}',  # Python dict literal.
          task_manager_config=config_lib.Config(
              # Reward recieved per test case. These bonuses will be scaled
              # based on how many test cases there are.
              correct_bonus=2.0,  # Bonus for code getting correct answer.
              code_length_bonus=1.0),  # Maximum bonus for short code.
          correct_syntax=False,
      ),
      batch_size=64,
      timestep_limit=32)


def default_config_with_updates(config_string, do_logging=True):
  if do_logging:
    logging.info('Config string: "%s"', config_string)
  config = default_config()
  config.strict_update(config_lib.Config.parse(config_string))
  if do_logging:
    logging.info('Config:\n%s', config.pretty_str())
  return config
