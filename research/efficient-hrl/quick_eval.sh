#!/usr/bin/env bash
# Experiment name, decides the name of the folder to load experiment data from
exp_name="`date +%Y%m%d_%H%M%S`"

# Algorithm context, choose from [hiro_xy, hiro_orig, hiro_repr]
algo_context="hiro_xy"

# Environment context, choose from [ant_maze, ant_block, ant_push_single,...]
env_context="ant_maze"

agent_context="base_uvf"

# Designate Specific GPU
device=0

# Meta-Action Period for Exploration
act=10

# Meta-Action Period for Training
train=10

# Other parameters
update=10
T=501

# Checkpoints to evaluate. Format: begin_end
checkpoints="5000000_9000000"

# Number of episodes to evaluate per checkpoint
num_episodes_eval=5



CUDA_VISIBLE_DEVICES=${device} \
    python scripts/local_eval.py ${exp_name} ${algo_context} ${env_context} ${agent_context} suite ${checkpoints} \
    agent/Context.meta_action_every_n=${act} \
    evaluate.num_episodes_eval=${num_episodes_eval} \
    every_n_episodes.steps_per_episode=${T} \
    every_n_steps.n=${T}
