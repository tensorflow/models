#!/usr/bin/env bash

# Experiment name, decides the name of the folder to save experiment data to
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

# Eval in background (disable this during debugging)
(sleep 20; CUDA_VISIBLE_DEVICES=${device} \
    python scripts/local_eval.py ${exp_name} ${algo_context} ${env_context} ${agent_context} suite None \
    agent/Context.meta_action_every_n=${act} \
    every_n_episodes.steps_per_episode=${T} \
    every_n_steps.n=${T} \
    evaluate.num_episodes_eval=5) &

# Train
# (Don't forget to put '\' at the end of every line)
CUDA_VISIBLE_DEVICES=${device} \
    python scripts/local_train_v2.py \
    v2 ${exp_name} ${algo_context} ${env_context} ${agent_context} \
    train_uvf.save_policy_every_n_steps=100000 \
    agent/Context.meta_action_every_n=${act} \
    train_uvf.num_collect_per_meta_update=${update} \
    train_uvf.max_steps_per_episode=${T} \
    train_uvf.meta_experience_length=${train} \
    train_uvf.debug=False \
    every_n_episodes.steps_per_episode=${T} \
    every_n_steps.n=${T} \
    train_uvf.dry_no_train=False
