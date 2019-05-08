#!/usr/bin/env bash

# Experiment name, decides the name of the folder to load experiment data from
datetime="`date +%Y%m%d_%H%M%S`"

# Algorithm context, choose from [hiro_xy, hiro_orig, hiro_repr]
hiro_style="hiro_xy"

# Environment context, choose from [ant_maze, ant_block, ant_push_single,...]
env="ant_maze"

# Designate Specific GPU
device=0

# Meta-Action Period for Exploration
act=10

CUDA_VISIBLE_DEVICES=${device} \
    python scripts/local_eval.py ${datetime} ${hiro_style} ${env} base_uvf suite "model.ckpt-5000001" \
        evaluate.num_episodes_eval=1 \
        evaluate.generate_videos=True \
        evaluate.num_episodes_videos=1 \
        get_video_settings.fps=20 \
        agent/Context.meta_action_every_n=${act}
