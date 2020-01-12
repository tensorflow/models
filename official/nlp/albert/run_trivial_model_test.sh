#!/bin/bash
# Small integration test script.
# The values in this file are **not** meant for reproducing actual results.

set -e
set -x

virtualenv -p python3 .
source ./bin/activate

OUTPUT_DIR_BASE="$(mktemp -d)"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/output"

pip install numpy
pip install -r requirements.txt
python -m run_pretraining_test \
    --output_dir="${OUTPUT_DIR}" \
    --do_train \
    --do_eval \
    --nouse_tpu \
    --train_batch_size=2 \
    --eval_batch_size=1 \
    --max_seq_length=4 \
    --num_train_steps=2 \
    --max_eval_steps=3


