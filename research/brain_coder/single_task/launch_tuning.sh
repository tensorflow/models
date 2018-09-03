#!/bin/bash
# Launches tuning jobs.
# Modify this file to launch workers with your prefered cloud API.
# The following implementation runs each worker as a subprocess on the local
# machine.

MODELS_DIR="/tmp/models"

# Get command line options.
OPTS=$(getopt -n "$0" -o "" --long "job_name:,config:,num_tuners:,num_workers_per_tuner:,num_ps_per_tuner:,max_npe:,num_repetitions:,stop_on_success:,fixed_hparams:,hparam_space_type:" -- "$@")
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

JOB_NAME=""           # Name of the process and the logs directory.
CONFIG=""             # Model and environment hparams.
# NUM_TUNERS: Number of tuning jobs to launch. Each tuning job can train a
# hparam combination. So more tuners means more hparams tried in parallel.
NUM_TUNERS=1
# NUM_WORKERS_PER_TUNER: Number of workers to launch for each tuning job. If
# using neural networks, each worker will be 1 replica.
NUM_WORKERS_PER_TUNER=1
# NUM_PS_PER_TUNER: Number of parameter servers to launch for this tuning job.
# Only set this if using neural networks. For 1 worker per tuner, no parameter
# servers are needed. For more than 1 worker per tuner, at least 1 parameter
# server per tuner is needed to store the global model for each tuner.
NUM_PS_PER_TUNER=0
# MAX_NPE: Maximum number of programs executed. Training will quit once this
# threshold is reached. If 0, the threshold is infinite.
MAX_NPE=0
NUM_REPETITIONS=25    # How many times to run this experiment.
STOP_ON_SUCCESS=true  # Whether to halt training when a solution is found.
# FIXED_HPARAMS: Hold hparams fixed in the grid search. This reduces the search
# space.
FIXED_HPARAMS=""
# HPARAM_SPACE_TYPE: Specifies the hparam search space. See
# `define_tuner_hparam_space` functions defined in pg_train.py and ga_train.py.
HPARAM_SPACE_TYPE="pg"

# Parse options into variables.
while true; do
  case "$1" in
    --job_name ) JOB_NAME="$2"; shift; shift ;;
    --config ) CONFIG="$2"; shift; shift ;;
    --num_tuners ) NUM_TUNERS="$2"; shift; shift ;;
    --num_workers_per_tuner ) NUM_WORKERS_PER_TUNER="$2"; shift; shift ;;
    --num_ps_per_tuner ) NUM_PS_PER_TUNER="$2"; shift; shift ;;
    --max_npe ) MAX_NPE="$2"; shift; shift ;;
    --num_repetitions ) NUM_REPETITIONS="$2"; shift; shift ;;
    --stop_on_success ) STOP_ON_SUCCESS="$2"; shift; shift ;;
    --fixed_hparams ) FIXED_HPARAMS="$2"; shift; shift ;;
    --hparam_space_type ) HPARAM_SPACE_TYPE="$2"; shift; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

# Launch jobs.
# TODO: multi-worker RL training

LOGDIR="$MODELS_DIR/$JOB_NAME"
mkdir -p $LOGDIR

BIN_DIR="bazel-bin/single_task"
for ((tuner=0;tuner<NUM_TUNERS;tuner+=1)); do
  for ((i=0;i<NUM_WORKERS_PER_TUNER;i++)); do
    # Expecting tune.par to be built.
    echo "$LOGDIR"
    $BIN_DIR/tune.par \
        --alsologtostderr \
        --config="$CONFIG" \
        --logdir="$LOGDIR" \
        --max_npe="$MAX_NPE" \
        --num_repetitions="$NUM_REPETITIONS" \
        --stop_on_success="$STOP_ON_SUCCESS" \
        --summary_tasks=1 \
        --hparam_space="$HPARAM_SPACE_TYPE" \
        --fixed_hparams="$FIXED_HPARAMS" \
        --tuner_id=$tuner \
        --num_tuners=$NUM_TUNERS \
        2> "$LOGDIR/tuner_$tuner.task_$i.log" &  # Run as subprocess
    echo "Launched tuner $tuner, task $i. Logs: $LOGDIR/tuner_$tuner.task_$i.log"
  done
done

# Use "pidof tune.par" to find jobs.
# Kill with "pkill tune.par"
