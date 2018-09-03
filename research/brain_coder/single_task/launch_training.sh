#!/bin/bash
# Launches training jobs.
# Modify this file to launch workers with your prefered cloud API.
# The following implementation runs each worker as a subprocess on the local
# machine.

MODELS_DIR="/tmp/models"

# Get command line options.
OPTS=$(getopt -n "$0" -o "" --long "job_name:,config:,num_workers:,num_ps:,max_npe:,num_repetitions:,stop_on_success:" -- "$@")
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

JOB_NAME=""           # Name of the process and the logs directory.
CONFIG=""             # Model and environment hparams.
# NUM_WORKERS: Number of workers to launch for this training job. If using
# neural networks, each worker will be 1 replica.
NUM_WORKERS=1
# NUM_PS: Number of parameter servers to launch for this training job. Only set
# this if using neural networks. For 1 worker, no parameter servers are needed.
# For more than 1 worker, at least 1 parameter server is needed to store the
# global model.
NUM_PS=0
# MAX_NPE: Maximum number of programs executed. Training will quit once this
# threshold is reached. If 0, the threshold is infinite.
MAX_NPE=0
NUM_REPETITIONS=1     # How many times to run this experiment.
STOP_ON_SUCCESS=true  # Whether to halt training when a solution is found.

# Parse options into variables.
while true; do
  case "$1" in
    --job_name ) JOB_NAME="$2"; shift; shift ;;
    --config ) CONFIG="$2"; shift; shift ;;
    --num_workers ) NUM_WORKERS="$2"; shift; shift ;;
    --num_ps ) NUM_PS="$2"; shift; shift ;;
    --max_npe ) MAX_NPE="$2"; shift; shift ;;
    --num_repetitions ) NUM_REPETITIONS="$2"; shift; shift ;;
    --stop_on_success ) STOP_ON_SUCCESS="$2"; shift; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

# Launch jobs.
# TODO: multi-worker RL training

LOGDIR="$MODELS_DIR/$JOB_NAME"
mkdir -p $LOGDIR

BIN_DIR="bazel-bin/single_task"
for (( i=0; i<NUM_WORKERS; i++))
do
  # Expecting run.par to be built.
  $BIN_DIR/run.par \
      --alsologtostderr \
      --config="$CONFIG" \
      --logdir="$LOGDIR" \
      --max_npe="$MAX_NPE" \
      --num_repetitions="$NUM_REPETITIONS" \
      --stop_on_success="$STOP_ON_SUCCESS" \
      --task_id="$i" \
      --num_workers="$NUM_WORKERS" \
      --summary_tasks=1 \
      2> "$LOGDIR/task_$i.log" &  # Run as subprocess
  echo "Launched task $i. Logs: $LOGDIR/task_$i.log"
done


# Use "pidof run.par" to find jobs.
# Kill with "pkill run.par"
