# Experiments for ICLR 2018 paper.

[Code Synthesis with Priority Queue Training](https://openreview.net/forum?id=r1AoGNlC-).

Runs policy gradient (REINFORCE), priority queue training, genetic algorithm,
and uniform random search.

Run all examples below out of your top-level repo directory, i.e. where your git
clone resides.


## Just tell me how to run something and see results
```bash
# These tasks are the fastest to learn. 'echo' and 'count-down' are very
# easy. run_eval_tasks.py will do most of the work to run all the jobs.
# Should take between 10 and 30 minutes.

# How many repetitions each experiment will run. In the paper, we use 25. Less
# reps means faster experiments, but noisier results.
REPS=25

# Extra description in the job names for these experiments. Use this description
# to distinguish between multiple runs of the same experiment.
DESC="demo"

# The tasks to run.
TASKS="reverse echo-second-seq"

# The model types and max NPE.
EXPS=( pg-20M topk-20M ga-20M rand-20M )

# Where training data is saved. This is chosen by launch_training.sh. Custom
# implementations of launch_training.sh may use different locations.
MODELS_DIR="/tmp/models"

# Run run_eval_tasks.py for each experiment name in EXPS.
for exp in "${EXPS[@]}"
do
  ./single_task/run_eval_tasks.py \
      --exp "$exp" --tasks $TASKS --desc "$DESC" --reps $REPS
done

# During training or after completion, run this to aggregate results into a
# table. This is also useful for seeing how much progress has been made.
# Make sure the arguments here match the settings used above.
# Note: This can take a few minutes because it reads from every experiment
# directory.
bazel run single_task:aggregate_experiment_results -- \
  --models_dir="$MODELS_DIR" \
  --max_npe="20M" \
  --task_list="$TASKS" \
  --model_types="[('pg', '$DESC'), ('topk', '$DESC'), ('ga', '$DESC'),
                  ('rand', '$DESC')]" \
  --csv_file="/tmp/results_table.csv"
```


## Reproduce tuning results in paper
```bash
bazel build -c opt single_task:tune.par

# PG and TopK Tuning.
MAX_NPE=5000000
CONFIG="
env=c(task_cycle=['reverse-tune','remove-tune']),
agent=c(
  algorithm='pg',
  grad_clip_threshold=50.0,param_init_factor=0.5,entropy_beta=0.05,lr=1e-5,
  optimizer='rmsprop',ema_baseline_decay=0.99,topk_loss_hparam=0.0,topk=0,
  replay_temperature=1.0,alpha=0.0,eos_token=False),
timestep_limit=50,batch_size=64"

./single_task/launch_tuning.sh \
    --job_name="iclr_pg_gridsearch.reverse-remove" \
    --config="$CONFIG" \
    --max_npe="$MAX_NPE" \
    --num_workers_per_tuner=1 \
    --num_ps_per_tuner=0 \
    --num_tuners=1 \
    --num_repetitions=50 \
    --hparam_space_type="pg" \
    --stop_on_success=true
./single_task/launch_tuning.sh \
    --job_name="iclr_pg_topk_gridsearch.reverse-remove" \
    --config="$CONFIG" \
    --max_npe="$MAX_NPE" \
    --num_workers_per_tuner=1 \
    --num_ps_per_tuner=0 \
    --num_tuners=1 \
    --num_repetitions=50 \
    --hparam_space_type="pg-topk" \
    --fixed_hparams="topk=10" \
    --stop_on_success=true
./single_task/launch_tuning.sh \
    --job_name="iclr_topk_gridsearch.reverse-remove" \
    --config="$CONFIG" \
    --max_npe="$MAX_NPE" \
    --num_workers_per_tuner=1 \
    --num_ps_per_tuner=0 \
    --num_tuners=1 \
    --num_repetitions=50 \
    --hparam_space_type="topk" \
    --fixed_hparams="topk=10" \
    --stop_on_success=true

# GA Tuning.
CONFIG="
env=c(task_cycle=['reverse-tune','remove-char-tune']),
agent=c(algorithm='ga'),
timestep_limit=50"
./single_task/launch_tuning.sh \
    --job_name="iclr_ga_gridsearch.reverse-remove" \
    --config="$CONFIG" \
    --max_npe="$MAX_NPE" \
    --num_workers_per_tuner=25 \
    --num_ps_per_tuner=0 \
    --num_tuners=1 \
    --num_repetitions=50 \
    --hparam_space_type="ga" \
    --stop_on_success=true

# Aggregate tuning results. Run after tuning jobs complete.
bazel run -c opt single_task:aggregate_tuning_results -- \
    --tuning_dir="$MODELS_DIR/iclr_pg_gridsearch.reverse-remove"
bazel run -c opt single_task:aggregate_tuning_results -- \
    --tuning_dir="$MODELS_DIR/iclr_pg_topk_gridsearch.reverse-remove"
bazel run -c opt single_task:aggregate_tuning_results -- \
    --tuning_dir="$MODELS_DIR/iclr_topk_gridsearch.reverse-remove"
bazel run -c opt single_task:aggregate_tuning_results -- \
    --tuning_dir="$MODELS_DIR/iclr_ga_gridsearch.reverse-remove"
```

## Reproduce eval results in paper
```bash
DESC="v0"  # Description for each experiment. "Version 0" is a good default.
EXPS=( pg-5M topk-5M ga-5M rand-5M pg-20M topk-20M ga-20M rand-20M )
for exp in "${EXPS[@]}"
do
  ./single_task/run_eval_tasks.py \
      --exp "$exp" --iclr_tasks --desc "$DESC"
done
```

## Run single experiment
```bash
EXP="topk-20M"  # Learning algorithm + max-NPE
TASK="reverse"  # Coding task
DESC="v0"  # Description for each experiment. "Version 0" is a good default.
./single_task/run_eval_tasks.py \
    --exp "$EXP" --task "$TASK" --desc "$DESC"
```

## Fetch eval results into a table
```bash
# These arguments should match the settings you used to run the experiments.
MODELS_DIR="/tmp/models"
MAX_NPE="20M"
DESC="v0"  # Same description used in the experiments.
# MODEL_TYPES specifies each model type and the description used in their
# experiments.
MODEL_TYPES="[('pg', '$DESC'), ('topk', '$DESC'),
              ('ga', '$DESC'), ('rand', '$DESC')]"
TASKS=""  # Empty string will default to all ICLR tasks.
# To specify custom task list, give task names separated by spaces. Example:
# TASKS="reverse remove-char"
bazel run single_task:aggregate_experiment_results -- \
    --models_dir="$MODELS_DIR" \
    --max_npe="$MAX_NPE" \
    --task_list="$TASKS" \
    --model_types="$MODEL_TYPES" \
    --csv_file="/tmp/results_table.csv"
```

## Reproduce shortest code examples in paper
```bash
# Maximum NPE is higher here. We only do 1 repetition, and the algorithm needs
# time to simplify its solution.
MODELS_DIR="/tmp/models"
NPE="500M"
DESC="short-code"
./single_task/run_eval_tasks.py \
    --exp "simpl-$NPE" --desc "$DESC" --iclr_tasks --reps 1

# Aggregate best code strings. Run after training completes.
TASKS=""  # Empty string. Will default to all ICLR tasks.
bazel run single_task:aggregate_experiment_results -- \
    --models_dir="$MODELS_DIR" \
    --max_npe="$NPE" \
    --task_list="$TASKS" \
    --model_types="[('topk', '$DESC')]" \
    --data=code
```
