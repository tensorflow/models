#!/bin/bash
# Run minigo... stop when it converges.
set -e

export OMP_NUM_THREADS=56
export KMP_HW_SUBSET=28c,2T
export MN_PREDICTIVE=1

export KMP_BLOCKTIME=1
export KMP_AFFINITY=compact,granularity=fine
export TF_MKL_OPTIMIZE_PRIMITVE_MEMUSE=false
ulimit -u 16384
rm -rf ~/results/minigo/final/
mkdir -p ~/results/minigo/final/
SEED=`date +%s`
echo ":::MLPv0.5.0 minigo `date +%s.%N` (reinforcement/tensorflow/run.sh:$LINENO) run_set_random_seed: $SEED"
cd minigo
bash loop_main.sh params/final.json $SEED
echo "bootstrap black/white win number"
grep B+ ~/results/minigo/final/sgf/000000-bootstrap/full/*.sgf |wc -l
grep W+ ~/results/minigo/final/sgf/000000-bootstrap/full/*.sgf |wc -l
