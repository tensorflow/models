#!/bin/bash
#
# We achieve parallelism through multiprocessing for minigo.
# This technique is rather crude, but gives the necessary
# speedup to run the benchmark in a useful length of time.

set -e

SEED=$2

FILE="TERMINATE_FLAG"
rm -f $FILE

GOPARAMS=$1 python3 loop_init.py
for i in {0..1000};
do
GOPARAMS=$1 python3 loop_selfplay.py $SEED $i 2>&1

GOPARAMS=$1 python3 loop_train_eval.py $SEED $i 2>&1



if [ -f $FILE ]; then
   echo "$FILE exists: finished!"
   cat $FILE
   break
else
   echo "$FILE does not exist; looping again."
fi
done
