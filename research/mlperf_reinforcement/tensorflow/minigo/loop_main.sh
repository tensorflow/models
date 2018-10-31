#!/bin/bash
#
# We achieve parallelism through multiprocessing for minigo.
# This technique is rather crude, but gives the necessary
# speedup to run the benchmark in a useful length of time.

set -e

SEED=$2

TERMINATE_FILE="TERMINATE_FLAG"
PK_FILE="PK_FLAG"
# HOST_FILE contains all the hosts exist the node that execute this command
HOST_FILE="../hostlist.txt"
WORKER_FILE="workers.txt"
rm -f $TERMINATE_FILE
rm -f $PK_FILE

######################################
# Overview of execution model
# There are 4 stages: 1. selfplay; 2. train; 3. evaluate; 4. old vs. new (P.K.)
# Execution logic is:
# step 1. do selfplay, generate games for training
# step 2. train from selfplay data, generate a new generation model
# step 3. evaluate the new model, see whether termination condition had reached
# step 4. old model versus new model.  If new model win a certain percent of games (55%), next epoch will use new model
#         for selfplay.  Otherwise, new model will be buried, and old model will be used for selfplay for next epoch.
# Dependence wise, Step 2 depends on step 1, step 3 and 4 both depends on step 2.  Step 1 depends on step 4 of previous
# epoch.
# Depending on the number of nodes, and how quickly we can do selfplay by scale out, there are three different execution
# mode, for different number of nodes.
# 1. Single node execution mode
#    In single node mode, step 1,2,4 will be executed sequentially
#    step 3 will be executed in parallel with step 4, and step 1 of next epoch.  Thus the overhead of step 4 can be
#    hidden.
# 2. Multinode non-predictive mode
#    In multinode non-predictive mode, step 1 will be executed on multiple nodes in parallel, thus step 1 time will be
#    reduced as we increase number of nodes for selfplay.  The whole execution flow is similar to single node execution
#    mode, except that step 1 is executing on multiple nodes. a "hostlist.txt" file under "training/reinforcement/tensorflow"
#    directory must be supplied for all the workers for selfplay, include the node running this script.
#    This mode is suitable for situation that multiple nodes can be used for training, but there are not many nodes to
#    reduce selfplay time to be shorter than training time.
# 3. Multinode predictive mode
#    In multinode predictive mode, we overlap step 1 and step 4.  Which means we start selfplay before the result of
#    old vs. new come out.  This means we 'predict' new model will win out.  When old vs. new shows new model is not
#    competent, we will discard these selfplay result and use selfplay data from old model instead.   Instead of run
#    selfplay from old model AFTER we know old model win out, we run selfplay from old model in parallel with training
#    stage of previous epoch.  Thus as soon as we know the old model win out in step 4, we copy these selfplay data
#    from old model directly, which avoid the overhead when new model is not competent.
#
#    As for step 3, it will be run on a dedicated node and overlap with both step 4 and step 1/2 of next epoch.  Which
#    totally hide evaluation cost even when epoch time is very short.
#
#    In multinode predictive mode, we have one machine dedicated for training and old vs. new, one machine dedicated for
#    evaluation, and the rest machine dedicated for selfplay.  In our code, we may also call old vs. new as 'P.K.' for
#    brevetty.
#
#    The execution flow of multinode predictive mode is as follow:
#    Phase 1: selfplay and evaluation and P.K.
#             * In first epoch, run step 1 for new model.  In second epoch and later epoches, run step 1 for new mode, and
#             step 3, 4 from last epoch.
#    Phase 2: train and heuristic selfplay and evaluation
#             * Check result of step 4 from last epoch, that whether new model win out from last epoch.  If new model
#               win out from last epoch, use selfplay data generated in phase 1.  If old model win out from last epoch,
#               use backup selfplay data from phase 2 of last epoch.
#             * Use selfplay data described above do training.
#             * In the mean while use the win out model from phase 1 to generate backup selfplay data.
#             * In the mean while, step 3 from last epoch continues.
#
#    Multinode predictive mode is suitable that selfplay scales out so well that is is equal or shorter than training
#    time+P.K. time, we can benefit with P.K. time hidden.  Use multinode predictive mode when there are lots of nodes.
#
# To select Single node mode, remove "hostlist.txt" file from "training/reinforcement/tensorflow/"
#
# To select Multinode mode.  Put a "hostlist.txt" file in "training/reinforcement/tensorflow/".  The hostlist.txt file
# should contain hostname or ip address of all worker nodes except the node used to run this script.  All worker nodes
# needs to be able to login with SSH without manually supply credentials, and share the same working directory with the
# machine running this script through a shared network storage.
#
# To run multinode non-predictive mode, set MN_PREDICTIVE to 0 in environment variable before run minigo
# > export MN_PREDICTIVE=0
# To run multinode predictive mode, set MN_PREDICTIVE to 1 in environment variable before run minigo
# > export MN_PREDICTIVE=1

echo GOPARAMS=$1
cat $1

GOPARAMS=$1 python3 loop_init.py
#freeze bootstrape model graph
for filename in ~/results/minigo/final/models/*.index;
do
    if [ ! -f ${filename%%.*}.transformed.pb ];
    then
        echo Transform $filename
        ./freeze_graph.sh ${filename%%.*}
    fi
done

if [ -f $HOST_FILE ]; then
    ### multinode path if ../hostlist exists

    if [ $MN_PREDICTIVE -eq 1 ]; then
        echo "run multinode predictive mode with the following hosts"
        cat $HOST_FILE

        cat $HOST_FILE | head --lines=-1 > $WORKER_FILE
        echo "the following nodes are worker node"
        cat $WORKER_FILE

        PID=""
        for i in {0..1000};
        do
            echo "PHASE: selfplay and evaluation and pk -- $i"; SECONDS=0
            while read -r host
            do
                ssh -t $host "ulimit -u 760000; cd `pwd`; GOPARAMS=$1 KMP_BLOCKTIME=1 KMP_HW_SUBSET=$KMP_HW_SUBSET python3 loop_selfplay.py $SEED `date +%N`$i $3 worker 2>&1" &
                PID="$! $PID"
            done < $WORKER_FILE
            # only wait for workers, evaluation thread
            wait $PID
            PID=""
            GOPARAMS=$1 python3 loop_selfplay.py $SEED $i driver 2>&1

            #if old win pk, bury new model and do self play again
            if [ -f $PK_FILE ]; then
                echo "$PK_FILE exists: bury new model and do selfplay again!"
                rm -f $PK_FILE
                GOPARAMS=$1 python3 loop_train_eval.py $SEED $i bury 2>&1
                GOPARAMS=$1 python3 loop_selfplay.py $SEED $i driver 2>&1
            else
                GOPARAMS=$1 python3 loop_selfplay.py $SEED $i clean_backup 2>&1
            fi
            echo "PHASE: $SECONDS seconds"

            if [ -f $TERMINATE_FILE ]; then
                echo "$TERMINATE_FILE exists: finished!"
                cat $TERMINATE_FILE
                break
            else
                echo "$TERMINATE_FILE does not exist; continue."
            fi

            echo "PHASE: train and heuristic selfplay and evaluation -- $i"; SECONDS=0
            while read -r host
            do
                ssh -t $host "ulimit -u 760000; cd `pwd`; GOPARAMS=$1 KMP_BLOCKTIME=1 KMP_HW_SUBSET=$KMP_HW_SUBSET python3 loop_selfplay.py $SEED `date +%N`$i $3 backup 2>&1" &
            done < $WORKER_FILE
            GOPARAMS=$1 OMP_NUM_THREADS=8 python3 loop_train_eval.py $SEED $i train 2>&1
            # transform the trained model into freezed graph
            for filename in ~/results/minigo/final/models/*.index;
            do
                if [ ! -f ${filename%%.*}.transformed.pb ];
                then
                    echo Transform $filename
                    ./freeze_graph.sh ${filename%%.*}
                fi
            done
            wait

            if [ -f $TERMINATE_FILE ]; then
                echo "$TERMINATE_FILE exists: finished!"
                cat $TERMINATE_FILE
                break
            else
                echo "$TERMINATE_FILE does not exist; continue."
            fi

            #both evaluate and PK can be done in parallel with selfplay
            ssh -t `cat $HOST_FILE|tail -n 1` "ulimit -u 760000; cd `pwd`; GOPARAMS=$1 KMP_BLOCKTIME=1 KMP_HW_SUBSET=$KMP_HW_SUBSET OMP_NUM_THREADS=8 python3 loop_train_eval.py $SEED $i eval 2>&1" &
            rm -f $PK_FILE
            GOPARAMS=$1 python3 loop_train_eval.py $SEED $i pk 2>&1 &
            PID=$!
            echo "PHASE: $SECONDS seconds"
        done
        wait
        ### end multinode predictive mode path
    else
        echo "run multinode non-predictive mode with the following hosts"
        cat $HOST_FILE

        for i in {0..1000};
        do
            echo "PHASE: selfplay and evaluate -- $i"; SECONDS=0
            while read -r host
            do
                ssh -t $host "ulimit -u 760000; cd `pwd`; GOPARAMS=$1 KMP_BLOCKTIME=1 KMP_HW_SUBSET=$KMP_HW_SUBSET python3 loop_selfplay.py $SEED `date +%N`$i $3 worker 2>&1" &
            done < $HOST_FILE
            wait
            GOPARAMS=$1 python3 loop_selfplay.py $SEED $i driver 2>&1
            echo "PHASE: $SECONDS seconds"

            if [ -f $TERMINATE_FILE ]; then
                echo "$TERMINATE_FILE exists: finished!"
                cat $TERMINATE_FILE
                break
            else
                echo "$TERMINATE_FILE does not exist; looping again."
            fi

            echo "PHASE: train -- $i"; SECONDS=0
            GOPARAMS=$1 OMP_NUM_THREADS=8 python3 loop_train_eval.py $SEED $i train 2>&1
            # transform the trained model into freezed graph
            for filename in ~/results/minigo/final/models/*.index;
            do
                if [ ! -f ${filename%%.*}.transformed.pb ];
                then
                    echo Transform $filename
                    ./freeze_graph.sh ${filename%%.*}
                fi
            done
            echo "PHASE: $SECONDS seconds"

            #evaluate can be done in parallel with pk and selfplay even in single node mode
            echo "PHASE: evaluate -- $i"; SECONDS=0
            GOPARAMS=$1 OMP_NUM_THREADS=8 python3 loop_train_eval.py $SEED $i eval 2>&1 &
            GOPARAMS=$1 python3 loop_train_eval.py $SEED $i pk 2>&1
            if [ -f $PK_FILE ]; then
                echo "$PK_FILE exists: bury new model."
                rm -f $PK_FILE
                GOPARAMS=$1 python3 loop_train_eval.py $SEED $i bury 2>&1
            fi
            echo "PHASE: $SECONDS seconds"
        done
        wait
        echo "PHASE: $SECONDS seconds"
        ### end multinode non-predictive mode path
    fi

else

    ### single node path
    echo "hostlist.txt not found, run single node mode"
    for i in {0..1000};
    do
        echo "PHASE: selfplay and evaluate -- $i"; SECONDS=0
        GOPARAMS=$1 python3 loop_selfplay.py $SEED `date +%N`$i $3 worker 2>&1
        GOPARAMS=$1 python3 loop_selfplay.py $SEED $i driver 2>&1
        echo "PHASE: $SECONDS seconds"

        echo "PHASE: train -- $i"; SECONDS=0
        GOPARAMS=$1 OMP_NUM_THREADS=8 python3 loop_train_eval.py $SEED $i train 2>&1
        # transform the trained model into freezed graph
        for filename in ~/results/minigo/final/models/*.index;
        do
            if [ ! -f ${filename%%.*}.transformed.pb ];
            then
                echo Transform $filename
                ./freeze_graph.sh ${filename%%.*}
            fi
        done
        echo "PHASE: $SECONDS seconds"

        echo "PHASE: wait eval -- $i"; SECONDS=0
        wait
        echo "PHASE: $SECONDS seconds"
        if [ -f $TERMINATE_FILE ]; then
            echo "$TERMINATE_FILE exists: finished!"
            cat $TERMINATE_FILE
            break
        else
            echo "$TERMINATE_FILE does not exist; looping again."
        fi

        #evaluate can be done in parallel with pk and selfplay even in single node mode
        echo "PHASE: evaluate -- $i"; SECONDS=0
        GOPARAMS=$1 OMP_NUM_THREADS=8 python3 loop_train_eval.py $SEED $i eval 2>&1 &
        GOPARAMS=$1 python3 loop_train_eval.py $SEED $i pk 2>&1
        if [ -f $PK_FILE ]; then
            echo "$PK_FILE exists: bury new model."
            rm -f $PK_FILE
            GOPARAMS=$1 python3 loop_train_eval.py $SEED $i bury 2>&1
        fi
        echo "PHASE: $SECONDS seconds"
    done
    wait
    echo "PHASE: $SECONDS seconds"
    ### end single node path

fi

echo ":::MLPv0.5.0 minigo `date +%s.%N` (reinforcement/tensorflow/minigo/loop_main.sh:$LINENO) run_final"
