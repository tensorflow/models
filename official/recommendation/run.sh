#!/bin/bash
set -e

DATASET="ml-1m"

ROOT_DIR="/tmp/MLPerf_NCF"
echo "Root directory: ${ROOT_DIR}"

mkdir -p ${ROOT_DIR}

DATA_DIR="${ROOT_DIR}/movielens_data"
#DATA_DIR="gs://taylorrobie-tpu-test-bucket-2/ncf_data_dir"
python ../datasets/movielens.py --data_dir ${DATA_DIR} --dataset ${DATASET}

TEST_DIR="${ROOT_DIR}/`date '+%Y-%m-%d_%H:%M:%S'`"
mkdir -p ${TEST_DIR}

{

for i in `seq 0 0`;
do
  START_TIME=$(date +%s)
  MODEL_DIR="${TEST_DIR}/model_dir_${i}"
#  MODEL_DIR="gs://taylorrobie-tpu-test-bucket-2/${START_TIME}_model_dir_${i}"
  RUN_LOG="${TEST_DIR}/run_${i}.log"
  echo ""
  echo "Beginning run ${i}"
  python ncf_main.py --model_dir ${MODEL_DIR} --data_dir ${DATA_DIR} \
                     --dataset ${DATASET} --hooks "" \
                     --clean \
                     --train_epochs 1 \
                     --batch_size 2048 \
                     --eval_batch_size 65536 \
                     --learning_rate 0.0005 \
                     --layers 256,256,128,64 --num_factors 64 \
                     --hr_threshold 0.635 \
#  |& tee ${RUN_LOG} \
#  | grep --line-buffered  -E --regexp="Iteration [0-9]+: HR = [0-9\.]+, NDCG = [0-9\.]+"

  END_TIME=$(date +%s)
  echo "Run ${i} complete: $(( $END_TIME - $START_TIME )) seconds."

done

} |& tee "${TEST_DIR}/summary.log"
