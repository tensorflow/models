#!/bin/bash
set -e

ROOT_DIR="/tmp/MLPerf_NCF_$(python3 -c 'import uuid; print(str(uuid.uuid4().hex))')"
echo "Root directory: ${ROOT_DIR}"

mkdir -p ${ROOT_DIR}

DATA_DIR="${ROOT_DIR}/movielens_data"
mkdir ${DATA_DIR}
python ../datasets/movielens.py --data_dir ${DATA_DIR} --dataset ml-20m

for i in `seq 0 4`;
do
  START_TIME=$(date +%s)
  MODEL_DIR="${ROOT_DIR}/run_${i}"
  RUN_LOG="${ROOT_DIR}/run_${i}.log"
  echo "Beginning run ${i}"
  python ncf_main.py --model_dir ${MODEL_DIR} --data_dir ${DATA_DIR} \
                     --dataset ml-20m --hooks "" \
                     --clean \
                     --train_epochs 20 \
                     --batch_size 16384 \
                     --learning_rate 0.0005 \
                     --layers 256,256,128,64 --num_factors 64 \
  |& cat > ${RUN_LOG}

  END_TIME=$(date +%s)
  echo "Run ${i} complete: $(( $END_TIME - $START_TIME )) seconds."

  grep -E --regexp="Iteration [0-9]+: HR = [0-9\.]+, NDCG = [0-9\.]+" ${RUN_LOG} | cat

done

#rm -r ${ROOT_DIR}
