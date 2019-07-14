#!/usr/local/bin/bash

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Starts training.

  --network_type      Can be one of [mobilenet_v1, mobilenet_v2, inception_v1,
                      inception_v2, inception_v3, inception_v4],
                      mobilenet_v1 by default.
  --train_whole_model Whether or not to train all layers of the model. false
                      by default, in which only the last few layers are trained.
  --train_steps       Number of steps to train the model.
  --quantize_delay    Start quantize training after training for quantize_delay
                       steps
  --help              Display this help.
END_OF_USAGE
}

network_type="mobilenet_v1"
train_whole_model="false"
train_steps=300
quantize_delay=100
while [[ $# -gt 0 ]]; do
  case "$1" in
    --network_type)
      network_type=$2
      shift 2 ;;
    --train_whole_model)
      train_whole_model="true"
      shift 2 ;;
    --train_steps)
      train_steps=$2
      shift 2 ;;
    --quantize_delay)
      quantize_delay=$2
      shift 2 ;;
    --help)
      usage
      exit 0 ;;
    --*)
      echo "Unknown flag $1"
      usage
      exit 1 ;;
  esac
done

source "$PWD/constants.sh"

mkdir "${TRAIN_DIR}"
image_size="${image_size_map[${network_type}]}"
ckpt_name="${ckpt_name_map[${network_type}]}"

# Because the pre-trained checkpoints are based on ImageNet, which has 1000
# labels, and your custom dataset usually has a different number of labels
# (which is true with the flowers dataset), you should exclude the last layer
# when loading the pre-trained checkpoint. This can be specified through
# checkpoint_exclude_scopes flag.
scopes="${scopes_map[${network_type}]}"
cd "../"
if [[ "${train_whole_model}" == "true" ]]; then
  echo "TRAINING all layers ..."
  python train_image_classifier.py \
    --train_dir="${SLIM_DIR}/${TRAIN_DIR}" \
    --dataset_name="${DATASET_NAME}" \
    --dataset_split_name=train \
    --dataset_dir="${SLIM_DIR}/${DATASET_DIR}" \
    --model_name="${network_type}" \
    --checkpoint_path="${SLIM_DIR}/${CKPT_DIR}/${ckpt_name}.ckpt" \
    --max_number_of_steps="${train_steps}" \
    --batch_size=10 \
    --learning_rate=0.01 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=60 \
    --save_summaries_secs=60 \
    --log_every_n_steps=20 \
    --optimizer=sgd \
    --weight_decay=0.00004 \
    --quantize_delay="${quantize_delay}" \
    --clone_on_cpu \
    --train_image_size="${image_size}" \
    --checkpoint_exclude_scopes="${scopes}"
else
  echo "TRAINING last few layers ..."
  # If you only want to retrain only the last few layers of the model, you need
  # to specify the trainable scopes trainable_scopes, to demonstrate this, we
  # will limit the trainable scope to Fully-connected layers. You can definitely
  # add other layers by tweaking this flag.
  python train_image_classifier.py \
    --train_dir="${SLIM_DIR}/${TRAIN_DIR}" \
    --dataset_name="${DATASET_NAME}" \
    --dataset_split_name=train \
    --dataset_dir="${SLIM_DIR}/${DATASET_DIR}" \
    --model_name="${network_type}" \
    --checkpoint_path="${SLIM_DIR}/${CKPT_DIR}/${ckpt_name}.ckpt" \
    --max_number_of_steps="${train_steps}" \
    --batch_size=100 \
    --learning_rate=0.03 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=60 \
    --save_summaries_secs=60 \
    --log_every_n_steps=20 \
    --optimizer=sgd \
    --weight_decay=0.00005 \
    --quantize_delay="${quantize_delay}" \
    --clone_on_cpu \
    --train_image_size="${image_size}" \
    --checkpoint_exclude_scopes="${scopes}" \
    --trainable_scopes="${scopes}"
fi
