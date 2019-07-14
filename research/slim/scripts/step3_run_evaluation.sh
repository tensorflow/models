#!/usr/local/bin/bash

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Convenient script that helps to run evaluation with latest checkpoint.

  --network_type      Can be one of [mobilenet_v1, mobilenet_v2, inception_v1,
                      inception_v2, inception_v3, inception_v4],
                      mobilenet_v1 by default.
  --help              Display this help.
END_OF_USAGE
}

network_type="mobilenet_v1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --network_type)
      network_type=$2
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
cd "../"
image_size="${image_size_map[${network_type}]}"
python eval_image_classifier.py \
  --checkpoint_path="${SLIM_DIR}/${TRAIN_DIR}" \
  --eval_dir="${SLIM_DIR}/${TRAIN_DIR}" \
  --dataset_name="${DATASET_NAME}" \
  --dataset_split_name=validation \
  --dataset_dir="${SLIM_DIR}/${DATASET_DIR}" \
  --model_name="${network_type}" \
  --eval_image_size="${image_size}" \
  --quantize
