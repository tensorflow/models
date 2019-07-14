#!/usr/local/bin/bash

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Converts TensorFlow checkpoint to EdgeTPU-compatible TFLite file.

  --network_type    Can be one of [mobilenet_v1, mobilenet_v2, inception_v1,
                    inception_v2, inception_v3, inception_v4],
                    mobilenet_v1 by default.
  --checkpoint_num  Checkpoint number, by default 0.
  --help            Display this help.
END_OF_USAGE
}

ckpt_number=0
network_type="mobilenet_v1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_num)
      ckpt_number=$2
      shift 2 ;;
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
mkdir "${SLIM_DIR}/${OUTPUT_DIR}"

echo "Copying label file to ${OUTPUT_DIR} ..."
cp "${SLIM_DIR}/${DATASET_DIR}/labels.txt" "${SLIM_DIR}/${OUTPUT_DIR}"

echo "GETTING frozen graph template ..."
image_size="${image_size_map[${network_type}]}"
python export_inference_graph.py  \
  --alsologtostderr   \
  --model_name="${network_type}"  \
  --image_size="${image_size}"  \
  --output_file="${SLIM_DIR}/${OUTPUT_DIR}/graph_template.pb" \
  --dataset_name=custom \
  --quantize

echo "CONVERTING from checkpoint to frozen graph ..."
output_tensors="${output_tensors_map[${network_type}]}"
freeze_graph \
  --input_graph="${SLIM_DIR}/${OUTPUT_DIR}/graph_template.pb" \
  --input_checkpoint="${SLIM_DIR}/${TRAIN_DIR}/model.ckpt-${ckpt_number}" \
  --input_binary \
  --output_graph="${SLIM_DIR}/${OUTPUT_DIR}/frozen_graph.pb" \
  --output_node_names="${output_tensors}"

echo "CONVERTING from frozen graph to TFLite graph ..."
input_tensors="${input_tensors_map[${network_type}]}"
tflite_convert \
  --output_file="${SLIM_DIR}/${OUTPUT_DIR}/output_tflite_graph.tflite" \
  --graph_def_file="${SLIM_DIR}/${OUTPUT_DIR}/frozen_graph.pb" \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays="${input_tensors}" \
  --output_arrays="${output_tensors}" \
  --mean_values=128 \
  --std_dev_values=128 \
  --input_shapes="1,${image_size},${image_size},3"

echo "TFLite graph generated at ${OUTPUT_DIR}/output_tflite_graph.tflite"
