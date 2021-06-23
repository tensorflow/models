#!/bin/bash
#
# Processes the COCO few-shot benchmark into TFRecord files. Requires `wget`.

tmp_dir=$(mktemp -d -t coco-XXXXXXXXXX)
base_image_dir="/tmp/coco_images"
output_dir="/tmp/coco_few_shot"
while getopts ":i:o:" o; do
  case "${o}" in
    o) output_dir=${OPTARG} ;;
    i) base_image_dir=${OPTARG} ;;
    *) echo "Usage: ${0} [-i <base_image_dir>] [-o <output_dir>]" 1>&2; exit 1 ;;
  esac
done

cocosplit_url="dl.yf.io/fs-det/datasets/cocosplit"
wget --recursive --no-parent -q --show-progress --progress=bar:force:noscroll \
    -P "${tmp_dir}" -A "trainvalno5k.json,5k.json,*10shot*.json,*30shot*.json" \
    "http://${cocosplit_url}/"
mv "${tmp_dir}/${cocosplit_url}/"* "${tmp_dir}"
rm -rf "${tmp_dir}/${cocosplit_url}/"

python process_coco_few_shot_json_files.py \
    --logtostderr --workdir="${tmp_dir}"

for seed in {0..9}; do
  for shots in 10 30; do
    python create_coco_tf_record.py \
        --logtostderr \
        --image_dir="${base_image_dir}/train2014" \
        --image_dir="${base_image_dir}/val2014" \
        --image_info_file="${tmp_dir}/${shots}shot_seed${seed}.json" \
        --object_annotations_file="${tmp_dir}/${shots}shot_seed${seed}.json" \
        --caption_annotations_file="" \
        --output_file_prefix="${output_dir}/${shots}shot_seed${seed}" \
        --num_shards=4
  done
done

python create_coco_tf_record.py \
    --logtostderr \
    --image_dir="${base_image_dir}/train2014" \
    --image_dir="${base_image_dir}/val2014" \
    --image_info_file="${tmp_dir}/datasplit/5k.json" \
    --object_annotations_file="${tmp_dir}/datasplit/5k.json" \
    --caption_annotations_file="" \
    --output_file_prefix="${output_dir}/5k" \
    --num_shards=10

python create_coco_tf_record.py \
    --logtostderr \
    --image_dir="${base_image_dir}/train2014" \
    --image_dir="${base_image_dir}/val2014" \
    --image_info_file="${tmp_dir}/datasplit/trainvalno5k_base.json" \
    --object_annotations_file="${tmp_dir}/datasplit/trainvalno5k_base.json" \
    --caption_annotations_file="" \
    --output_file_prefix="${output_dir}/trainvalno5k_base" \
    --num_shards=200

python create_coco_tf_record.py \
    --logtostderr \
    --image_dir="${base_image_dir}/train2014" \
    --image_dir="${base_image_dir}/val2014" \
    --image_info_file="${tmp_dir}/datasplit/5k_base.json" \
    --object_annotations_file="${tmp_dir}/datasplit/5k_base.json" \
    --caption_annotations_file="" \
    --output_file_prefix="${output_dir}/5k_base" \
    --num_shards=10

rm -rf "${tmp_dir}"
