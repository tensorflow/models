#!/bin/bash
#
# Processes the COCO few-shot benchmark into TFRecord files. Requires `wget`.

tmp_dir=$(mktemp -d -t coco-XXXXXXXXXX)
output_dir="/tmp/coco_few_shot"
while getopts "o:" o; do
  case "${o}" in
    o) output_dir=${OPTARG} ;;
    *) echo "Usage: ${0} [-o <output_dir>]" 1>&2; exit 1 ;;
  esac
done

cocosplit_url="dl.yf.io/fs-det/datasets/cocosplit"
wget --recursive --no-parent -q --show-progress --progress=bar:force:noscroll \
    -P "${tmp_dir}" -A "5k.json,*10shot*.json,*30shot*.json" \
    "http://${cocosplit_url}/"
mv "${tmp_dir}/${cocosplit_url}/"* "${tmp_dir}"
rm -rf "${tmp_dir}/${cocosplit_url}/"

python process_coco_few_shot_json_files.py \
    --logtostderr --workdir="${tmp_dir}"

for seed in {0..9}; do
  for shots in 10 30; do
    python create_coco_tf_record.py \
        --logtostderr \
        --image_dir=/namespace/vale-project/datasets/mscoco_raw/images/train2014 \
        --image_dir=/namespace/vale-project/datasets/mscoco_raw/images/val2014 \
        --image_info_file="${tmp_dir}/${shots}shot_seed${seed}.json" \
        --object_annotations_file="${tmp_dir}/${shots}shot_seed${seed}.json" \
        --caption_annotations_file="" \
        --output_file_prefix="${output_dir}/${shots}shot_seed${seed}" \
        --num_shards=4
  done
done

python create_coco_tf_record.py \
    --logtostderr \
    --image_dir=/namespace/vale-project/datasets/mscoco_raw/images/train2014 \
    --image_dir=/namespace/vale-project/datasets/mscoco_raw/images/val2014 \
    --image_info_file="${tmp_dir}/datasplit/5k.json" \
    --object_annotations_file="${tmp_dir}/datasplit/5k.json" \
    --caption_annotations_file="" \
    --output_file_prefix="${output_dir}/5k" \
    --num_shards=10

rm -rf "${tmp_dir}"
