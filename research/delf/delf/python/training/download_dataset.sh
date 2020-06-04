#!/bin/bash

# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This script downloads the Google Landmarks v2 dataset. To download the dataset
# run the script like in the following example:
#   bash download_dataset.sh 500 100 20
# 
# The script takes the following parameters, in order:
# - number of image files from the TRAIN split to download (maximum 500)
# - number of image files from the INDEX split to download (maximum 100)
# - number of image files from the TEST split to download (maximum 20)

image_files_train=$1 # Number of image files to download from the TRAIN split
image_files_index=$2 # Number of image files to download from the INDEX split
image_files_test=$3 # Number of image files to download from the TEST split

splits=("train" "test" "index")
dataset_root_folder=gldv2_dataset

metadata_url="https://s3.amazonaws.com/google-landmark/metadata"
ground_truth_url="https://s3.amazonaws.com/google-landmark/ground_truth"
csv_train=(${metadata_url}/train.csv ${metadata_url}/train_clean.csv ${metadata_url}/train_attribution.csv ${metadata_url}/train_label_to_category.csv)
csv_index=(${metadata_url}/index.csv ${metadata_url}/index_image_to_landmark.csv ${metadata_url}/index_label_to_category.csv)
csv_test=(${metadata_url}/test.csv ${ground_truth_url}/recognition_solution_v2.1.csv ${ground_truth_url}/retrieval_solution_v2.1.csv)

images_tar_file_base_url="https://s3.amazonaws.com/google-landmark"
images_md5_file_base_url="https://s3.amazonaws.com/google-landmark/md5sum"
num_processes=6

make_folder() {
  # Creates a folder and checks if it exists. Exits if folder creation fails.
  local folder=$1
  if [ -d "${folder}" ]; then
    echo "Folder ${folder} already exists. Skipping folder creation."
  else
    echo "Creating folder ${folder}."
    if mkdir ${folder}; then
      echo "Successfully created folder ${folder}."
    else
      echo "Failed to create folder ${folder}. Exiting."
      exit 1
    fi
  fi
}

download_file() {
  # Downloads a file from an URL into a specified folder.
  local file_url=$1
  local folder=$2
  local file_path="${folder}/`basename ${file_url}`"
  echo "Downloading file ${file_url} to folder ${folder}."
  pushd . > /dev/null
  cd ${folder}
  curl -Os ${file_url}
  popd > /dev/null
}

validate_md5_checksum() {
  # Validate the MD5 checksum of a downloaded file.
  local content_file=$1
  local md5_file=$2
  echo "Checking MD5 checksum of file ${content_file} against ${md5_file}"
  if [[ "${OSTYPE}" == "linux-gnu" ]]; then
    content_md5=`md5sum ${content_file}`
  elif [[ "${OSTYPE}" == "darwin"* ]]; then
    content_md5=`md5 -r "${content_file}"`
  fi
  content_md5=`cut -d' ' -f1<<<"${content_md5}"`
  expected_md5=`cut -d' ' -f1<<<cat "${md5_file}"`
  if [[ "$content_md5" != "" && "$content_md5" = "$expected_md5" ]]; then
    echo "Check passed."
  else
    echo "Check failed. MD5 checksums don't match. Exiting."
    exit 1
  fi
}

extract_tar_file() {
  # Extracts the content of a tar file to a specified folder.
  local tar_file=$1
  local folder=$2
  echo "Extracting file ${tar_file} to folder ${folder}"
  tar -C ${folder} -xf ${tar_file}
}

download_image_file() {
  # Downloads one image file of a split and untar it.
  local split=$1
  local idx=`printf "%03g" $2`
  local split_folder=$3
  local images_tar_file=images_${idx}.tar
  local images_tar_file_url=${images_tar_file_base_url}/${split}/${images_tar_file}
  local images_tar_file_path=${split_folder}/${images_tar_file}
  local images_md5_file=md5.images_${idx}.txt
  local images_md5_file_url=${images_md5_file_base_url}/${split}/${images_md5_file}
  local images_md5_file_path=${split_folder}/${images_md5_file}
  download_file "${images_tar_file_url}" "${split_folder}"
  download_file "${images_md5_file_url}" "${split_folder}"
  validate_md5_checksum "${images_tar_file_path}" "${images_md5_file_path}"
  extract_tar_file "${images_tar_file_path}" "${split_folder}"
}

download_image_files() {
  # Downloads all image files of a split and untars them.
  local split=$1
  local split_folder=$2
  local image_files="image_files_${split}"
  local max_idx=$(expr ${!image_files} - 1)
  echo "Downloading ${!image_files} files form the split ${split} in the folder ${split_folder}."
  for i in $(seq 0 ${num_processes} ${max_idx}); do
    local curr_max_idx=$(expr ${i} + ${num_processes} - 1)
    local last_idx=$((${curr_max_idx}>${max_idx}?${max_idx}:${curr_max_idx}))
    for j in $(seq ${i} 1 ${last_idx}); do download_image_file "${split}" "${j}" "${split_folder}" & done
    wait
  done
}

download_csv_files() {
  # Downloads all medatada CSV files of a split.
  local split=$1
  local split_folder=$2
  local csv_list="csv_${split}[*]"
  for csv_file in ${!csv_list}; do
    download_file "${csv_file}" "${split_folder}"
  done
}

download_split() {
  # Downloads all artifacts, metadata CSV files and image files of a single split.
  local split=$1
  local split_folder=${dataset_root_folder}/${split}
  make_folder "${split_folder}"
  download_csv_files "${split}" "${split_folder}"
  download_image_files "${split}" "${split_folder}"
}

download_all_splits() {
  # Downloads all artifacts, metadata CSV files and image files of all splits.
  make_folder "${dataset_root_folder}"
  for split in "${splits[@]}"; do
    download_split "$split"
  done
}

download_all_splits

exit 0
