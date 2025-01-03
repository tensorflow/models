#!/bin/bash
# This script downloads the videos for the AVA dataset. There are no arguments.
# Copy this script into the desired parent directory of the ava_vids_raw/
# directory created in this script to store the raw videos.

mkdir ava_vids_raw
cd ava_vids_raw

curl -O s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt

echo "Downloading all videos."

cat "ava_file_names_trainval_v2.1.txt" | while read line
do
  curl -O s3.amazonaws.com/ava-dataset/trainval/$line
  echo "Downloaded " $line
done

rm "ava_file_names_trainval_v2.1.txt"
cd ..

# Trimming causes issues with frame seeking in the python script, so it is best left out.
# If included, need to modify the python script to subtract 900 seconds wheen seeking.

# echo "Trimming all videos."

# mkdir ava_vids_trimmed
# for filename in ava_vids_raw/*; do
#   ffmpeg -ss 900 -to 1800 -i $filename -c copy ava_vids_trimmed/${filename##*/}
# done
