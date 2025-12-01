#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -o errexit

# --- Parse command-line arguments ---
batch_size=0 # Default to 0, meaning process all at once
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --gcs_path=*) gcs_path="${1#*=}"; shift ;;
    --batch_size=*) batch_size="${1#*=}"; shift ;;
    *) echo "❌ Unknown parameter passed: $1"; exit 1 ;;
  esac
done

# --- Check if required arguments were provided ---
if [ -z "$gcs_path" ]; then
  echo "❌ Error: --gcs_path must be specified"
  echo "✅ Usage: ./run_pipeline.sh --gcs_path=/path/to/images [--batch_size=N]"
  exit 1
fi

# --- Run pipeline ---
echo "✅ Activating virtual environment..."
source myenv/bin/activate
cd milk_pouch_project

# List the image files in the GCS path.
# NOTE: Adjust the grep pattern if other image types are expected.
echo "🖨️ Listing image files from GCS bucket: $gcs_path"
mapfile -t all_gcs_files < <(gsutil ls "${gcs_path}*" | grep -iE '\.(png)$' | grep -v "/predictions/")
num_files=${#all_gcs_files[@]}

if (( num_files == 0 )); then
  echo "No image files found in $gcs_path. Exiting."
  deactivate
  exit 0
fi

# Create directories if they don't exist
mkdir -p input_images
mkdir -p predictions

# Determine batch size
if (( batch_size <= 0 )); then
  echo "Processing all $num_files files at once."
  batch_size=$num_files
else
  echo "Processing files in batches of $batch_size."
fi

# Iterate through files in batches
for (( i=0; i<num_files; i+=batch_size )); do
  batch_start=$i
  batch_end=$(( i + batch_size ))
  if (( batch_end > num_files )); then
    batch_end=$num_files
  fi

  # Get the current batch of files and calculate the number of files in it.
  current_batch=("${all_gcs_files[@]:batch_start:batch_size}")
  num_in_batch=${#current_batch[@]}
  echo "--- Processing batch $(( i / batch_size + 1 ))/$(( (num_files + batch_size - 1) / batch_size )) ($num_in_batch files) ---"

  # Clear previous batch's inputs and predictions
  echo "🧹 Clearing input_images/ and predictions/..."
  rm -rf input_images/*
  rm -rf predictions/*

  # Copy current batch files from GCS
  echo "🖨️ Copying $num_in_batch files from GCS to input_images/..."
  gsutil -m cp "${current_batch[@]}" input_images/

  # Extract objects
  echo "🔎 Extracting objects from images..."
  python3 extract_objects.py

  # Classify objects
  echo "🧠 Classifying objects..."
  python3 classify_images.py

  # Move predictions back to GCS
  if [ -d "predictions" ] && [ "$(ls -A predictions)" ]; then
    echo "🖨️ Moving predictions for this batch back to GCS bucket: $gcs_path"
    gsutil -m cp -r predictions/ "$gcs_path"
  else
    echo "⚠️ No predictions generated for this batch."
  fi

done

echo "🧹 Deactivating virtual environment..."
deactivate
echo "✅ Done."
