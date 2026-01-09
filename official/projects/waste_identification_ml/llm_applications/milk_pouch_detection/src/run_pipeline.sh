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
echo "=== DEBUGGING START ==="
echo "DEBUG: gcs_path variable is: '${gcs_path}'"
echo "DEBUG: Running 'gsutil ls \"${gcs_path}\"' to check accessibility:"
gsutil ls "${gcs_path}" || echo "❌ gsutil ls failed"
echo "DEBUG: Running 'gsutil ls -r \"${gcs_path}\" | head -n 10' to check content:"
gsutil ls -r "${gcs_path}" | head -n 10 || echo "❌ gsutil recursive ls failed"
echo "=== DEBUGGING END ==="

echo "🖨️ Listing image files from GCS bucket: $gcs_path"
mapfile -t all_gcs_files < <(gsutil ls -r "${gcs_path}" | grep -iE '\.(png|jpg|jpeg)$' | grep -v "/predictions/" | grep -v "/processed/")
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
  if ! python3 extract_objects.py; then
    echo "⚠️ Batch $(( i / batch_size + 1 )) failed during object extraction. Skipping to next batch."
    continue
  fi

  # Classify objects
  echo "🧠 Classifying objects..."
  if ! python3 classify_images.py; then
    echo "⚠️ Batch $(( i / batch_size + 1 )) failed during image classification. Skipping to next batch."
    continue
  fi

  # Move predictions back to GCS
  if [ -d "predictions" ] && [ -n "$(find predictions -type f -print -quit)" ]; then
    echo "🖨️ Moving predictions for this batch back to GCS bucket: $gcs_path"
    gsutil -m cp -r predictions/ "$gcs_path"
  else
    echo "⚠️ No predictions generated for this batch."
  fi

  # --- Move processed input files to 'processed/' directory preserving structure ---

  # Ensure clean_gcs_path ends with / for correct substitution
  clean_gcs_path="$gcs_path"
  [[ "$clean_gcs_path" != */ ]] && clean_gcs_path="$clean_gcs_path/"

  target_root="${clean_gcs_path}processed/"

  # Group files by their destination directory to optimize gsutil calls
  declare -a current_move_batch
  current_move_dir=""

  echo "📦 Moving processed files to ${target_root}..."

  for file_url in "${current_batch[@]}"; do
    # Get the directory of the file (e.g., gs://bucket/dev/2025-12-24/)
    dir_url="$(dirname "$file_url")/"

    # Calculate destination directory by injecting 'processed/'
    # 1. Remove the base gcs_path from the file's dir to get the relative subdir (e.g., 2025-12-24/)
    relative_dir="${dir_url#$clean_gcs_path}"
    # 2. Append this relative dir to the processed root
    dest_dir="${target_root}${relative_dir}"

    # If the destination directory changes, flush the current batch
    if [[ "$dest_dir" != "$current_move_dir" ]]; then
      if (( ${#current_move_batch[@]} > 0 )); then
        gsutil -m mv "${current_move_batch[@]}" "$current_move_dir"
        current_move_batch=()
      fi
      current_move_dir="$dest_dir"
    fi
    current_move_batch+=("$file_url")
  done

  # Flush any remaining files
  if (( ${#current_move_batch[@]} > 0 )); then
    gsutil -m mv "${current_move_batch[@]}" "$current_move_dir"
  fi

  unset current_move_batch

done

echo "🧹 Deactivating virtual environment..."
deactivate
echo "✅ Done."
