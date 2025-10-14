#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -o errexit

# --- Parse command-line arguments ---
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --gcs_path=*) gcs_path="${1#*=}"; shift ;;
    *) echo "❌ Unknown parameter passed: $1"; exit 1 ;;
  esac
done
# --- Check if required arguments were provided ---
if [ -z "$gcs_path" ]; then
  echo "❌ Error: --gcs_path must be specified"
  echo "✅ Usage: ./run_pipeline.sh --gcs_path=/path/to/images"
  exit 1
fi

# --- Run pipeline ---
echo "✅ Activating virtual environment..."
source myenv/bin/activate

echo "🖨️ Copying images files from GCS bucket: $gcs_path"
mkdir -p input_images
gsutil -m cp "$gcs_path"* input_images/

echo "🔎 Extracting objects from images..."
python3 extract_objects.py

echo "🧠 Classifying objects"
python3 classify_images.py

echo "🖨️ Moving predictions back to GCS bucket..."
gsutil -m cp -r predictions/ "$gcs_path"

echo "🧹 Deactivating virtual environment..."
deactivate
echo "✅ Done."
